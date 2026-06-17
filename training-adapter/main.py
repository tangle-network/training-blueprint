"""
Universal Training Adapter

One server, all methods, best backend auto-selected.

Customer sends: model + method + dataset + hyperparams
Adapter picks the fastest available backend that supports that method.
Operator just installs backends: `pip install unsloth trl`

Backend priority per method:
  LoRA/QLoRA/SFT → unsloth (2x faster) → TRL → torchtune
  DPO/GRPO       → unsloth (if available) → TRL
  Full fine-tune  → TRL → torchtune
  Reward modeling → TRL only

Endpoints:
  POST /v1/train/init         — load model + configure
  POST /v1/train/step         — run N steps, return loss + grads
  POST /v1/train/momentum     — get/set optimizer state (DeMo sync)
  POST /v1/train/checkpoint   — save checkpoint with hash
  POST /v1/train/load         — resume from checkpoint
  POST /eval_held_out         — per-example held-out losses for base vs candidate
  GET  /v1/train/status       — step, loss, GPU memory
  GET  /v1/train/capabilities — what methods + models this server supports
  GET  /health                — liveness
"""

import os
import io
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training-adapter")

app = FastAPI(title="Training Adapter", version="2.0.0")

# ---------------------------------------------------------------------------
# Detect available backends at startup
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS: dict[str, bool] = {}


def detect_backends():
    for name, pkg in [
        ("unsloth", "unsloth"),
        ("trl", "trl"),
        ("torchtune", "torchtune"),
    ]:
        try:
            __import__(pkg)
            AVAILABLE_BACKENDS[name] = True
            logger.info(f"Backend available: {name}")
        except ImportError:
            AVAILABLE_BACKENDS[name] = False

    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
        if has_gpu:
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {props.name}, {props.total_mem // (1024**2)} MB")
    except ImportError:
        pass

    AVAILABLE_BACKENDS["gpu"] = has_gpu


detect_backends()

# Best backend per method (first available wins)
METHOD_BACKEND_PRIORITY: dict[str, list[str]] = {
    "sft": ["unsloth", "trl", "torchtune"],
    "lora": ["unsloth", "trl", "torchtune"],
    "qlora": ["unsloth", "trl"],
    "full": ["trl", "torchtune", "unsloth"],
    "dpo": ["unsloth", "trl"],
    "grpo": ["unsloth", "trl"],
    "reward": ["trl"],
}


def pick_backend(method: str) -> str:
    forced = os.environ.get("TRAINING_BACKEND")
    if forced:
        forced = forced.lower()
        if AVAILABLE_BACKENDS.get(forced):
            return forced
        raise RuntimeError(
            f"TRAINING_BACKEND={forced} is not available. "
            f"Available: {[k for k, v in AVAILABLE_BACKENDS.items() if v and k != 'gpu']}"
        )

    priority = METHOD_BACKEND_PRIORITY.get(method, ["trl"])
    for name in priority:
        if AVAILABLE_BACKENDS.get(name):
            return name
    raise RuntimeError(
        f"No backend available for method '{method}'. Install: pip install unsloth trl"
    )


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class InitRequest(BaseModel):
    base_model: str
    method: str = "lora"
    dataset_url: Optional[str] = None
    dataset_format: str = "chat"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    load_in_4bit: bool = True
    beta: float = 0.1
    num_generations: int = 4
    shard_start: Optional[int] = None
    shard_end: Optional[int] = None
    demo_top_k_ratio: float = 0.001


class StepRequest(BaseModel):
    num_steps: int = 1
    return_gradient_norms: bool = False


class CheckpointRequest(BaseModel):
    path: str
    save_merged: bool = False


class MomentumRequest(BaseModel):
    action: str = "get"


class SparseUpdate(BaseModel):
    indices: list[int]
    values: list[float]
    shape: list[int]
    step: int
    peer_id: str = ""


class DemoStepRequest(BaseModel):
    num_steps: int = 1


class DemoStepResponse(BaseModel):
    updates: list[SparseUpdate]
    loss: float
    steps_completed: int
    total_steps: int


class DemoApplySyncRequest(BaseModel):
    peer_updates: list[list[SparseUpdate]]


class EvalHeldOutRequest(BaseModel):
    base_model: str
    max_examples: Optional[int] = Field(default=None)


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class TrainingState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.config: Optional[InitRequest] = None
        self.backend_name: str = ""
        self.step: int = 0
        self.last_loss: float = 0.0
        self.start_time: float = 0.0
        self.tokens_processed: int = 0
        self.demo_baseline: Optional[dict] = None

    def init_unsloth(self, config: InitRequest):
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit and config.method in ("qlora", "lora"),
        )

        if config.method in ("lora", "qlora", "sft"):
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )

        dataset = self._load_dataset(config)

        if config.method in ("sft", "lora", "qlora", "full"):
            train_config = SFTConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                max_steps=config.max_steps,
                warmup_steps=config.warmup_steps,
                lr_scheduler_type=config.lr_scheduler,
                weight_decay=config.weight_decay,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                max_seq_length=config.max_seq_length,
            )
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=train_config,
            )
        elif config.method == "dpo":
            dpo_config = DPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                beta=config.beta,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            self.trainer = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=dpo_config,
            )
        elif config.method == "grpo":
            from trl import GRPOTrainer, GRPOConfig

            grpo_config = GRPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                num_generations=config.num_generations,
                beta=config.beta,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            self.trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=grpo_config,
            )

    def init_trl(self, config: InitRequest):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
        import torch

        quant_config = None
        if config.load_in_4bit and config.method in ("qlora", "lora"):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if config.method in ("sft", "lora", "qlora"):
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)

        dataset = self._load_dataset(config)

        if config.method in ("sft", "lora", "qlora", "full"):
            train_config = SFTConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                max_steps=config.max_steps,
                warmup_steps=config.warmup_steps,
                lr_scheduler_type=config.lr_scheduler,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
                bf16=False,
                fp16=False,
                max_length=config.max_seq_length,
            )
            self.trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=dataset,
                args=train_config,
            )
        elif config.method == "dpo":
            dpo_config = DPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                beta=config.beta,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            self.trainer = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=dpo_config,
            )
        elif config.method == "grpo":
            from trl import GRPOTrainer, GRPOConfig

            grpo_config = GRPOConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                num_generations=config.num_generations,
                beta=config.beta,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            self.trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=grpo_config,
            )
        elif config.method == "reward":
            from trl import RewardTrainer, RewardConfig

            reward_config = RewardConfig(
                output_dir="./output",
                per_device_train_batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                num_train_epochs=config.num_epochs,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            self.trainer = RewardTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=reward_config,
            )

    def _load_dataset(self, config: InitRequest):
        if not config.dataset_url:
            return None
        from datasets import load_dataset

        url = config.dataset_url

        # Local file:// URL (absolute path)
        if url.startswith("file://"):
            from urllib.parse import urlparse
            from pathlib import Path

            path = urlparse(url).path
            ext = Path(path).suffix.lower().lstrip(".")
            fmt = {
                "jsonl": "json",
                "json": "json",
                "csv": "csv",
                "parquet": "parquet",
            }.get(ext, "json")
            ds = load_dataset(fmt, data_files=path, split="train")

        # S3/GCS/R2 — HuggingFace datasets handles these via fsspec
        elif (
            url.startswith("s3://")
            or url.startswith("gs://")
            or url.startswith("r2://")
        ):
            # Requires: pip install s3fs gcsfs
            # Auth via env: AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS, etc.
            ext = url.rsplit(".", 1)[-1].lower()
            fmt = {
                "jsonl": "json",
                "json": "json",
                "csv": "csv",
                "parquet": "parquet",
            }.get(ext, "json")
            ds = load_dataset(fmt, data_files=url, split="train")

        # HTTP/HTTPS URL
        elif url.startswith("http"):
            ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
            fmt = {
                "jsonl": "json",
                "json": "json",
                "csv": "csv",
                "parquet": "parquet",
            }.get(ext, "json")
            ds = load_dataset(fmt, data_files=url, split="train")

        # HuggingFace Hub dataset name (e.g. "trl-lib/Capybara")
        else:
            ds = load_dataset(url, split="train")

        # Shard the dataset when running as one operator in a distributed job.
        start = config.shard_start if config.shard_start is not None else 0
        end = config.shard_end if config.shard_end is not None else len(ds)
        if start > 0 or end < len(ds):
            ds = ds.select(range(start, min(end, len(ds))))

        # Ensure a plain-text field exists for the TRL SFTTrainer.
        if config.dataset_format == "chat" and "text" not in ds.column_names:
            ds = ds.map(lambda ex: {"text": _extract_text(ex)})
        return ds

    def train_steps(self, num_steps: int, return_norms: bool = False) -> dict:
        import torch

        if not self.trainer:
            raise RuntimeError("Trainer not initialized")

        self.trainer.args.max_steps = self.step + num_steps
        train_result = self.trainer.train(resume_from_checkpoint=False)

        self.step += num_steps
        self.last_loss = train_result.training_loss
        self.tokens_processed += (
            num_steps
            * (self.config.batch_size if self.config else 1)
            * (self.config.max_seq_length if self.config else 2048)
        )

        gpu_mem = (
            torch.cuda.memory_allocated() // (1024**2)
            if torch.cuda.is_available()
            else 0
        )
        lr = (
            self.trainer.optimizer.param_groups[0]["lr"]
            if self.trainer.optimizer
            else 0.0
        )

        result = {
            "steps_completed": num_steps,
            "total_steps": self.step,
            "loss": self.last_loss,
            "learning_rate": lr,
            "gpu_memory_used_mb": gpu_mem,
            "tokens_processed": self.tokens_processed,
        }

        if return_norms and self.model:
            norms = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    norms.append(float(param.grad.norm().item()))
            result["gradient_norms"] = norms[:20]  # first 20 layers

        return result

    def get_momentum(self) -> bytes:
        import torch

        if not self.trainer or not self.trainer.optimizer:
            return b""
        buf = io.BytesIO()
        torch.save(self.trainer.optimizer.state_dict(), buf)
        return buf.getvalue()

    def set_momentum(self, data: bytes):
        import torch

        if not self.trainer or not self.trainer.optimizer:
            return
        state = torch.load(io.BytesIO(data), weights_only=False)
        self.trainer.optimizer.load_state_dict(state)

    def save_checkpoint(self, path: str, merge: bool = False):
        os.makedirs(path, exist_ok=True)
        if merge and hasattr(self.model, "save_pretrained_merged"):
            self.model.save_pretrained_merged(path, self.tokenizer)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def get_gpu_info(self) -> tuple[int, int]:
        try:
            import torch

            if torch.cuda.is_available():
                return (
                    torch.cuda.memory_allocated() // (1024**2),
                    torch.cuda.get_device_properties(0).total_mem // (1024**2),
                )
        except Exception:
            pass
        return (0, 0)


def _extract_text(example: dict) -> str:
    """Best-effort text extraction from a dataset example.

    Handles plain-text fields and common chat formats. Returns empty string only
    when no usable text is found; callers should reject empty examples rather
    than emit NaN losses.
    """
    # Plain-text fields.
    for key in ("text", "content", "message", "prompt", "raw"):
        if key in example and isinstance(example[key], str):
            return example[key]

    # Chat formats: {"messages": [{"role": ..., "content": ...}, ...]}
    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for msg in example["messages"]:
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("value") or msg.get("text")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "\n".join(parts)

    # Conversations: {"conversations": [{"from": ..., "value": ...}, ...]}
    if "conversations" in example and isinstance(example["conversations"], list):
        parts = []
        for turn in example["conversations"]:
            if isinstance(turn, dict):
                content = turn.get("value") or turn.get("content") or turn.get("text")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "\n".join(parts)

    # Fallback: join all top-level scalar fields.
    parts = [str(v) for v in example.values() if isinstance(v, (str, int, float))]
    return "\n".join(parts)


def _load_held_out_dataset(max_examples: Optional[int] = None):
    """Load the private held-out validation split used for certification.

    The held-out split must be separate from training data. Operators configure the
    URL via HELD_OUT_DATASET_URL and the split name via HELD_OUT_DATASET_SPLIT
    (default: 'validation') so the split stays private to the verifier running this
    adapter. There is no silent fallback to 'train'; if the configured split does not
    exist, the endpoint fails closed with a clear error.
    """
    from datasets import load_dataset

    url = os.environ.get("HELD_OUT_DATASET_URL")
    if not url:
        raise RuntimeError(
            "HELD_OUT_DATASET_URL not set. "
            "Configure a private held-out dataset for certification."
        )

    if url.startswith("file://"):
        from urllib.parse import urlparse

        path = urlparse(url).path
        ext = path.rsplit(".", 1)[-1].lower()
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(
            ext, "json"
        )
        ds = load_dataset(fmt, data_files=path)
    elif url.startswith("s3://") or url.startswith("gs://") or url.startswith("r2://"):
        ext = url.rsplit(".", 1)[-1].lower()
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(
            ext, "json"
        )
        ds = load_dataset(fmt, data_files=url)
    elif url.startswith("http"):
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(
            ext, "json"
        )
        ds = load_dataset(fmt, data_files=url)
    elif Path(url).is_file():
        # Local file path (jsonl, json, csv, parquet).
        ext = url.rsplit(".", 1)[-1].lower()
        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(
            ext, "json"
        )
        ds = load_dataset(fmt, data_files=url)
    else:
        ds = load_dataset(url)

    split = os.environ.get("HELD_OUT_DATASET_SPLIT", "validation")
    if split not in ds:
        raise RuntimeError(
            f"Held-out split '{split}' not found in dataset. "
            f"Available splits: {list(ds.keys())}. "
            f"Set HELD_OUT_DATASET_SPLIT to a split that is NOT your training data."
        )
    ds = ds[split]
    if max_examples is not None and max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


def _load_base_model_for_eval(base_model: str, config: InitRequest):
    """Load the original pretrained model (no PEFT) for held-out comparison.

    The caller must use state.tokenizer for tokenization so base and candidate are
    evaluated on identical token IDs. Memory usage is the caller's responsibility;
    if VRAM is tight, run on a machine that can hold both models or use a smaller
    base.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    kwargs = {
        "quantization_config": quant_config,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model.eval()
    return model


def _compute_per_example_losses(
    model, tokenizer, dataset, max_length: int, batch_size: int = 1
):
    """Compute average cross-entropy loss per held-out example.

    Returns one finite float per valid dataset example. Examples that tokenize to
    only padding are skipped (not emitted as NaN), because serde_json rejects NaN
    and the eval gate treats non-finite losses as uncertified.
    """
    import math
    import torch

    device = next(model.parameters()).device
    losses: list[float] = []
    skipped = 0
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
                texts = [_extract_text(ex) for ex in batch]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                labels = enc["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(**enc)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                per_token = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                per_token = per_token.view(shift_labels.size())

                for b in range(per_token.size(0)):
                    valid = shift_labels[b] != -100
                    example_loss = per_token[b][valid]
                    if example_loss.numel() > 0:
                        val = float(example_loss.mean().cpu())
                        if math.isfinite(val):
                            losses.append(val)
                        else:
                            skipped += 1
                    else:
                        skipped += 1
    finally:
        if was_training:
            model.train()

    if skipped:
        logger.warning(f"Skipped {skipped} held-out example(s) with no/finite loss")
    return losses


def _clear_gpu_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


state = TrainingState()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "backends": {k: v for k, v in AVAILABLE_BACKENDS.items()}}


@app.get("/v1/train/capabilities")
def capabilities():
    """What this server can do — the customer checks this before submitting."""
    methods = {}
    for method, backends in METHOD_BACKEND_PRIORITY.items():
        available = [b for b in backends if AVAILABLE_BACKENDS.get(b)]
        if available:
            methods[method] = {"backend": available[0], "all_backends": available}
    return {
        "methods": methods,
        "gpu": AVAILABLE_BACKENDS.get("gpu", False),
        "backends": {k: v for k, v in AVAILABLE_BACKENDS.items() if k != "gpu"},
    }


@app.post("/v1/train/init")
def init_training(req: InitRequest):
    backend_name = pick_backend(req.method)
    state.config = req
    state.backend_name = backend_name
    state.step = 0
    state.last_loss = 0.0
    state.start_time = time.time()
    state.tokens_processed = 0

    try:
        if backend_name == "unsloth":
            state.init_unsloth(req)
        elif backend_name == "trl":
            state.init_trl(req)
        else:
            raise RuntimeError(f"Backend {backend_name} init not implemented")

        logger.info(
            f"Initialized: method={req.method}, backend={backend_name}, model={req.base_model}"
        )
        return {
            "status": "initialized",
            "backend": backend_name,
            "model": req.base_model,
            "method": req.method,
        }
    except Exception as e:
        logger.exception("Init failed")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/v1/train/step")
def train_step(req: StepRequest):
    if not state.trainer:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        return state.train_steps(req.num_steps, req.return_gradient_norms)
    except Exception as e:
        logger.exception("Training step failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/train/momentum")
def handle_momentum(req: MomentumRequest, request: Request):
    import torch

    if not state.trainer:
        raise HTTPException(status_code=400, detail="Not initialized")
    if req.action == "get":
        buf = io.BytesIO()
        torch.save(state.trainer.optimizer.state_dict(), buf)
        data = buf.getvalue()
        return {
            "size_bytes": len(data),
            "hash": hashlib.sha256(data).hexdigest() if data else "",
        }
    elif req.action == "set":
        # Momentum/optimizer state sent as raw torch-serialized bytes in body.
        try:
            body = request.body()
            if not body:
                raise HTTPException(status_code=400, detail="Empty optimizer state body")
            payload = io.BytesIO(body)
            state.trainer.optimizer.load_state_dict(torch.load(payload, weights_only=False))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to apply optimizer state")
            raise HTTPException(status_code=400, detail=f"Invalid optimizer state: {e}")
        return {"status": "applied"}
    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


@app.post("/v1/train/checkpoint")
def save_checkpoint(req: CheckpointRequest):
    if not state.model:
        raise HTTPException(status_code=400, detail="Not initialized")
    state.save_checkpoint(req.path, req.save_merged)
    h = hashlib.sha256()
    for f in sorted(Path(req.path).rglob("*")):
        if (
            f.is_file() and f.stat().st_size < 100_000_000
        ):  # skip huge files for hashing
            h.update(f.read_bytes())
    return {"status": "saved", "path": req.path, "hash": h.hexdigest()}


@app.post("/v1/train/load")
def load_checkpoint(req: CheckpointRequest):
    if not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    # Re-init from checkpoint path as if it were the base model
    state.config.base_model = req.path
    init_training(state.config)
    return {"status": "loaded", "path": req.path}


@app.post("/v1/train/save_state")
def save_state(request: Request):
    import torch
    from peft import PeftModel, get_peft_model_state_dict

    if not state.model or not state.trainer or not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        model_state_dict = (
            get_peft_model_state_dict(state.model)
            if isinstance(state.model, PeftModel)
            else state.model.state_dict()
        )
        buf = io.BytesIO()
        torch.save(
            {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": state.trainer.optimizer.state_dict(),
                "step": state.step,
                "config": state.config.model_dump(),
            },
            buf,
        )
        return Response(content=buf.getvalue(), media_type="application/octet-stream")
    except Exception as e:
        logger.exception("save_state failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/train/load_state")
def load_state(request: Request):
    import torch

    if not state.model or not state.trainer:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        body = request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty state body")
        payload = io.BytesIO(body)
        checkpoint = torch.load(payload, weights_only=False, map_location="cpu")
        if isinstance(state.model, PeftModel):
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(state.model, checkpoint["model_state_dict"])
        else:
            state.model.load_state_dict(checkpoint["model_state_dict"])
        state.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        state.step = checkpoint.get("step", state.step)
        return {"status": "loaded"}
    except Exception as e:
        logger.exception("load_state failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval_held_out")
def eval_held_out(req: EvalHeldOutRequest):
    """Return per-example held-out losses for the base model and the candidate.

    The operator's eval gate uses these paired losses to certify whether the
    candidate checkpoint actually improved on data it was not trained on.
    """
    if not state.model or not state.tokenizer or not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")

    base_model = None
    try:
        max_examples = req.max_examples
        if max_examples is None:
            env_max = os.environ.get("HELD_OUT_MAX_EXAMPLES")
            if env_max:
                try:
                    max_examples = int(env_max)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid HELD_OUT_MAX_EXAMPLES: {env_max!r}",
                    ) from exc
            else:
                max_examples = 200

        ds = _load_held_out_dataset(max_examples)
        tokenizer = state.tokenizer

        # Load the raw base model for comparison. Keep this inside the try block so
        # OOM or load failures always clean up.
        base_model = _load_base_model_for_eval(req.base_model, state.config)

        max_length = state.config.max_seq_length
        base_losses = _compute_per_example_losses(base_model, tokenizer, ds, max_length)
        candidate_losses = _compute_per_example_losses(
            state.model, tokenizer, ds, max_length
        )

        if not base_losses or not candidate_losses:
            raise HTTPException(status_code=500, detail="No valid losses computed")
        if len(base_losses) != len(candidate_losses):
            raise HTTPException(
                status_code=500,
                detail=f"Loss length mismatch: base={len(base_losses)} candidate={len(candidate_losses)}",
            )

        logger.info(
            f"Held-out eval: base_mean={sum(base_losses) / len(base_losses):.4f}, "
            f"candidate_mean={sum(candidate_losses) / len(candidate_losses):.4f}, "
            f"examples={len(base_losses)}"
        )
        return {"base": base_losses, "candidate": candidate_losses}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Held-out eval failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if base_model is not None:
            try:
                del base_model
                _clear_gpu_cache()
            except Exception:
                pass


@app.get("/v1/train/status")
def get_status():
    gpu_used, gpu_total = state.get_gpu_info()
    elapsed = time.time() - state.start_time if state.start_time else 0
    tps = state.tokens_processed / elapsed if elapsed > 0 else 0
    return {
        "backend": state.backend_name,
        "model": state.config.base_model if state.config else "",
        "method": state.config.method if state.config else "",
        "step": state.step,
        "loss": state.last_loss,
        "gpu_memory_used_mb": gpu_used,
        "gpu_memory_total_mb": gpu_total,
        "tokens_per_second": round(tps, 1),
        "tokens_processed": state.tokens_processed,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# DeMo (Decoupled Momentum) sync helpers
# ---------------------------------------------------------------------------


def _dct_1d(x: "torch.Tensor") -> "torch.Tensor":
    """Type-II DCT of a 1-D tensor (naive implementation; fine for tiny models)."""
    import torch
    N = x.numel()
    if N == 0:
        return x.clone()
    n = torch.arange(N, dtype=x.dtype, device=x.device)
    k = torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(1)
    scale = torch.ones(N, dtype=x.dtype, device=x.device)
    scale[0] = 1.0 / (2.0 ** 0.5)
    cos = torch.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * N))
    return (2.0 / N) ** 0.5 * scale * (x.unsqueeze(0) @ cos.T).squeeze(0)


def _idct_1d(x: "torch.Tensor") -> "torch.Tensor":
    """Type-III inverse DCT of a 1-D tensor."""
    import torch
    N = x.numel()
    if N == 0:
        return x.clone()
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    n = torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(1)
    scale = torch.ones(N, dtype=x.dtype, device=x.device)
    scale[0] = 1.0 / (2.0 ** 0.5)
    cos = torch.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * N))
    return (2.0 / N) ** 0.5 * (scale * x).unsqueeze(0) @ cos


def _topk_sparsify_flat(delta: "torch.Tensor", ratio: float) -> tuple[list[int], list[float]]:
    """Return the top-k largest-by-magnitude flattened indices and values."""
    import torch
    flat = delta.detach().cpu().flatten().to(torch.float32)
    k = max(1, int(round(flat.numel() * ratio)))
    k = min(k, flat.numel())
    if k == 0:
        return [], []
    abs_flat = flat.abs()
    _, indices = torch.topk(abs_flat, k)
    indices = indices.tolist()
    values = flat[indices].tolist()
    return indices, values


def _decompress_flat(indices: list[int], values: list[float], shape: list[int]) -> "torch.Tensor":
    """Reconstruct a dense tensor from a flattened sparse update."""
    import torch
    size = int(torch.prod(torch.tensor(shape)).item())
    dense = torch.zeros(size, dtype=torch.float32)
    if indices:
        dense[indices] = torch.tensor(values, dtype=torch.float32)
    return dense.reshape(shape)


def _get_exp_avg_state() -> dict:
    """Return a copy of the optimizer's live first-moment (exp_avg) state per param."""
    if not state.trainer or not state.trainer.optimizer:
        return {}
    opt_state = state.trainer.optimizer.state
    return {
        pid: opt_state[pid]["exp_avg"].detach().clone()
        for pid in opt_state
        if isinstance(opt_state[pid], dict) and "exp_avg" in opt_state[pid]
    }


def _compress_demo_updates(
    baseline: dict, current: dict, ratio: float, step: int, peer_id: str
) -> list[SparseUpdate]:
    """Compute compressed momentum deltas relative to a baseline."""
    updates = []
    for pid in list(current.keys()):
        if pid not in baseline:
            continue
        base = baseline[pid]
        cur = current[pid]
        if base.shape != cur.shape:
            if base.numel() != cur.numel():
                logger.warning(
                    f"Skipping DeMo update for param {pid}: shape mismatch "
                    f"{tuple(base.shape)} vs {tuple(cur.shape)}"
                )
                continue
            base = base.reshape(cur.shape)
        delta = cur - base
        indices, values = _topk_sparsify_flat(delta, ratio)
        if not indices:
            continue
        shape = list(cur.shape)
        updates.append(
            SparseUpdate(
                indices=indices,
                values=values,
                shape=shape,
                step=step,
                peer_id=peer_id,
            )
        )
    return updates


def _apply_demo_updates(peer_updates: list[list[SparseUpdate]], ratio: float) -> None:
    """Aggregate peer compressed momentum updates and apply them to the optimizer."""
    import torch
    if not state.trainer or not state.trainer.optimizer:
        raise RuntimeError("Trainer not initialized")

    opt = state.trainer.optimizer
    opt_state = opt.state
    param_ids = [
        pid for pid in opt_state
        if isinstance(opt_state[pid], dict) and "exp_avg" in opt_state[pid]
    ]

    # peer_updates is a list of update sets, one per peer. Each update set is a
    # list aligned with param_ids.
    for idx, pid in enumerate(param_ids):
        deltas = []
        for updates in peer_updates:
            if idx >= len(updates):
                continue
            u = updates[idx]
            dense = _decompress_flat(u.indices, u.values, u.shape).to(
                opt_state[pid]["exp_avg"].device
            )
            if dense.shape != opt_state[pid]["exp_avg"].shape:
                dense = dense.reshape(opt_state[pid]["exp_avg"].shape)
            deltas.append(dense)
        if not deltas:
            continue
        avg_delta = torch.stack(deltas).mean(dim=0)
        opt_state[pid]["exp_avg"] = opt_state[pid]["exp_avg"] + avg_delta.to(
            opt_state[pid]["exp_avg"].device
        )

    # Reset the baseline to the post-sync momentum so the next delta is computed
    # against the freshly synchronized state.
    state.demo_baseline = _get_exp_avg_state()


@app.post("/v1/train/demo_step")
def demo_step(req: DemoStepRequest):
    if not state.trainer or not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")

    # Snapshot the current momentum before the local training burst.
    if state.demo_baseline is None:
        state.demo_baseline = _get_exp_avg_state()

    try:
        result = state.train_steps(req.num_steps, return_norms=False)
        current = _get_exp_avg_state()
        updates = _compress_demo_updates(
            state.demo_baseline,
            current,
            state.config.demo_top_k_ratio,
            state.step,
            "",  # peer_id filled in by the broadcasting layer
        )
        return DemoStepResponse(
            updates=updates,
            loss=result["loss"],
            steps_completed=result["steps_completed"],
            total_steps=result["total_steps"],
        )
    except Exception as e:
        logger.exception("DeMo step failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/train/demo_apply_sync")
def demo_apply_sync(req: DemoApplySyncRequest):
    if not state.trainer or not state.config:
        raise HTTPException(status_code=400, detail="Call /v1/train/init first")
    try:
        _apply_demo_updates(req.peer_updates, state.config.demo_top_k_ratio)
        return {"status": "applied"}
    except Exception as e:
        logger.exception("DeMo apply sync failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("TRAINING_PORT", "8000"))
    logger.info(f"Starting training adapter on 0.0.0.0:{port}")
    logger.info(f"Available backends: {AVAILABLE_BACKENDS}")
    uvicorn.run(app, host="0.0.0.0", port=port)

"""Tests for the /eval_held_out held-out evaluation endpoint.

These tests exercise the real helper functions and endpoint shape using a tiny
random transformer model so they run without GPU and without downloading a full
pretrained checkpoint.
"""

import os
import sys

import pytest

# Import the module under test. The adapter lives next to tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import (
    EvalHeldOutRequest,
    InitRequest,
    _compute_per_example_losses,
    _extract_text,
    eval_held_out,
    state,
)


def test_extract_text_plain():
    assert _extract_text({"text": "hello world"}) == "hello world"
    assert _extract_text({"content": "foo bar"}) == "foo bar"


def test_extract_text_chat_messages():
    example = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    text = _extract_text(example)
    assert "What is 2+2?" in text
    assert "4" in text


def test_extract_text_conversations():
    example = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
    }
    text = _extract_text(example)
    assert "Hi" in text
    assert "Hello" in text


def test_extract_text_empty():
    assert _extract_text({"messages": []}) == ""
    assert _extract_text({"nested": {"text": "ignored"}}) == ""


# Guard the ML-dependent tests so CI without torch/transformers skips cleanly.
_torch_available = True
try:
    import torch  # noqa: F401
except Exception:
    _torch_available = False

_transformers_available = True
try:
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
except Exception:
    _transformers_available = False


def _make_tiny_model_tokenizer():
    """Return a tiny random GPT-2-style model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=128,
        n_positions=512,
        n_ctx=512,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tokenizer


def _make_tiny_dataset():
    """Return a tiny Dataset of plain-text examples."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "The capital of France is Paris.",
            ]
        }
    )


@pytest.mark.skipif(
    not (_torch_available and _transformers_available),
    reason="torch and transformers required",
)
def test_compute_per_example_losses_finite():
    model, tokenizer = _make_tiny_model_tokenizer()
    dataset = _make_tiny_dataset()

    losses = _compute_per_example_losses(model, tokenizer, dataset, max_length=64)

    assert len(losses) == len(dataset)
    assert all(isinstance(v, float) for v in losses)
    assert all(v > 0 and v < 100 for v in losses)
    assert all(__import__("math").isfinite(v) for v in losses)


@pytest.mark.skipif(
    not (_torch_available and _transformers_available),
    reason="torch and transformers required",
)
def test_eval_held_out_endpoint_shape():
    """End-to-end test of POST /eval_held_out with a tiny model."""
    base_model, tokenizer = _make_tiny_model_tokenizer()
    # Candidate: clone and perturb weights so losses differ from base.
    candidate_model = __import__("copy").deepcopy(base_model)
    with __import__("torch").no_grad():
        for p in candidate_model.parameters():
            p.add_(__import__("torch").randn_like(p) * 0.01)

    state.model = candidate_model
    state.tokenizer = tokenizer
    state.config = InitRequest(
        base_model="gpt2",
        method="lora",
        max_seq_length=64,
        load_in_4bit=False,
    )

    # Point to a local JSONL held-out file so no network is needed.
    test_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "test-data.jsonl")
    )
    os.environ["HELD_OUT_DATASET_URL"] = test_data
    os.environ["HELD_OUT_DATASET_SPLIT"] = "train"
    os.environ["HELD_OUT_MAX_EXAMPLES"] = "3"

    try:
        resp = eval_held_out(EvalHeldOutRequest(base_model="gpt2"))
        assert "base" in resp
        assert "candidate" in resp
        assert len(resp["base"]) == len(resp["candidate"]) == 3
        assert all(__import__("math").isfinite(v) for v in resp["base"])
        assert all(__import__("math").isfinite(v) for v in resp["candidate"])
    finally:
        state.model = None
        state.tokenizer = None
        state.config = None
        for key in (
            "HELD_OUT_DATASET_URL",
            "HELD_OUT_DATASET_SPLIT",
            "HELD_OUT_MAX_EXAMPLES",
        ):
            os.environ.pop(key, None)


@pytest.mark.skipif(
    not (_torch_available and _transformers_available),
    reason="torch and transformers required",
)
def test_eval_held_out_missing_config_errors():
    state.model = None
    state.tokenizer = None
    state.config = None
    os.environ.pop("HELD_OUT_DATASET_URL", None)

    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        eval_held_out(EvalHeldOutRequest(base_model="gpt2"))

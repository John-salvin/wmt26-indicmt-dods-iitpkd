"""Smoke tests for data loading. Run locally before pushing."""
import os

import pytest


def test_imports():
    """Verify all required libraries import."""
    import torch
    import transformers
    import peft
    import datasets
    import sacrebleu
    assert True


def test_sacrebleu_flores200_tokenizer():
    """The flores200 tokenizer must exist."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(
        ["this is a test"],
        [["this is a test"]],
        tokenize="flores200",
    )
    assert bleu.score == 100.0


@pytest.mark.skipif(
    not os.path.exists("/scratch"),
    reason="Cluster path /scratch not available in local dev",
)
def test_scratch_writable():
    user = os.environ.get("USER", "unknown")
    path = f"/scratch/{user}/wmt26/tests"
    os.makedirs(path, exist_ok=True)
    assert os.access(path, os.W_OK)

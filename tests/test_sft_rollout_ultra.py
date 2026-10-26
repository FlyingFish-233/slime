from types import SimpleNamespace

import pytest
import torch

from slime.observability.rollout_data_utils import save_debug_rollout_data
from slime.rollout import sft_rollout_ultra
from slime.utils.types import Sample


class FakeMaskGenerator:
    def __init__(self, token_ids, loss_mask):
        self.token_ids = token_ids
        self.loss_mask = loss_mask

    def get_loss_mask(self, messages, tools=None):
        return self.token_ids.copy(), self.loss_mask.copy()

    def get_response_lengths(self, masks):
        return [len(mask) - mask.index(1) if 1 in mask else 0 for mask in masks]


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "tools": None, "return_dict": False}
        return "<|im_start|>assistant\nfull rendered message<|im_end|>"


def _run_rollout(monkeypatch, token_ids, loss_mask, seq_length, *, debug=True):
    monkeypatch.setattr(sft_rollout_ultra, "TOKENIZER", FakeTokenizer())
    monkeypatch.setattr(sft_rollout_ultra, "PROCESSOR", object())
    monkeypatch.setattr(sft_rollout_ultra, "MASK_GENERATOR", FakeMaskGenerator(token_ids, loss_mask))
    sample = Sample(index=7, prompt=[{"role": "assistant", "content": "full rendered message"}])
    buffer = SimpleNamespace(get_samples=lambda batch_size: [[sample]])
    args = SimpleNamespace(
        rollout_batch_size=1,
        seq_length=seq_length,
        save_debug_rollout_data="rollout_{rollout_id}.pt" if debug else None,
    )
    return sft_rollout_ultra.generate_rollout(args, 12, buffer)[0][0]


def test_keeps_first_tokens_and_exports_full_rendered_text(monkeypatch, tmp_path):
    sample = _run_rollout(monkeypatch, [10, 11, 12, 13, 14], [0, 1, 0, 1, 1], 3)

    assert sample.tokens == [10, 11]
    assert sample.response_length == 1
    assert sample.loss_mask == [1]
    assert sample.metadata == {
        "sft_original_length": 5,
        "sft_truncated": True,
        "sft_supervised_tokens": 1,
        "sft_tail_masked_tokens_removed": 1,
    }

    save_debug_rollout_data(str(tmp_path / "rollout_{rollout_id}.pt"), [sample], rollout_id=12, evaluation=False)
    saved = torch.load(tmp_path / "rollout_12.pt", weights_only=False)["samples"][0]
    assert saved["rendered_text"] == "<|im_start|>assistant\nfull rendered message<|im_end|>"
    assert saved["tokens"] == [10, 11]


def test_rejects_prefix_without_assistant_targets(monkeypatch):
    with pytest.raises(ValueError, match="sample_index=7.*removed all assistant targets"):
        _run_rollout(monkeypatch, [10, 11, 12, 13], [0, 0, 0, 1], 3)


def test_keeps_short_sample_intact(monkeypatch):
    sample = _run_rollout(monkeypatch, [10, 11], [0, 1], 3, debug=False)
    assert sample.tokens == [10, 11]
    assert sample.loss_mask == [1]
    assert sample.metadata["sft_truncated"] is False
    assert sample.metadata["sft_tail_masked_tokens_removed"] == 0
    assert not hasattr(sample, "rendered_text")


def test_truncated_prefix_ending_with_target_keeps_all_tokens(monkeypatch):
    sample = _run_rollout(monkeypatch, [10, 11, 12, 13], [0, 1, 1, 0], 3)
    assert sample.tokens == [10, 11, 12]
    assert sample.loss_mask == [1, 1]
    assert sample.metadata["sft_tail_masked_tokens_removed"] == 0

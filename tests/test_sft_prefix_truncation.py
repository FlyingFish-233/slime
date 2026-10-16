from types import SimpleNamespace

import pytest

from slime.rollout import sft_rollout_ultra as rollout


@pytest.mark.parametrize("length", [199999, 200000, 200001])
def test_prefix_boundary(length):
    tokens = list(range(length))
    mask = [0, 0] + [1] * (length - 2)
    actual, actual_mask, truncated = rollout._truncate_rollout(tokens, mask, 200000)
    assert actual == tokens[:200000]
    assert actual_mask == mask[:200000]
    assert truncated == (length > 200000)


@pytest.mark.parametrize("tokens,mask,limit", [([1, 2], [1], 1), ([1, 2], [0, 1], 1), ([1], [0], 2), ([1], [1], 0)])
def test_invalid_supervision(tokens, mask, limit):
    with pytest.raises(ValueError):
        rollout._truncate_rollout(tokens, mask, limit)


def test_rollout_recomputes_supervision(monkeypatch):
    class Mask:
        def get_loss_mask(self, messages, tools=None):
            return list(range(8)), [0, 0, 1, 1, 0, 1, 1, 1]

        def get_response_lengths(self, masks):
            return [len(m) - m.index(1) for m in masks]

    monkeypatch.setattr(rollout, "TOKENIZER", object())
    monkeypatch.setattr(rollout, "PROCESSOR", object())
    monkeypatch.setattr(rollout, "MASK_GENERATOR", Mask())
    sample = SimpleNamespace(prompt=[], metadata={}, index=42)
    buffer = SimpleNamespace(get_samples=lambda n: [[sample]])
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1, seq_length=5)
    result = rollout.generate_rollout(args, 0, buffer)[0][0]
    assert result.tokens == [0, 1, 2, 3, 4]
    assert result.response_length == 3
    assert result.loss_mask == [1, 1, 0]
    assert result.metadata["sft_original_length"] == 8
    assert result.metadata["sft_supervised_tokens"] == 2

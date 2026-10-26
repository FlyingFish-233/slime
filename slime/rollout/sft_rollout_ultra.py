import logging

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _truncate_rollout(token_ids, loss_mask, max_length):
    if len(token_ids) != len(loss_mask):
        raise ValueError("SFT token_ids and loss_mask must have equal lengths")
    if max_length <= 0:
        raise ValueError(f"SFT seq_length must be positive, got {max_length}")
    if 1 not in loss_mask:
        raise ValueError("SFT sample has no assistant targets")

    original_length = len(token_ids)
    token_ids = token_ids[:max_length]
    loss_mask = loss_mask[:max_length]
    if 1 not in loss_mask:
        raise ValueError(
            f"Truncation to the first {max_length} tokens removed all assistant targets "
            f"from a {original_length}-token sample"
        )

    was_truncated = original_length > max_length
    tail_masked_tokens_removed = 0
    if was_truncated:
        # Causal attention means tokens after the final supervised target cannot
        # affect any loss-bearing position, so they only consume compute.
        last_target = len(loss_mask) - 1 - loss_mask[::-1].index(1)
        tail_masked_tokens_removed = len(loss_mask) - last_target - 1
        token_ids = token_ids[: last_target + 1]
        loss_mask = loss_mask[: last_target + 1]
    return token_ids, loss_mask, was_truncated, tail_masked_tokens_removed


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Build SFT samples, keeping the first ``args.seq_length`` tokens."""
    assert not evaluation

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    truncated_count = 0

    for i, sample in enumerate(samples):
        (sample,) = sample
        messages = sample.prompt
        tools = sample.metadata.get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
        if args.save_debug_rollout_data is not None:
            sample.rendered_text = TOKENIZER.apply_chat_template(
                messages, tokenize=False, tools=tools, return_dict=False
            )

        original_length = len(token_ids)
        try:
            token_ids, loss_mask, was_truncated, tail_masked_tokens_removed = _truncate_rollout(
                token_ids, loss_mask, args.seq_length
            )
        except ValueError as exc:
            raise ValueError(
                f"SFT rollout_id={rollout_id}, batch_index={i}, "
                f"sample_index={sample.index}: {exc}"
            ) from exc
        truncated_count += was_truncated

        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
        sample.metadata["sft_original_length"] = original_length
        sample.metadata["sft_truncated"] = was_truncated
        sample.metadata["sft_supervised_tokens"] = sum(loss_mask)
        sample.metadata["sft_tail_masked_tokens_removed"] = tail_masked_tokens_removed

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                "SFT example: original_length=%d, length=%d, response_length=%d, supervised_tokens=%d",
                original_length, len(token_ids), response_length, sum(loss_mask),
            )
            SAMPLE_PRINTED = True

    if truncated_count:
        logger.info(
            "SFT rollout kept the first %d tokens for %d/%d samples",
            args.seq_length, truncated_count, len(samples),
        )

    return samples

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.utils.ppo_utils import calculate_log_probs_and_entropy


@pytest.mark.parametrize('chunk_size', [1, 4, 32])
def test_loss_and_gradients_match(chunk_size):
    torch.manual_seed(3)
    source = torch.randn(13, 32)
    tokens = torch.randint(32, (13,))
    mask = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0]).float()
    results = []
    for recompute in [False, True]:
        logits = source.clone().requires_grad_()
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits, tokens, None, chunk_size=chunk_size,
            with_entropy=False, recompute_chunks=recompute,
        )
        loss = -(log_probs.flatten() * mask).sum() / mask.sum()
        loss.backward()
        results.append((loss.detach(), logits.grad))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def _distributed_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        full = torch.randn(11, 32)
        tokens = torch.randint(32, (11,))
        local = full.chunk(2, dim=-1)[rank].clone().requires_grad_()
        reference = full.clone().requires_grad_()
        weights = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]).float()
        expected = -(reference.log_softmax(-1)[torch.arange(11), tokens] * weights).sum()
        actual, _ = calculate_log_probs_and_entropy(local, tokens, dist.group.WORLD, chunk_size=3, recompute_chunks=True)
        actual_loss = -(actual.flatten() * weights).sum()
        actual_loss.backward()
        expected.backward()
        torch.testing.assert_close(actual_loss, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(local.grad, reference.grad.chunk(2, dim=-1)[rank], rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_recompute():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_worker, args=(os.path.join(directory, 'rendezvous'),), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_recompute_reduces_peak_memory():
    torch.manual_seed(11)
    source = torch.randn(2048, 8192, dtype=torch.bfloat16, device='cuda')
    tokens = torch.randint(8192, (2048,), device='cuda')
    peaks, values = [], []
    for recompute in [False, True]:
        torch.cuda.empty_cache()
        logits = source.clone().requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        log_probs, _ = calculate_log_probs_and_entropy(logits, tokens, None, chunk_size=128, recompute_chunks=recompute)
        loss = -log_probs.mean()
        loss.backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        values.append((loss.detach().cpu(), logits.grad.cpu()))
        del logits, log_probs, loss
    torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
    torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)
    print(f'CUDA peak allocated bytes: baseline={peaks[0]}, chunk_recompute={peaks[1]}')
    assert peaks[1] < peaks[0] * .85, peaks

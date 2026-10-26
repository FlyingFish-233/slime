import pytest
import torch

from slime.utils.ppo_utils import calculate_log_probs_and_entropy


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_sft_logit_storage_reuse_matches_standard_gradients(device):
    torch.manual_seed(23)
    inputs = torch.randn(7, 5, device=device)
    weights = torch.randn(5, 13, device=device)
    targets = torch.tensor([1, 7, 3, 12, 0, 5, 8], device=device)
    mask = torch.tensor([1, 0, 1, 1, 0, 1, 1], dtype=torch.float32, device=device)

    def run(reuse):
        x = inputs.detach().clone().requires_grad_()
        w = weights.detach().clone().requires_grad_()
        logits = x @ w
        log_probs, _ = calculate_log_probs_and_entropy(
            logits,
            targets,
            None,
            chunk_size=3,
            with_entropy=False,
            reuse_logits_for_grad=reuse,
        )
        loss = (log_probs.flatten() * mask).sum()
        loss.backward()
        return loss.detach(), x.grad.detach(), w.grad.detach()

    reused = run(True)
    normal = run(False)
    for actual, expected in zip(reused, normal, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    x = inputs.detach().clone().requires_grad_()
    w = weights.detach().clone().requires_grad_()
    logits = x @ w
    expected_loss = (logits.log_softmax(-1).gather(-1, targets[:, None]).flatten() * mask).sum()
    expected_loss.backward()
    for actual, expected in zip(reused, (expected_loss, x.grad, w.grad), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

import pytest
import torch

from slime_plugins.models.chunked_gdn import head_chunked_gdn


@pytest.mark.parametrize('chunk_size', [1, 3, 8])
def test_head_chunks_preserve_outputs_and_gradients(chunk_size):
    def kernel(q, k, v, *, g, beta, **kwargs):
        return (q * k + v * beta[..., None] + g[..., None]).cumsum(1), None

    torch.manual_seed(7)
    source = [torch.randn(1, 17, 10, 4) for _ in range(3)]
    source += [torch.randn(1, 17, 10) for _ in range(2)]
    weight = torch.randn_like(source[0])
    results = []
    for chunked in (False, True):
        inputs = [x.clone().requires_grad_() for x in source]
        if chunked:
            out = head_chunked_gdn(*inputs, None, kernel, chunk_size)
        else:
            out = kernel(*inputs[:3], g=inputs[3], beta=inputs[4])[0]
        grads = torch.autograd.grad((out * weight).sum(), inputs)
        results.append((out.detach(), grads))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA FLA')
def test_fla_varlen_outputs_gradients_and_peak_memory():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(17)
    source = [torch.randn(1, 16384, 32, 128, dtype=torch.bfloat16, device='cuda') for _ in range(3)]
    source += [-torch.rand(1, 16384, 32, device='cuda'), torch.rand(1, 16384, 32, device='cuda', dtype=torch.bfloat16)]
    cu = torch.tensor([0, 4001, 16384], dtype=torch.int32, device='cuda')
    weight = torch.randn_like(source[0])
    results, peaks = [], []
    for chunked in (False, True):
        torch.cuda.empty_cache()
        inputs = [x.clone().requires_grad_() for x in source]
        torch.cuda.reset_peak_memory_stats()
        if chunked:
            out = head_chunked_gdn(*inputs, cu, chunk_gated_delta_rule, 8)
        else:
            out = chunk_gated_delta_rule(
                *inputs[:3], g=inputs[3], beta=inputs[4], initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
            )[0]
        grads = torch.autograd.grad(out, inputs, weight)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
        results.append((out.detach().cpu(), tuple(x.cpu() for x in grads)))
        del inputs, out, grads
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    print(f'FLA peak bytes: baseline={peaks[0]}, head_chunked={peaks[1]}')
    assert peaks[1] < peaks[0] * .85, peaks

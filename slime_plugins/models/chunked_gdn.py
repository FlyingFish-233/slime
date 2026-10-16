"""Recompute independent GDN heads in bounded groups during backward."""

import torch


class _HeadChunkedGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, g, beta, cu_seqlens, kernel, chunk_size):
        ctx.save_for_backward(query, key, value, g, beta)
        ctx.cu_seqlens, ctx.kernel, ctx.chunk_size = cu_seqlens, kernel, chunk_size
        output = torch.empty_like(value)
        for start in range(0, value.size(2), chunk_size):
            end = start + chunk_size
            inputs = [x[:, :, start:end].contiguous() for x in (query, key, value, g, beta)]
            part = _run(kernel, inputs, cu_seqlens)
            output[:, :, start:end].copy_(part)
            del inputs, part
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        gradients = [torch.empty_like(x) for x in tensors]
        for start in range(0, tensors[2].size(2), ctx.chunk_size):
            end = start + ctx.chunk_size
            with torch.enable_grad():
                inputs = [x[:, :, start:end].detach().contiguous().requires_grad_(True) for x in tensors]
                part = _run(ctx.kernel, inputs, ctx.cu_seqlens)
                grads = torch.autograd.grad(part, inputs, grad_output[:, :, start:end].contiguous())
            for destination, grad in zip(gradients, grads, strict=True):
                destination[:, :, start:end].copy_(grad)
            del inputs, part, grads, grad
        return *gradients, None, None, None


def _run(kernel, inputs, cu_seqlens):
    query, key, value, g, beta = inputs
    output, _ = kernel(
        query, key, value, g=g, beta=beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    return output


def head_chunked_gdn(query, key, value, g, beta, cu_seqlens, kernel, chunk_size):
    """Preserve the full sequence; only independent attention heads are split."""
    if chunk_size <= 0:
        raise ValueError("GDN head chunk size must be positive")
    if not query.size(2) == key.size(2) == value.size(2) == g.size(2) == beta.size(2):
        raise ValueError("Expand grouped query/key heads before chunking GDN")
    return _HeadChunkedGDN.apply(query, key, value, g, beta, cu_seqlens, kernel, chunk_size)

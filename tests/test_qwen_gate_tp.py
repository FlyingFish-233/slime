from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.attention import SelfAttention

from slime_plugins.models.qwen3_5 import GatedSelfAttention


@pytest.mark.parametrize('tp', [4, 8])
def test_gate_matches_query_partition_and_gradient(monkeypatch, tp):
    full_gate = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 1, 16, 4).div(64).requires_grad_()
    full_output = torch.randn(1, 1, 16, 4)
    actual_outputs = []
    partitioned_gates = []
    for rank in range(tp):
        replicas = tp // 2
        kv_group = rank // replicas
        query_heads = 16 // tp
        query = full_output[:, :, rank * query_heads : (rank + 1) * query_heads]
        unsliced_gate = full_gate[:, :, kv_group * 8 : (kv_group + 1) * 8]
        monkeypatch.setattr(SelfAttention, 'get_query_key_value_tensors', lambda self, *a, q=query, g=unsliced_gate, **kw: (q, None, None, g))
        attention = GatedSelfAttention.__new__(GatedSelfAttention)
        torch.nn.Module.__init__(attention)
        attention.world_size = tp
        attention.config = SimpleNamespace(num_query_groups=2)
        attention.pg_collection = SimpleNamespace(tp=SimpleNamespace(rank=lambda rank=rank: rank))
        q, _, _, gate = attention.get_query_key_value_tensors(None, output_gate=True)
        partitioned_gates.append(gate)
        actual_outputs.append(q * gate.sigmoid())
    actual = torch.cat(actual_outputs, dim=2)
    expected = full_output * full_gate.sigmoid()
    torch.testing.assert_close(torch.cat(partitioned_gates, dim=2), full_gate, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual_grad, = torch.autograd.grad(actual.sum(), full_gate, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), full_gate)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_already_partitioned_gate_is_unchanged(monkeypatch):
    query = torch.randn(2, 1, 2, 4)
    gate = torch.randn_like(query)
    monkeypatch.setattr(SelfAttention, 'get_query_key_value_tensors', lambda *a, **kw: (query, None, None, gate))
    attention = GatedSelfAttention.__new__(GatedSelfAttention)
    torch.nn.Module.__init__(attention)
    assert attention.get_query_key_value_tensors(None, output_gate=True)[3] is gate

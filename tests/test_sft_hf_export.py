import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

_converter = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools' / 'convert_torch_dist_to_hf.py'))
get_named_params = _converter['get_named_params']
save_tensors = _converter['save_tensors']


def test_grouped_experts_export_trained_fused_weights(tmp_path):
    origin, output = tmp_path / 'origin', tmp_path / 'output'
    origin.mkdir()
    args = SimpleNamespace(num_experts=2, num_layers=1, hidden_size=4, num_attention_heads=2,
                           num_query_groups=1, kv_channels=2, vocab_size=8, q_lora_rank=None)
    gate_up = torch.arange(48).reshape(2, 6, 4).float()
    down = torch.arange(24).reshape(2, 4, 3).float()
    prefix = 'model.language_model.layers.0.mlp.experts.'
    expected = {prefix + 'gate_up_proj': gate_up, prefix + 'down_proj': down}
    save_file({k: torch.zeros_like(v) for k, v in expected.items()}, origin / 'base.safetensors')
    (origin / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'base.safetensors' for k in expected}}))
    state = {'decoder.layers.0.mlp.experts.experts.linear_fc1.weight': gate_up,
             'decoder.layers.0.mlp.experts.experts.linear_fc2.weight': down}
    # Existing callers still receive individual expert tensors by default.
    assert len(list(get_named_params(args, state))) == 4
    save_tensors(args, 'qwen3_5_moeconfig', state, str(output), 1024, origin_hf_dir=str(origin))
    metadata = json.loads((output / 'model.safetensors.index.json').read_text())
    index = metadata['weight_map']
    assert set(index) == set(expected)
    assert metadata['metadata']['total_size'] == sum(t.numel() * t.element_size() for t in expected.values())
    for key, tensor in expected.items():
        torch.testing.assert_close(load_file(output / index[key])[key], tensor, rtol=0, atol=0)


def test_missing_trained_language_tensor_cannot_fall_back_to_origin(tmp_path):
    origin = tmp_path / 'origin'
    origin.mkdir()
    key = 'model.language_model.norm.weight'
    save_file({key: torch.ones(4)}, origin / 'base.safetensors')
    (origin / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'base.safetensors'}}))
    with pytest.raises(ValueError, match='refusing origin fallback'):
        save_tensors(SimpleNamespace(), 'qwen3_5_moeconfig', {}, str(tmp_path / 'output'), 1024, origin_hf_dir=str(origin))

import torch

from eppgt_repro.config import ModelConfig
from eppgt_repro.model import build_model
from eppgt_repro.training import apply_finetune_mode, load_model_weights, remap_legacy_key


def test_model_forward_has_expected_shape():
    config = ModelConfig(
        protein_dim=100,
        atom_dim=46,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    model = build_model(config)
    protein = torch.randn(2, 6, 100)
    protein_lengths = torch.tensor([6, 5])
    compounds = torch.randn(2, 4, 46)
    adjs = torch.eye(4).repeat(2, 1, 1)
    compound_lengths = torch.tensor([4, 3])
    logits = model(protein, protein_lengths, compounds, adjs, compound_lengths)
    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_load_model_weights_handles_module_prefix():
    config = ModelConfig(
        protein_dim=100,
        atom_dim=46,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    model = build_model(config)
    state_dict = {f"module.{key}": value.clone() for key, value in model.state_dict().items()}
    load_model_weights(model, state_dict, strict=True)


def test_remap_legacy_key_matches_refactor_names():
    assert remap_legacy_key("module.encoder.blks.block0.attention.W_q.weight") == "encoder.blocks.0.attention.w_q.weight"
    assert remap_legacy_key("module.decoder.blks.block11.addnorm3.ln.bias") == "decoder.blocks.11.addnorm3.layer_norm.bias"


def test_apply_finetune_mode_head_only_freezes_backbone():
    config = ModelConfig(
        protein_dim=100,
        atom_dim=46,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    model = build_model(config)
    summary = apply_finetune_mode(model, "head-only")
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert summary["trainable_parameter_names"] == trainable_names
    assert all(name.startswith("decoder.dense3.") for name in trainable_names)
    assert any(not param.requires_grad for name, param in model.named_parameters() if not name.startswith("decoder.dense3."))

import pytest
import torch

from cistempoforge import CisTempoForgeModel


def batch(n_days, batch_size=2, peaks=2, length=40):
    return {
        "dna": torch.randn(batch_size, peaks, 4, length),
        "atac": torch.randn(batch_size, n_days, peaks, 1, length),
        "peak_mask": torch.ones(batch_size, peaks, dtype=torch.bool),
        "promoter_mask": torch.tensor([[True, False]] * batch_size),
        "mean_absolute_tss_distance": torch.rand(batch_size, peaks),
        "structure_features": torch.randn(batch_size, 8),
        "structure_mask": torch.ones(batch_size, 8),
    }


@pytest.mark.parametrize("n_days", [6, 7])
def test_dynamic_day_forward_contract(n_days):
    model = CisTempoForgeModel(
        n_days=n_days, hidden_dim=8, motif_filters=2, dropout=0,
        variant="independent_structure", fusion="concat",
    )
    output = model(batch(n_days))
    assert output["absolute"].shape == (2, n_days)
    assert torch.max(torch.abs(output["trend"].mean(1))) < 1e-6
    assert torch.allclose(output["absolute"], output["level"][:, None] + output["trend"])


def test_independent_level_gradient_isolation():
    model = CisTempoForgeModel(
        n_days=6, hidden_dim=8, motif_filters=2, dropout=0,
        variant="independent_structure", fusion="concat",
    )
    model(batch(6))["level"].square().mean().backward()
    assert all(parameter.grad is None for parameter in model.dynamic.parameters())
    assert any(parameter.grad is not None for parameter in model.level_dynamic.parameters())


def test_fusion_contract_and_day_mismatch():
    with pytest.raises(ValueError, match="require a fusion"):
        CisTempoForgeModel(variant="independent_structure", fusion="none")
    model = CisTempoForgeModel(
        n_days=6, hidden_dim=8, motif_filters=2,
        variant="canonical", fusion="none",
    )
    with pytest.raises(ValueError, match="expects 6 days"):
        model(batch(7))

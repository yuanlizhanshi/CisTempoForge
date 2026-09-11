from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "cistempoforge_dynamic_structure_v1"
VARIANTS = {
    "canonical", "relative_control", "relative_structure",
    "absolute_replacement", "dual", "dual_structure", "independent_structure",
}
FUSIONS = {"none", "concat", "residual", "film"}


class SequenceScanner(nn.Module):
    def __init__(self, hidden_dim: int, motif_filters: int, motif_width: int, dropout: float):
        super().__init__()
        padding = motif_width // 2
        self.dna_scanner = nn.Conv1d(4, motif_filters, motif_width, padding=padding, bias=False)
        self.dna_norm = nn.BatchNorm1d(motif_filters)
        self.atac_scanner = nn.Conv1d(1, motif_filters, motif_width, padding=padding, bias=False)
        self.atac_norm = nn.BatchNorm1d(motif_filters)
        self.bin_project = nn.Sequential(
            nn.Conv1d(motif_filters * 6, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.position_attention = nn.Conv1d(hidden_dim, 1, 1)
        self.peak_project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def _encode(self, motif: torch.Tensor, gate: torch.Tensor):
        joint = motif * gate
        pooled = torch.cat([
            F.max_pool1d(motif, 20, 20), F.avg_pool1d(motif, 20, 20),
            F.max_pool1d(joint, 20, 20), -F.max_pool1d(-joint, 20, 20),
            F.avg_pool1d(joint, 20, 20), F.avg_pool1d(gate, 20, 20),
        ], 1)
        bins = self.bin_project(pooled)
        weights = torch.softmax(self.position_attention(bins), -1)
        return self.peak_project((bins * weights).sum(-1)), weights.squeeze(1)

    def temporal(self, dna: torch.Tensor, atac: torch.Tensor, center: bool):
        n, days, _, length = atac.shape
        motif = torch.relu(self.dna_norm(self.dna_scanner(dna)))
        signal = atac - atac.mean(1, keepdim=True) if center else atac
        gate = torch.tanh(self.atac_norm(self.atac_scanner(signal.reshape(n * days, 1, length))))
        motif_day = motif[:, None].expand(-1, days, -1, -1).reshape(
            n * days, motif.shape[1], length
        )
        state, weights = self._encode(motif_day, gate)
        return state.reshape(n, days, -1), weights.reshape(n, days, -1)

    def static(self, dna: torch.Tensor, atac: torch.Tensor):
        motif = torch.relu(self.dna_norm(self.dna_scanner(dna)))
        gate = torch.tanh(self.atac_norm(self.atac_scanner(atac.mean(1))))
        return self._encode(motif, gate)


class StructureFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, method: str):
        super().__init__()
        self.method = method
        self.embedding = nn.Sequential(
            nn.Linear(16, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        if method == "concat":
            self.fused_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
        elif method == "residual":
            self.structure_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
            )
        elif method == "film":
            self.film = nn.Linear(hidden_dim // 2, hidden_dim * 2)

    def forward(self, state, features, mask):
        if self.method == "none":
            return self.base_head(state).squeeze(-1)
        structure = self.embedding(torch.cat([features, mask], -1))
        if self.method == "concat":
            return self.fused_head(torch.cat([state, structure], -1)).squeeze(-1)
        if self.method == "residual":
            return (self.base_head(state) + self.structure_head(structure)).squeeze(-1)
        gamma, beta = self.film(structure).chunk(2, -1)
        return self.base_head(state * (1 + 0.1 * torch.tanh(gamma)) + beta).squeeze(-1)


class CisTempoForgeModel(nn.Module):
    def __init__(self, n_days: int = 6, variant: str = "independent_structure",
                 fusion: str = "concat", hidden_dim: int = 128, motif_filters: int = 32,
                 motif_width: int = 19, dropout: float = 0.2,
                 detach_level_state: bool = False):
        super().__init__()
        if n_days < 2:
            raise ValueError("n_days must be at least 2")
        if variant not in VARIANTS or fusion not in FUSIONS:
            raise ValueError("unknown model variant/fusion")
        needs_structure = variant in {"relative_structure", "dual_structure", "independent_structure"}
        if needs_structure != (fusion != "none"):
            raise ValueError("structure variants require a fusion and other variants require none")
        self.n_days, self.variant, self.fusion = int(n_days), variant, fusion
        self.detach_level_state = bool(detach_level_state)
        self.dynamic = SequenceScanner(hidden_dim, motif_filters, motif_width, dropout)
        self.distance_project = nn.Linear(1, hidden_dim)
        self.dynamic_peak_attention = nn.Linear(hidden_dim, 1)
        self.local_peak_attention = nn.Linear(hidden_dim, 1)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout, "gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.day_embedding = nn.Parameter(torch.zeros(1, n_days, hidden_dim))
        nn.init.normal_(self.day_embedding, std=0.02)
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.local_trend_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        if variant in {"dual", "dual_structure"}:
            self.static = SequenceScanner(hidden_dim, motif_filters, motif_width, dropout)
            self.static_distance = nn.Linear(1, hidden_dim)
            self.peak_type_embedding = nn.Embedding(2, hidden_dim)
            self.static_peak_attention = nn.Linear(hidden_dim, 1)
        if variant == "independent_structure":
            self.level_dynamic = SequenceScanner(hidden_dim, motif_filters, motif_width, dropout)
            self.level_distance = nn.Linear(1, hidden_dim)
            self.level_peak_attention = nn.Linear(hidden_dim, 1)
            level_layer = nn.TransformerEncoderLayer(
                hidden_dim, 4, hidden_dim * 2, dropout, "gelu",
                batch_first=True, norm_first=True,
            )
            self.level_temporal = nn.TransformerEncoder(level_layer, 2)
            self.level_day_embedding = nn.Parameter(torch.zeros(1, n_days, hidden_dim))
            nn.init.normal_(self.level_day_embedding, std=0.02)
            self.level_encoder_trend_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        self.level_fusion = StructureFusion(hidden_dim, dropout, fusion)

    @staticmethod
    def _attention(state, mask, layer):
        logits = layer(state).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(logits, -1) * mask.float()
        return weight / weight.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(self, batch):
        dna, atac = batch["dna"].float(), batch["atac"].float()
        mask = batch["peak_mask"].bool()
        distance = batch["mean_absolute_tss_distance"].float()
        batch_size, days, peaks, _, length = atac.shape
        if days != self.n_days:
            raise ValueError(f"model expects {self.n_days} days, received {days}")
        uncentered = self.variant == "absolute_replacement"
        local, position = self.dynamic.temporal(
            dna.reshape(batch_size * peaks, 4, length),
            atac.transpose(1, 2).reshape(batch_size * peaks, days, 1, length),
            center=not uncentered,
        )
        local = local.reshape(batch_size, peaks, days, -1).transpose(1, 2)
        local_distance = local + self.distance_project(distance[..., None])[:, None]
        day_mask = mask[:, None].expand(-1, days, -1)
        weights = self._attention(local_distance, day_mask, self.dynamic_peak_attention)
        dynamic_gene = (local_distance * weights[..., None]).sum(2)
        temporal = self.temporal(dynamic_gene + self.day_embedding)
        raw_trend = self.trend_head(temporal).squeeze(-1)
        trend = raw_trend - raw_trend.mean(1, keepdim=True)
        local_weights = self._attention(local_distance, day_mask, self.local_peak_attention)
        local_gene = (local_distance * local_weights[..., None]).sum(2)
        local_temporal = self.temporal(local_gene + self.day_embedding)
        local_raw = self.local_trend_head(local_temporal).squeeze(-1)
        local_trend = local_raw - local_raw.mean(1, keepdim=True)
        if batch.get("_trend_only", False):
            return {"trend": trend, "local_trend": local_trend}

        level_encoder_trend = None
        if self.variant in {"dual", "dual_structure"}:
            static_peak, static_position = self.static.static(
                dna.reshape(batch_size * peaks, 4, length),
                atac.transpose(1, 2).reshape(batch_size * peaks, days, 1, length),
            )
            static_peak = static_peak.reshape(batch_size, peaks, -1)
            peak_type = batch["promoter_mask"].long()
            static_peak = (
                static_peak + self.static_distance(distance[..., None])
                + self.peak_type_embedding(peak_type)
            )
            static_weight = self._attention(static_peak, mask, self.static_peak_attention)
            level_state = (static_peak * static_weight[..., None]).sum(1)
        elif self.variant == "independent_structure":
            level_local, static_position = self.level_dynamic.temporal(
                dna.reshape(batch_size * peaks, 4, length),
                atac.transpose(1, 2).reshape(batch_size * peaks, days, 1, length),
                center=True,
            )
            level_local = level_local.reshape(batch_size, peaks, days, -1).transpose(1, 2)
            level_local = level_local + self.level_distance(distance[..., None])[:, None]
            static_weight = self._attention(level_local, day_mask, self.level_peak_attention)
            level_gene = (level_local * static_weight[..., None]).sum(2)
            level_temporal = self.level_temporal(level_gene + self.level_day_embedding)
            level_state = level_temporal.mean(1)
            level_raw = self.level_encoder_trend_head(level_temporal).squeeze(-1)
            level_encoder_trend = level_raw - level_raw.mean(1, keepdim=True)
        else:
            static_weight, static_position = None, None
            level_state = temporal.mean(1)
        if self.detach_level_state:
            level_state = level_state.detach()
        level = self.level_fusion(
            level_state, batch["structure_features"].float(), batch["structure_mask"].float()
        )
        return {
            "level": level, "trend": trend, "absolute": level[:, None] + trend,
            "local_trend": local_trend, "dynamic_peak_attention": weights,
            "local_peak_attention": local_weights,
            "level_encoder_trend": level_encoder_trend,
            "dynamic_position_attention": position,
            "static_peak_attention": static_weight,
            "static_position_attention": static_position,
        }

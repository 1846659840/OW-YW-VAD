from __future__ import annotations

import torch
from torch import nn

from owywvad.config import AppConfig


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dilations: tuple[int, int, int]):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d)
                for d in dilations
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dilations])
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            x = norm(torch.relu(y) + residual)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.out_norm(x + attn_out)


class OWYWVADModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        prompt_dim: int,
        num_known_classes: int,
        hidden_dim: int,
        dilations: tuple[int, int, int],
        init_fusion: tuple[float, float, float, float],
    ):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, dilations)
        self.classifier = nn.Linear(hidden_dim, num_known_classes + 1)
        self.localizer = nn.Linear(hidden_dim, 1)
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        logits = torch.log(torch.tensor(init_fusion, dtype=torch.float32).clamp_min(1e-6))
        self.fusion_logits = nn.Parameter(logits)

    def normalized_fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=0)

    def forward(self, frame_features: torch.Tensor, prompt_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(frame_features)
        prompt_context = torch.tanh(self.prompt_proj(prompt_scores))
        fused = encoded + 0.1 * prompt_context
        class_logits = self.classifier(fused)
        loc_logits = self.localizer(fused).squeeze(-1)
        return {
            "encoded": fused,
            "class_logits": class_logits,
            "loc_logits": loc_logits,
            "fusion_weights": self.normalized_fusion_weights(),
        }


def build_model(config: AppConfig, input_dim: int = 25, prompt_dim: int = 13) -> OWYWVADModel:
    weights = config.model.fusion_weights
    return OWYWVADModel(
        input_dim=input_dim,
        prompt_dim=prompt_dim,
        num_known_classes=config.model.num_known_classes,
        hidden_dim=config.model.hidden_dim,
        dilations=config.model.tcn_dilations,
        init_fusion=(weights.alpha, weights.beta, weights.gamma, weights.delta),
    )


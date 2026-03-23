from __future__ import annotations

import torch
import torch.nn.functional as F

from owywvad.config import AppConfig
from owywvad.memory.bank import MemoryBank
from owywvad.priors.dynamics import DynamicsPrior


def entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=dim)


def compute_scores(
    outputs: dict[str, torch.Tensor],
    prompt_scores: torch.Tensor,
    raw_features: torch.Tensor,
    region_ids: torch.Tensor,
    config: AppConfig,
    memory_bank: MemoryBank | None = None,
    dynamics_prior: DynamicsPrior | None = None,
) -> dict[str, torch.Tensor]:
    posterior = torch.softmax(outputs["class_logits"], dim=-1)
    s_cls = 1.0 - posterior[..., 0]
    s_mem = memory_bank.score(outputs["encoded"]) if memory_bank else torch.zeros_like(s_cls)
    if dynamics_prior:
        dyn_scores = [dynamics_prior.score(seq_features, seq_regions) for seq_features, seq_regions in zip(raw_features, region_ids)]
        s_dyn = torch.stack(dyn_scores, dim=0)
    else:
        s_dyn = torch.zeros_like(s_cls)
    uncertainty = entropy(posterior, dim=-1) + entropy(torch.softmax(prompt_scores / config.model.uncertainty_temperature, dim=-1), dim=-1)
    fusion = outputs["fusion_weights"]
    final_score = torch.sigmoid(fusion[0] * s_cls + fusion[1] * s_mem + fusion[2] * s_dyn + fusion[3] * uncertainty)
    return {
        "posterior": posterior,
        "s_cls": s_cls,
        "s_mem": s_mem,
        "s_dyn": s_dyn,
        "uncertainty": uncertainty,
        "final_score": final_score,
    }


def memory_compactness_loss(encoded: torch.Tensor, temperature: float) -> torch.Tensor:
    flattened = encoded.reshape(-1, encoded.shape[-1])
    normalized = F.normalize(flattened, dim=-1)
    similarity = normalized @ normalized.T
    logits = similarity / max(temperature, 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def smoothness_loss(scores: torch.Tensor) -> torch.Tensor:
    return torch.abs(scores[:, 1:] - scores[:, :-1]).mean()


def mil_ranking_loss(scores: torch.Tensor, video_labels: torch.Tensor) -> torch.Tensor:
    pos_mask = video_labels > 0
    neg_mask = video_labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.zeros((), device=scores.device)
    pos = scores[pos_mask].max(dim=1).values.mean()
    neg = scores[neg_mask].max(dim=1).values.mean()
    return torch.relu(1.0 - pos + neg)

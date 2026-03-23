from __future__ import annotations

import torch


class DynamicsPrior:
    def __init__(self) -> None:
        self.means: dict[int, torch.Tensor] = {}
        self.vars: dict[int, torch.Tensor] = {}

    def fit(self, raw_features: torch.Tensor, region_ids: torch.Tensor) -> None:
        for region in torch.unique(region_ids):
            mask = region_ids == region
            region_features = raw_features[mask][:, 6:12]
            if region_features.numel() == 0:
                continue
            self.means[int(region)] = region_features.mean(dim=0).cpu()
            self.vars[int(region)] = region_features.var(dim=0, unbiased=False).clamp_min(1e-4).cpu()

    def score(self, raw_features: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        scores = []
        for feat, region in zip(raw_features, region_ids):
            idx = int(region.item())
            mean = self.means.get(idx)
            var = self.vars.get(idx)
            if mean is None or var is None:
                scores.append(torch.tensor(0.0, device=raw_features.device))
                continue
            target = feat[6:12]
            scores.append((((target - mean.to(raw_features.device)) ** 2) / var.to(raw_features.device)).mean())
        return torch.stack(scores, dim=0)

    def state_dict(self) -> dict[str, dict[int, torch.Tensor]]:
        return {"means": self.means, "vars": self.vars}

    @classmethod
    def from_state_dict(cls, state: dict[str, dict[int, torch.Tensor]] | None) -> "DynamicsPrior":
        obj = cls()
        if not state:
            return obj
        obj.means = state.get("means", {})
        obj.vars = state.get("vars", {})
        return obj


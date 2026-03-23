from __future__ import annotations

import torch


class MemoryBank:
    def __init__(self, prototypes: torch.Tensor | None = None, neighbors: int = 5):
        self.prototypes = prototypes
        self.neighbors = neighbors

    def fit(self, features: torch.Tensor, num_prototypes: int) -> None:
        if features.numel() == 0:
            self.prototypes = torch.zeros((1, 1), dtype=torch.float32)
            return
        flattened = features.reshape(-1, features.shape[-1]).detach().cpu()
        step = max(1, flattened.shape[0] // max(1, num_prototypes))
        self.prototypes = flattened[::step][:num_prototypes].contiguous()

    def score(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.prototypes is None or self.prototypes.numel() == 0:
            return torch.zeros(encoded.shape[:-1], device=encoded.device)
        bank = self.prototypes.to(encoded.device)
        distances = torch.cdist(encoded.reshape(-1, encoded.shape[-1]), bank)
        k = min(self.neighbors, distances.shape[1])
        values, _ = torch.topk(distances, k=k, largest=False, dim=1)
        return values.mean(dim=1).reshape(encoded.shape[:-1])

    def state_dict(self) -> dict[str, torch.Tensor | int | None]:
        return {"prototypes": self.prototypes, "neighbors": self.neighbors}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor | int | None] | None) -> "MemoryBank":
        if not state:
            return cls()
        return cls(prototypes=state.get("prototypes"), neighbors=int(state.get("neighbors", 5)))


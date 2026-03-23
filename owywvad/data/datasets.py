from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from owywvad.data.manifests import CacheRecord
from owywvad.utils import read_jsonl


class CachedVideoDataset(Dataset):
    def __init__(self, index_path: Path, split: str | None = None):
        rows = [CacheRecord.model_validate(row) for row in read_jsonl(index_path)]
        if split is not None:
            rows = [row for row in rows if row.split == split]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.rows[index]
        payload = torch.load(row.feature_path, map_location="cpu")
        payload["dataset"] = row.dataset
        payload["split"] = row.split
        payload["video_id"] = row.video_id
        payload["video_label"] = row.video_label
        payload["known_class"] = row.known_class
        payload["open_set"] = row.open_set
        payload["scene_id"] = row.scene_id
        return payload


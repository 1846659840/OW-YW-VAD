from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetSpec(BaseModel):
    name: str
    description: str
    official_homepage: str
    manual_download: bool = True
    source_env: str | None = None
    expected_splits: list[str] = Field(default_factory=list)


class PreparedVideoRecord(BaseModel):
    dataset: str
    split: str
    video_id: str
    video_path: str
    frame_labels_path: str
    pixel_masks_path: str | None = None
    video_label: int = 0
    known_class: int = 0
    open_set: str = "normal"
    scene_id: str = "scene_0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheRecord(BaseModel):
    dataset: str
    split: str
    video_id: str
    feature_path: str
    frame_labels_path: str
    pixel_masks_path: str | None = None
    video_label: int = 0
    known_class: int = 0
    open_set: str = "normal"
    scene_id: str = "scene_0"
    metadata: dict[str, Any] = Field(default_factory=dict)


def as_relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from owywvad.config import AppConfig
from owywvad.data.manifests import CacheRecord, PreparedVideoRecord
from owywvad.data.registry import dataset_names
from owywvad.perception.yoloworld import YOLOWorldAdapter
from owywvad.prompts.loader import load_prompt_groups
from owywvad.tokens.builder import build_frame_tokens
from owywvad.tracking.bytetrack import ByteTrackAdapter
from owywvad.utils import ensure_dir, read_jsonl, save_yaml, write_jsonl


def _load_frames(video_path: Path) -> np.ndarray:
    if video_path.suffix == ".npz":
        return np.load(video_path)["frames"].astype(np.float32)
    if video_path.suffix == ".npy":
        return np.load(video_path).astype(np.float32)
    raise ValueError(f"Unsupported video container: {video_path}")


def build_cache(dataset: str, config: AppConfig) -> list[str]:
    if dataset == "all":
        messages: list[str] = []
        for name in dataset_names():
            messages.extend(build_cache(name, config))
        return messages
    resolved = config.resolve()
    processed_dir = resolved.processed_data / dataset
    cache_dir = ensure_dir(resolved.cache_data / dataset)
    detector = YOLOWorldAdapter(config, resolved.external)
    tracker = ByteTrackAdapter(config)
    prompt_groups = load_prompt_groups(resolved.prompt_file)
    rows = []
    for item in read_jsonl(processed_dir / "metadata.jsonl"):
        record = PreparedVideoRecord.model_validate(item)
        frames = _load_frames(Path(record.video_path))
        detections = detector.extract(frames, prompt_groups)
        tracks = tracker.link(detections)
        payload = build_frame_tokens(frames, detections, tracks, prompt_groups)
        payload["frame_labels"] = torch.tensor(np.load(record.frame_labels_path), dtype=torch.long)
        payload["pixel_masks"] = torch.tensor(np.load(record.pixel_masks_path), dtype=torch.float32) if record.pixel_masks_path else torch.zeros((frames.shape[0], frames.shape[1], frames.shape[2]))
        payload["video_label"] = int(record.video_label)
        payload["known_class"] = int(record.known_class)
        payload["open_set"] = record.open_set
        payload["scene_id"] = record.scene_id
        feature_path = cache_dir / f"{record.video_id}.pt"
        torch.save(payload, feature_path)
        rows.append(
            CacheRecord(
                dataset=record.dataset,
                split=record.split,
                video_id=record.video_id,
                feature_path=str(feature_path.resolve()),
                frame_labels_path=record.frame_labels_path,
                pixel_masks_path=record.pixel_masks_path,
                video_label=record.video_label,
                known_class=record.known_class,
                open_set=record.open_set,
                scene_id=record.scene_id,
                metadata=record.metadata,
            ).model_dump()
        )
    write_jsonl(cache_dir / "index.jsonl", rows)
    save_yaml(cache_dir / "cache_manifest.yaml", {"dataset": dataset, "num_records": len(rows), "prompt_file": str(resolved.prompt_file)})
    return [f"{dataset}: cached {len(rows)} videos"]


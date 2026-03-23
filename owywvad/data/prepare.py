from __future__ import annotations

from pathlib import Path

from owywvad.config import AppConfig
from owywvad.data.manifests import PreparedVideoRecord
from owywvad.data.registry import dataset_names
from owywvad.utils import ensure_dir, read_jsonl, save_yaml, write_jsonl


def _prepare_from_metadata(raw_dir: Path, processed_dir: Path, dataset: str) -> list[str]:
    metadata_path = raw_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in {raw_dir}")
    rows = []
    for row in read_jsonl(metadata_path):
        record = PreparedVideoRecord(
            dataset=dataset,
            split=row["split"],
            video_id=row["video_id"],
            video_path=str((raw_dir / row["video_path"]).resolve()),
            frame_labels_path=str((raw_dir / row["frame_labels_path"]).resolve()),
            pixel_masks_path=str((raw_dir / row["pixel_masks_path"]).resolve()) if row.get("pixel_masks_path") else None,
            video_label=int(row.get("video_label", 0)),
            known_class=int(row.get("known_class", 0)),
            open_set=row.get("open_set", "normal"),
            scene_id=row.get("scene_id", "scene_0"),
            metadata=row.get("metadata", {}),
        )
        rows.append(record.model_dump())
    write_jsonl(processed_dir / "metadata.jsonl", rows)
    save_yaml(processed_dir / "manifest.yaml", {"dataset": dataset, "num_videos": len(rows), "splits": sorted({row["split"] for row in rows})})
    return [f"{dataset}: prepared {len(rows)} videos"]


def prepare_dataset(dataset: str, config: AppConfig) -> list[str]:
    resolved = config.resolve()
    ensure_dir(resolved.processed_data)
    if dataset == "all":
        messages: list[str] = []
        for name in dataset_names():
            messages.extend(prepare_dataset(name, config))
        return messages
    raw_dir = resolved.raw_data / dataset
    processed_dir = ensure_dir(resolved.processed_data / dataset)
    return _prepare_from_metadata(raw_dir, processed_dir, dataset)


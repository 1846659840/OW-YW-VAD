from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from owywvad.config import AppConfig
from owywvad.data.manifests import DatasetSpec
from owywvad.data.registry import DATASET_REGISTRY, dataset_names
from owywvad.utils import ensure_dir, save_yaml, write_jsonl, write_text


def _copy_or_download(source: str, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    if source.startswith("http://") or source.startswith("https://"):
        destination = target_dir / Path(source).name
        urllib.request.urlretrieve(source, destination)
        return destination
    source_path = Path(source)
    if source_path.is_dir():
        destination = target_dir / source_path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        return destination
    destination = target_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination


def _manual_instructions(spec: DatasetSpec, dataset_dir: Path) -> None:
    write_text(
        dataset_dir / "INSTRUCTIONS.txt",
        (
            f"Dataset: {spec.name}\n"
            f"Official page: {spec.official_homepage}\n"
            "This benchmark is intentionally handled in a semi-automatic workflow.\n"
            "Place the official extracted files in this directory or rerun:\n"
            f"  python -m owywvad data fetch {spec.name} --source /path/to/archive_or_folder\n"
        ),
    )
    save_yaml(
        dataset_dir / "dataset_status.yaml",
        {
            "dataset": spec.name,
            "status": "awaiting_manual_source",
            "official_homepage": spec.official_homepage,
            "expected_splits": spec.expected_splits,
        },
    )


def _make_blob_video(length: int, anomaly_frames: tuple[int, int] | None, region: str, intensity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.zeros((length, 64, 64), dtype=np.float32)
    labels = np.zeros((length,), dtype=np.int64)
    masks = np.zeros((length, 64, 64), dtype=np.float32)
    for t in range(length):
        x = 6 + t
        y = 14 + (t // 3)
        if region == "restricted":
            x = 34 + (t // 2)
            y = 28
        if region == "reverse":
            x = 50 - t
            y = 24
        x0 = max(0, min(56, x))
        y0 = max(0, min(56, y))
        frames[t, y0 : y0 + 8, x0 : x0 + 8] = intensity
        if anomaly_frames and anomaly_frames[0] <= t < anomaly_frames[1]:
            labels[t] = 1
            masks[t, y0 : y0 + 8, x0 : x0 + 8] = 1.0
            frames[t, y0 : y0 + 8, x0 : x0 + 8] = min(1.0, intensity + 0.35)
    return frames, labels, masks


def _build_toy_pack(raw_root: Path) -> list[str]:
    messages: list[str] = []
    specs = {
        "shanghaitech": [
            ("train", "sh_train_normal_001", None, 0, 0, "normal", "scene_a"),
            ("train", "sh_train_normal_002", None, 0, 0, "normal", "scene_b"),
            ("test", "sh_test_abnormal_001", (10, 18), 1, 1, "known", "scene_c"),
            ("test", "sh_test_normal_001", None, 0, 0, "normal", "scene_d"),
        ],
        "ubnormal": [
            ("train", "ub_train_normal_001", None, 0, 0, "normal", "scene_a"),
            ("train", "ub_train_known_001", (8, 17), 1, 1, "known", "scene_b"),
            ("val", "ub_val_known_001", (12, 20), 2, 1, "known", "scene_c"),
            ("test", "ub_test_unknown_001", (11, 22), 0, 2, "unknown", "scene_d"),
            ("test", "ub_test_known_001", (9, 18), 1, 1, "known", "scene_e"),
            ("test", "ub_test_normal_001", None, 0, 0, "normal", "scene_f"),
        ],
        "ucf_crime": [
            ("train", "ucf_train_normal_001", None, 0, 0, "normal", "scene_a"),
            ("train", "ucf_train_anomaly_001", (12, 25), 1, 1, "known", "scene_b"),
            ("test", "ucf_test_anomaly_001", (9, 27), 1, 1, "known", "scene_c"),
            ("test", "ucf_test_normal_001", None, 0, 0, "normal", "scene_d"),
        ],
    }
    for dataset, rows in specs.items():
        dataset_dir = ensure_dir(raw_root / dataset)
        records = []
        for split, video_id, anomaly_range, known_class, video_label, open_set, scene_id in rows:
            region = "normal"
            if open_set == "unknown":
                region = "reverse"
            elif dataset == "shanghaitech":
                region = "restricted"
            frames, labels, masks = _make_blob_video(32, anomaly_range, region, intensity=0.35)
            video_path = dataset_dir / f"{video_id}.npz"
            labels_path = dataset_dir / f"{video_id}_labels.npy"
            masks_path = dataset_dir / f"{video_id}_masks.npy"
            np.savez_compressed(video_path, frames=frames)
            np.save(labels_path, labels)
            np.save(masks_path, masks)
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "video_id": video_id,
                    "video_path": video_path.name,
                    "frame_labels_path": labels_path.name,
                    "pixel_masks_path": masks_path.name,
                    "video_label": video_label,
                    "known_class": known_class,
                    "open_set": open_set,
                    "scene_id": scene_id,
                    "metadata": {"toy_pack": True},
                }
            )
        write_jsonl(dataset_dir / "metadata.jsonl", records)
        save_yaml(dataset_dir / "dataset_status.yaml", {"dataset": dataset, "status": "toy_ready"})
        messages.append(f"toy dataset prepared in raw/{dataset}")
    return messages


def fetch_dataset(dataset: str, config: AppConfig, source: str | None = None) -> list[str]:
    raw_root = config.resolve().raw_data
    ensure_dir(raw_root)
    if dataset == "toy":
        return _build_toy_pack(raw_root)
    if dataset == "all":
        messages: list[str] = []
        for name in dataset_names():
            messages.extend(fetch_dataset(name, config, None))
        return messages
    spec = DATASET_REGISTRY[dataset]
    dataset_dir = ensure_dir(raw_root / spec.name)
    selected_source = source or (os.getenv(spec.source_env) if spec.source_env else None)
    if selected_source:
        saved = _copy_or_download(selected_source, dataset_dir)
        save_yaml(dataset_dir / "dataset_status.yaml", {"dataset": spec.name, "status": "source_received", "source": str(saved)})
        return [f"{spec.name}: source copied to {saved}"]
    _manual_instructions(spec, dataset_dir)
    return [f"{spec.name}: awaiting manual source in {dataset_dir}"]


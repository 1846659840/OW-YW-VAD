from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from owywvad.config import AppConfig


@dataclass
class Detection:
    frame_index: int
    bbox: tuple[int, int, int, int]
    confidence: float
    prompt_scores: np.ndarray
    roi_feature: np.ndarray


class YOLOWorldAdapter:
    def __init__(self, config: AppConfig, external_root: Path):
        self.config = config
        self.external_root = external_root
        self.repo_dir = external_root / "YOLO-World"

    def backend_name(self) -> str:
        return "yoloworld" if self.repo_dir.exists() else "reference"

    def extract(self, frames: np.ndarray, prompt_groups: dict[str, list[str]]) -> list[list[Detection]]:
        detections: list[list[Detection]] = []
        prompt_count = sum(len(v) for v in prompt_groups.values())
        for frame_index, frame in enumerate(frames):
            mask = frame > max(0.2, float(frame.mean()) + 0.05)
            if mask.any():
                ys, xs = np.where(mask)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
            else:
                x0 = y0 = 0
                x1 = y1 = 0
            roi = frame[y0 : y1 + 1, x0 : x1 + 1] if x1 > x0 and y1 > y0 else frame[:8, :8]
            confidence = float(np.clip(frame.max(), 0.1, 0.99))
            prompt_scores = np.linspace(0.1, 0.9, prompt_count, dtype=np.float32)
            prompt_scores = prompt_scores * confidence
            roi_feature = np.array(
                [
                    float(roi.mean()),
                    float(roi.std()),
                    float(frame.mean()),
                    float(frame.std()),
                    float((x1 - x0 + 1) / frame.shape[1]),
                    float((y1 - y0 + 1) / frame.shape[0]),
                ],
                dtype=np.float32,
            )
            detections.append(
                [
                    Detection(
                        frame_index=frame_index,
                        bbox=(x0, y0, x1, y1),
                        confidence=confidence,
                        prompt_scores=prompt_scores,
                        roi_feature=roi_feature,
                    )
                ]
            )
        return detections


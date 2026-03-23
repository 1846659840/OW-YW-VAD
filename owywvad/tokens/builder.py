from __future__ import annotations

import numpy as np
import torch

from owywvad.perception.yoloworld import Detection
from owywvad.tracking.bytetrack import TrackDetection


def _region_embedding(center_x: float, center_y: float) -> np.ndarray:
    return np.array(
        [
            1.0 if center_x < 0.5 and center_y < 0.5 else 0.0,
            1.0 if center_x >= 0.5 and center_y < 0.5 else 0.0,
            1.0 if center_x < 0.5 and center_y >= 0.5 else 0.0,
            1.0 if center_x >= 0.5 and center_y >= 0.5 else 0.0,
        ],
        dtype=np.float32,
    )


def build_frame_tokens(
    frames: np.ndarray,
    detections: list[list[Detection]],
    tracks: list[list[TrackDetection]],
    prompt_groups: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    feature_rows: list[np.ndarray] = []
    prompt_rows: list[np.ndarray] = []
    bbox_rows: list[np.ndarray] = []
    region_ids: list[int] = []
    prev_center = np.array([0.0, 0.0], dtype=np.float32)
    for frame, frame_dets, frame_tracks in zip(frames, detections, tracks):
        det = frame_dets[0]
        x0, y0, x1, y1 = det.bbox
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        cx = (x0 + x1) / 2.0 / frame.shape[1]
        cy = (y0 + y1) / 2.0 / frame.shape[0]
        center = np.array([cx, cy], dtype=np.float32)
        delta = center - prev_center
        prev_center = center
        region_embed = _region_embedding(cx, cy)
        region_id = int(region_embed.argmax())
        region_ids.append(region_id)
        prompt_scores = det.prompt_scores.astype(np.float32)
        token = np.concatenate(
            [
                det.roi_feature,
                np.array([cx, cy, width / frame.shape[1], height / frame.shape[0]], dtype=np.float32),
                delta.astype(np.float32),
                prompt_scores[: min(8, len(prompt_scores))],
                region_embed,
                np.array([float(frame_tracks[0].track_id)], dtype=np.float32),
            ]
        )
        feature_rows.append(token.astype(np.float32))
        prompt_rows.append(prompt_scores.astype(np.float32))
        bbox_rows.append(np.array([x0, y0, x1, y1], dtype=np.float32))
    return {
        "frame_features": torch.tensor(np.stack(feature_rows), dtype=torch.float32),
        "prompt_scores": torch.tensor(np.stack(prompt_rows), dtype=torch.float32),
        "boxes": torch.tensor(np.stack(bbox_rows), dtype=torch.float32),
        "region_ids": torch.tensor(np.array(region_ids), dtype=torch.long),
    }

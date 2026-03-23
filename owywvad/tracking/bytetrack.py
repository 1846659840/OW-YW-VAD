from __future__ import annotations

from dataclasses import dataclass

from owywvad.config import AppConfig
from owywvad.perception.yoloworld import Detection


@dataclass
class TrackDetection:
    track_id: int
    detection: Detection


class ByteTrackAdapter:
    def __init__(self, config: AppConfig):
        self.config = config

    def link(self, detections: list[list[Detection]]) -> list[list[TrackDetection]]:
        linked: list[list[TrackDetection]] = []
        for frame_detections in detections:
            linked.append([TrackDetection(track_id=index, detection=det) for index, det in enumerate(frame_detections)])
        return linked


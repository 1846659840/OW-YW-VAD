from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from owywvad.config import AppConfig
from owywvad.losses.objectives import compute_scores
from owywvad.memory.bank import MemoryBank
from owywvad.models.model import build_model
from owywvad.perception.yoloworld import YOLOWorldAdapter
from owywvad.priors.dynamics import DynamicsPrior
from owywvad.prompts.loader import load_prompt_groups
from owywvad.tokens.builder import build_frame_tokens
from owywvad.tracking.bytetrack import ByteTrackAdapter
from owywvad.utils import ensure_dir, write_json
from owywvad.viz.plots import plot_sequence_summary


def _load_video(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        return np.load(path)["frames"].astype(np.float32)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    raise ValueError("Reference inference currently supports .npz and .npy inputs")


def infer_video(input_path: str, checkpoint_path: str, config: AppConfig) -> dict[str, str | int]:
    resolved = config.resolve()
    frames = _load_video(Path(input_path))
    prompts = load_prompt_groups(resolved.prompt_file)
    detections = YOLOWorldAdapter(config, resolved.external).extract(frames, prompts)
    tracks = ByteTrackAdapter(config).link(detections)
    tokens = build_frame_tokens(frames, detections, tracks, prompts)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(config, input_dim=tokens["frame_features"].shape[-1], prompt_dim=tokens["prompt_scores"].shape[-1])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bank = MemoryBank.from_state_dict(checkpoint.get("memory_bank"))
    prior = DynamicsPrior.from_state_dict(checkpoint.get("dynamics_prior"))
    with torch.no_grad():
        outputs = model(tokens["frame_features"].unsqueeze(0), tokens["prompt_scores"].unsqueeze(0))
        score_dict = compute_scores(
            outputs,
            tokens["prompt_scores"].unsqueeze(0),
            tokens["frame_features"].unsqueeze(0),
            tokens["region_ids"].unsqueeze(0),
            config,
            bank,
            prior,
        )
    scores = score_dict["final_score"][0].cpu().tolist()
    labels = ["normal" if score < config.model.decision_thresholds.tau_a else "anomaly" for score in scores]
    pred_dir = ensure_dir(resolved.outputs / config.run.name / "predictions")
    fig_dir = ensure_dir(resolved.outputs / config.run.name / "figures")
    stem = Path(input_path).stem
    json_path = pred_dir / f"{stem}.json"
    fig_path = fig_dir / f"{stem}.png"
    write_json(json_path, {"scores": scores, "labels": labels})
    plot_sequence_summary(frames, scores, [0] * len(scores), fig_path, title=stem)
    return {"prediction_json": str(json_path), "figure": str(fig_path), "num_frames": len(scores)}

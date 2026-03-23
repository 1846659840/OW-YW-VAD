from __future__ import annotations

import time

import numpy as np
import torch

from owywvad.config import AppConfig
from owywvad.data.datasets import CachedVideoDataset
from owywvad.eval.metrics import average_precision, binary_auc, macro_auc, rbdr, tbdr
from owywvad.losses.objectives import compute_scores
from owywvad.memory.bank import MemoryBank
from owywvad.models.model import build_model
from owywvad.priors.dynamics import DynamicsPrior
from owywvad.utils import ensure_dir, write_json, write_jsonl
from owywvad.viz.plots import plot_sequence_summary


def _mask_auc(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> float:
    predicted = np.zeros_like(masks, dtype=np.float32)
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = [int(v) for v in box]
        predicted[idx, y0 : y1 + 1, x0 : x1 + 1] = float(score)
    return binary_auc((masks > 0).reshape(-1).astype(np.int64), predicted.reshape(-1))


def evaluate_dataset(dataset: str, checkpoint_path: str, config: AppConfig) -> dict[str, float | str]:
    resolved = config.resolve()
    records = CachedVideoDataset(resolved.cache_data / dataset / "index.jsonl", split="test")
    sample = records[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(config, input_dim=sample["frame_features"].shape[-1], prompt_dim=sample["prompt_scores"].shape[-1])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bank = MemoryBank.from_state_dict(checkpoint.get("memory_bank"))
    prior = DynamicsPrior.from_state_dict(checkpoint.get("dynamics_prior"))
    all_scores = []
    all_labels = []
    all_classes = []
    pixel_scores = []
    prediction_rows = []
    figure_dir = ensure_dir(resolved.outputs / config.run.name / "figures" / dataset)
    start = time.time()
    for item in records:
        with torch.no_grad():
            outputs = model(item["frame_features"].unsqueeze(0), item["prompt_scores"].unsqueeze(0))
            score_dict = compute_scores(
                outputs,
                item["prompt_scores"].unsqueeze(0),
                item["frame_features"].unsqueeze(0),
                item["region_ids"].unsqueeze(0),
                config,
                bank,
                prior,
            )
        frame_scores = score_dict["final_score"][0].cpu().numpy()
        frame_labels = item["frame_labels"].cpu().numpy()
        all_scores.append(frame_scores)
        all_labels.append(frame_labels)
        all_classes.append(np.where(frame_labels > 0, int(item["known_class"]), 0))
        if dataset == "shanghaitech":
            pixel_scores.append(_mask_auc(item["pixel_masks"].cpu().numpy(), item["boxes"].cpu().numpy(), frame_scores))
        prediction_rows.append({"dataset": dataset, "video_id": item["video_id"], "scores": frame_scores.tolist(), "labels": frame_labels.tolist()})
        plot_sequence_summary(np.zeros((len(frame_scores), 64, 64), dtype=np.float32), frame_scores.tolist(), frame_labels.tolist(), figure_dir / f"{item['video_id']}.png", title=f"{dataset}:{item['video_id']}")
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    classes = np.concatenate(all_classes)
    elapsed = max(time.time() - start, 1e-6)
    metrics = {"dataset": dataset, "frame_auc": round(binary_auc(labels, scores) * 100, 4)}
    if dataset == "ubnormal":
        metrics["micro_auc"] = metrics["frame_auc"]
        metrics["macro_auc"] = round(macro_auc(classes, scores) * 100, 4)
        metrics["RBDC"] = round(rbdr(labels, scores) * 100, 4)
        metrics["TBDC"] = round(tbdr(labels, scores) * 100, 4)
    elif dataset == "shanghaitech":
        metrics["pixel_auc"] = round(float(np.mean(pixel_scores)) * 100, 4) if pixel_scores else 0.0
    elif dataset == "ucf_crime":
        metrics["AP"] = round(average_precision(labels, scores) * 100, 4)
        metrics["FPS"] = round(float(len(scores) / elapsed), 4)
    metrics_dir = ensure_dir(resolved.outputs / config.run.name / "metrics")
    pred_dir = ensure_dir(resolved.outputs / config.run.name / "predictions")
    write_json(metrics_dir / f"{dataset}.json", metrics)
    write_jsonl(pred_dir / f"{dataset}.jsonl", prediction_rows)
    return metrics


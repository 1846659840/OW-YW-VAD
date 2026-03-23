from __future__ import annotations

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1
    pos_ranks = ranks[labels == 1].sum()
    return float((pos_ranks - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    ap = 0.0
    previous = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - previous)
        previous = r
    return float(ap)


def macro_auc(class_labels: np.ndarray, anomaly_scores: np.ndarray) -> float:
    values = []
    for cls in sorted(np.unique(class_labels)):
        if cls <= 0:
            continue
        binary = (class_labels == cls).astype(np.int64)
        if binary.sum() == 0:
            continue
        values.append(binary_auc(binary, anomaly_scores))
    return float(np.mean(values)) if values else 0.0


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for idx, value in enumerate(labels.tolist()):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def rbdr(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    positives = labels == 1
    if positives.sum() == 0:
        return 0.0
    return float(((scores >= threshold) & positives).sum() / positives.sum())


def tbdr(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    gt_segments = _segments(labels)
    pred_segments = _segments((scores >= threshold).astype(np.int64))
    if not gt_segments:
        return 0.0
    matched = 0
    for gt_start, gt_end in gt_segments:
        for pred_start, pred_end in pred_segments:
            if max(gt_start, pred_start) <= min(gt_end, pred_end):
                matched += 1
                break
    return float(matched / len(gt_segments))


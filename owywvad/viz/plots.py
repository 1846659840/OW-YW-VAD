from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from owywvad.utils import ensure_dir


def plot_sequence_summary(frames: np.ndarray, scores: list[float], labels: list[int], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=220)
    ax.plot(x, scores, color="#0b5fa5", linewidth=2.15, label="Score")
    ax.fill_between(x, 0, 1, where=np.array(labels) > 0, color="#f28e2b", alpha=0.16, label="GT anomaly")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_comparison_bars(summary: dict[str, float], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    labels = list(summary.keys())
    values = [summary[key] for key in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=220)
    bars = ax.bar(labels, values, color=["#0b5fa5", "#3d7ea6", "#7ebdc2"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("OW-YW-VAD Main Results")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.set_ylim(0, max(values) + 8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

from __future__ import annotations

import csv

from owywvad.config import AppConfig
from owywvad.eval.evaluate import evaluate_dataset
from owywvad.train.runner import train_stage
from owywvad.utils import ensure_dir, read_csv, write_json
from owywvad.viz.plots import plot_comparison_bars


def reproduce_paper_main(config: AppConfig) -> dict[str, str]:
    resolved = config.resolve()
    output_root = ensure_dir(resolved.outputs / config.run.name)
    ckpt_dir = ensure_dir(output_root / "checkpoints")
    stage_paths = {
        "stage2": ckpt_dir / "stage2.pt",
        "stage3": ckpt_dir / "stage3.pt",
        "stage4": ckpt_dir / "stage4.pt",
    }
    for stage, path in stage_paths.items():
        if not path.exists():
            train_stage(stage, config)
    results = {
        "ubnormal": evaluate_dataset("ubnormal", str(stage_paths["stage4"]), config),
        "shanghaitech": evaluate_dataset("shanghaitech", str(stage_paths["stage4"]), config),
        "ucf_crime": evaluate_dataset("ucf_crime", str(stage_paths["stage4"]), config),
    }
    proposed = {
        "UBnormal": float(results["ubnormal"].get("micro_auc", results["ubnormal"]["frame_auc"])),
        "ShanghaiTech": float(results["shanghaitech"]["frame_auc"]),
        "UCF-Crime": float(results["ucf_crime"]["frame_auc"]),
    }
    plot_comparison_bars(proposed, output_root / "figures" / "paper_main_results.png")
    baselines = read_csv(resolved.baselines_csv)
    table_path = ensure_dir(output_root / "tables") / "comparison_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "year", "ubnormal_micro_auc", "shanghaitech_frame_auc", "ucf_crime_frame_auc", "source"])
        for row in baselines:
            writer.writerow([row["method"], row["year"], row["ubnormal_micro_auc"], row["shanghaitech_frame_auc"], row["ucf_crime_frame_auc"], row["source"]])
        writer.writerow(["Proposed method", "current", f"{proposed['UBnormal']:.4f}", f"{proposed['ShanghaiTech']:.4f}", f"{proposed['UCF-Crime']:.4f}", "Current run"])
    write_json(output_root / "tables" / "summary.json", {"results": results, "proposed": proposed})
    return {
        "stage2_checkpoint": str(stage_paths["stage2"]),
        "stage3_checkpoint": str(stage_paths["stage3"]),
        "stage4_checkpoint": str(stage_paths["stage4"]),
        "table": str(table_path),
        "figure": str(output_root / "figures" / "paper_main_results.png"),
    }


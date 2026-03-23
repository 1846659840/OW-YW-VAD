from pathlib import Path
import shutil

from owywvad.cli import main
from owywvad.utils import project_root, read_json


def _clean_runtime_dirs() -> None:
    root = project_root()
    for rel in ["data", "outputs"]:
        target = root / rel
        if target.exists():
            shutil.rmtree(target)


def test_toy_pipeline_end_to_end() -> None:
    _clean_runtime_dirs()
    assert main(["data", "fetch", "toy"]) == 0
    assert main(["data", "prepare", "all"]) == 0
    assert main(["cache", "build", "all"]) == 0
    assert main(["train", "stage2"]) == 0
    assert main(["train", "stage3"]) == 0
    assert main(["train", "stage4"]) == 0
    assert main(
        [
            "evaluate",
            "ubnormal",
            "--checkpoint",
            "outputs/paper_main/checkpoints/stage4.pt",
        ]
    ) == 0
    metrics = read_json(project_root() / "outputs" / "paper_main" / "metrics" / "ubnormal.json")
    assert "micro_auc" in metrics
    assert Path(project_root() / "outputs" / "paper_main" / "checkpoints" / "stage4.pt").exists()

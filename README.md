# OW-YW-VAD GitHub Reproduction Project

This repository is an executable reference project for the paper **OW-YW-VAD: Open-World YOLO-World-Guided Video Anomaly Detection**. It provides one CLI for dependency setup, dataset preparation, cache building, stage-wise training, inference, evaluation, and paper result reproduction.

## Paper Authors

Youxi Li, Xiangjun Chen, Liming Wang, Xiaocheng Huang, and Qiang Liu

## Repository Maintainer

Youxi Li

## Quickstart

```bash
python -m pip install -r requirements-lock.txt
python -m owywvad deps install
python -m owywvad data fetch toy
python -m owywvad data prepare all --config configs/paper_main.yaml
python -m owywvad cache build all --config configs/paper_main.yaml
python -m owywvad train stage2 --config configs/paper_main.yaml
python -m owywvad train stage3 --config configs/paper_main.yaml
python -m owywvad train stage4 --config configs/paper_main.yaml
python -m owywvad reproduce paper-main --config configs/paper_main.yaml
```

The `toy` fetch command creates a small smoke-test dataset pack under the three canonical benchmark names. It is intended for installation checks and CI. The paper-level results require the real datasets and the paper hardware setting.

## Official Commands

```bash
python -m owywvad deps install
python -m owywvad data fetch <dataset|all>
python -m owywvad data prepare <dataset|all> --config configs/paper_main.yaml
python -m owywvad cache build <dataset|all> --config configs/paper_main.yaml
python -m owywvad train stage2|stage3|stage4 --config configs/paper_main.yaml
python -m owywvad infer video --input <path> --checkpoint <ckpt> --config configs/paper_main.yaml
python -m owywvad evaluate <dataset> --checkpoint <ckpt> --config configs/paper_main.yaml
python -m owywvad reproduce paper-main --config configs/paper_main.yaml
```

## Project Layout

```text
OW-YW-VAD/
  configs/
  owywvad/
  tests/
  baseline_results.csv
  deps.yaml
  environment.yml
  pyproject.toml
```

## Dataset Flow

The project follows a semi-automatic and license-safe workflow.

- Public direct resources can be downloaded automatically when a URL is available.
- Gated datasets can be supplied through `--source`, `OWYWVAD_<DATASET>_SOURCE`, or manual placement inside `data/raw/<dataset>/`.
- The CLI validates directory structure and split metadata before continuing.

## Paper Defaults

The main configuration matches the paper defaults:

- `Python 3.10`, `PyTorch 2.1`
- single `NVIDIA A100 40GB`
- `12 FPS`, `640x640`
- `AdamW`, `lr=1e-4`, `weight_decay=1e-4`, `batch_size=16`
- `L=16`, `J=512`, `K=5`, `tau_u=0.5`, `tau_m=0.07`
- `alpha/beta/gamma/delta = 0.40/0.25/0.20/0.15`
- `tau_a/tau_c = 0.50/0.60`
- `lambda1..5 = 1.0/1.0/0.1/0.5/0.2`
- `ByteTrack high/low/match = 0.6/0.1/0.8`
- stage epochs `20/30/20`, warm-up `3`, grad clip `5.0`

## One-Video Inference

```bash
python -m owywvad infer video \
  --input path/to/video_or_npz \
  --checkpoint outputs/paper_main/checkpoints/stage4.pt \
  --config configs/paper_main.yaml
```

Outputs are written to `outputs/<run_name>/predictions/` and `outputs/<run_name>/figures/`.

## Reproducing the Paper

`python -m owywvad reproduce paper-main --config configs/paper_main.yaml` performs:

1. dependency validation
2. dataset availability checks
3. stage-wise training
4. evaluation on UBnormal, ShanghaiTech, and UCF-Crime
5. export of result tables and summary figures

The baseline comparison table is read from `baseline_results.csv`. The proposed method row is generated from the current run.

## Common Issues

- If a dataset is gated, pass `--source /path/to/archive_or_folder`.
- If external dependencies are missing, the wrappers fall back to a deterministic reference backend. This is enough for smoke tests, not for paper-level accuracy.
- If `python -m owywvad` cannot import the package, run the command from inside `githubcode/` or install the package with `pip install -e .`.

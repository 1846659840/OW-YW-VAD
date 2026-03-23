from __future__ import annotations

import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from owywvad.config import AppConfig
from owywvad.data.datasets import CachedVideoDataset
from owywvad.losses.objectives import compute_scores, memory_compactness_loss, mil_ranking_loss, smoothness_loss
from owywvad.memory.bank import MemoryBank
from owywvad.models.model import build_model
from owywvad.priors.dynamics import DynamicsPrior
from owywvad.utils import ensure_dir, save_yaml


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collate(batch: list[dict[str, torch.Tensor | int | str]]) -> dict[str, torch.Tensor]:
    keys = ["frame_features", "prompt_scores", "boxes", "region_ids", "frame_labels", "pixel_masks"]
    result = {key: torch.stack([item[key] for item in batch]) for key in keys}
    result["video_label"] = torch.tensor([int(item["video_label"]) for item in batch], dtype=torch.long)
    result["known_class"] = torch.tensor([int(item["known_class"]) for item in batch], dtype=torch.long)
    return result


def _checkpoint_dir(config: AppConfig) -> Path:
    return ensure_dir(config.resolve().outputs / config.run.name / "checkpoints")


def _load_checkpoint_if_any(stage: str, config: AppConfig) -> tuple[torch.nn.Module, MemoryBank, DynamicsPrior]:
    ckpt_dir = _checkpoint_dir(config)
    model = build_model(config)
    memory_bank = MemoryBank(neighbors=config.model.memory_neighbors)
    dynamics_prior = DynamicsPrior()
    if stage == "stage2":
        return model, memory_bank, dynamics_prior
    previous = ckpt_dir / ("stage2.pt" if stage == "stage3" else "stage3.pt")
    if previous.exists():
        payload = torch.load(previous, map_location="cpu")
        model = build_model(
            config,
            input_dim=payload.get("input_dim", 25),
            prompt_dim=payload.get("prompt_dim", 13),
        )
        model.load_state_dict(payload["model"])
        memory_bank = MemoryBank.from_state_dict(payload.get("memory_bank"))
        dynamics_prior = DynamicsPrior.from_state_dict(payload.get("dynamics_prior"))
    return model, memory_bank, dynamics_prior


def _fit_bank_and_prior(model: torch.nn.Module, loader: DataLoader, config: AppConfig) -> tuple[MemoryBank, DynamicsPrior]:
    model.eval()
    encoded_rows = []
    raw_rows = []
    region_rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["frame_features"], batch["prompt_scores"])
            mask = batch["frame_labels"] == 0
            if mask.any():
                encoded_rows.append(outputs["encoded"][mask])
                raw_rows.append(batch["frame_features"][mask])
                region_rows.append(batch["region_ids"][mask])
    bank = MemoryBank(neighbors=config.model.memory_neighbors)
    prior = DynamicsPrior()
    if encoded_rows:
        encoded = torch.cat(encoded_rows, dim=0)
        raw = torch.cat(raw_rows, dim=0)
        regions = torch.cat(region_rows, dim=0)
        bank.fit(encoded, num_prototypes=config.model.memory_prototypes)
        prior.fit(raw, regions)
    return bank, prior


def train_stage(stage: str, config: AppConfig) -> dict[str, float | str]:
    _seed_everything(config.run.seed)
    resolved = config.resolve()
    dataset_name = {"stage2": config.datasets.stage2, "stage3": config.datasets.stage3, "stage4": config.datasets.stage4}[stage]
    dataset = CachedVideoDataset(resolved.cache_data / dataset_name / "index.jsonl", split="train")
    if stage == "stage2":
        dataset.rows = [row for row in dataset.rows if row.video_label == 0]
    if len(dataset) == 0:
        raise RuntimeError(f"No cached training samples for {dataset_name}")
    sample = dataset[0]
    model, memory_bank, dynamics_prior = _load_checkpoint_if_any(stage, config)
    if sample["frame_features"].shape[-1] != 25 or sample["prompt_scores"].shape[-1] != 13:
        model = build_model(config, input_dim=sample["frame_features"].shape[-1], prompt_dim=sample["prompt_scores"].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    loader = DataLoader(dataset, batch_size=min(config.optimizer.batch_size, len(dataset)), shuffle=True, collate_fn=_collate)
    epochs = getattr(config.training, f"{stage}_epochs")
    weights = config.training.loss_weights
    history = []
    start = time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["frame_features"], batch["prompt_scores"])
            score_dict = compute_scores(outputs, batch["prompt_scores"], batch["frame_features"], batch["region_ids"], config, memory_bank, dynamics_prior)
            if stage == "stage2":
                loss = weights.lambda3 * memory_compactness_loss(outputs["encoded"], config.model.memory_temperature) + weights.lambda5 * smoothness_loss(score_dict["final_score"])
            elif stage == "stage3":
                class_targets = torch.where(
                    batch["frame_labels"] > 0,
                    batch["known_class"].unsqueeze(1).expand_as(batch["frame_labels"]).clamp(min=1),
                    torch.zeros_like(batch["frame_labels"]),
                )
                loss = (
                    weights.lambda1 * F.cross_entropy(outputs["class_logits"].reshape(-1, outputs["class_logits"].shape[-1]), class_targets.reshape(-1))
                    + weights.lambda2 * F.binary_cross_entropy_with_logits(outputs["loc_logits"], batch["frame_labels"].float())
                    + weights.lambda3 * memory_compactness_loss(outputs["encoded"], config.model.memory_temperature)
                    + weights.lambda5 * smoothness_loss(score_dict["final_score"])
                )
            else:
                loss = weights.lambda4 * mil_ranking_loss(score_dict["final_score"], batch["video_label"]) + weights.lambda5 * smoothness_loss(score_dict["final_score"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
        if stage in {"stage2", "stage3"}:
            memory_bank, dynamics_prior = _fit_bank_and_prior(model, loader, config)
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, len(loader))})
    ckpt_path = _checkpoint_dir(config) / f"{stage}.pt"
    torch.save(
        {
            "stage": stage,
            "model": model.state_dict(),
            "memory_bank": memory_bank.state_dict(),
            "dynamics_prior": dynamics_prior.state_dict(),
            "input_dim": sample["frame_features"].shape[-1],
            "prompt_dim": sample["prompt_scores"].shape[-1],
        },
        ckpt_path,
    )
    log_dir = ensure_dir(resolved.outputs / config.run.name / "logs")
    save_yaml(log_dir / f"{stage}.yaml", {"stage": stage, "dataset": dataset_name, "epochs": epochs, "history": history})
    save_yaml(resolved.outputs / config.run.name / "run_manifest.yaml", {"run_name": config.run.name, "last_stage": stage, "checkpoint": str(ckpt_path)})
    return {"stage": stage, "dataset": dataset_name, "checkpoint": str(ckpt_path), "seconds": round(time() - start, 3)}


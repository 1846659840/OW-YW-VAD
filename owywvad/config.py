from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from owywvad.utils import load_yaml, project_root


class RunConfig(BaseModel):
    name: str = "paper_main"
    seed: int = 7
    device: str = "cpu"
    mixed_precision: bool = True
    num_workers: int = 0


class PathsConfig(BaseModel):
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    cache_data: str = "data/cache"
    outputs: str = "outputs"
    external: str = "external"
    baselines_csv: str = "baseline_results.csv"


class DatasetStageConfig(BaseModel):
    stage2: str
    stage3: str
    stage4: str
    all: list[str]


class PromptConfig(BaseModel):
    file: str


class FusionWeights(BaseModel):
    alpha: float
    beta: float
    gamma: float
    delta: float


class DecisionThresholds(BaseModel):
    tau_a: float
    tau_c: float


class ModelConfig(BaseModel):
    fps: int
    input_size: tuple[int, int]
    num_known_classes: int = 3
    trajectory_length: int
    tcn_dilations: tuple[int, int, int]
    hidden_dim: int
    memory_prototypes: int
    memory_neighbors: int
    uncertainty_temperature: float
    memory_temperature: float
    fusion_weights: FusionWeights
    decision_thresholds: DecisionThresholds

    @field_validator("input_size", mode="before")
    @classmethod
    def _normalize_size(cls, value: object) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        raise ValueError("input_size must contain two integers")

    @field_validator("tcn_dilations", mode="before")
    @classmethod
    def _normalize_dilations(cls, value: object) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return int(value[0]), int(value[1]), int(value[2])
        raise ValueError("tcn_dilations must contain three integers")


class OptimizerConfig(BaseModel):
    name: Literal["AdamW"] = "AdamW"
    lr: float
    weight_decay: float
    batch_size: int
    grad_clip: float
    warmup_epochs: int


class TrackerConfig(BaseModel):
    high_conf: float
    low_conf: float
    matching_threshold: float


class LossWeights(BaseModel):
    lambda1: float
    lambda2: float
    lambda3: float
    lambda4: float
    lambda5: float


class TrainingConfig(BaseModel):
    stage2_epochs: int
    stage3_epochs: int
    stage4_epochs: int
    mil_pooling: str
    ucf_segments: int
    loss_weights: LossWeights


class AppConfig(BaseModel):
    run: RunConfig
    paths: PathsConfig
    datasets: DatasetStageConfig
    prompts: PromptConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    tracker: TrackerConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def _validate_weights(self) -> "AppConfig":
        weights = self.model.fusion_weights
        total = weights.alpha + weights.beta + weights.gamma + weights.delta
        if abs(total - 1.0) > 1e-6:
            raise ValueError("fusion weights must sum to 1.0")
        return self

    def resolve(self, root: Path | None = None) -> "ResolvedConfig":
        base = root or project_root()
        return ResolvedConfig(
            root=base,
            raw_data=base / self.paths.raw_data,
            processed_data=base / self.paths.processed_data,
            cache_data=base / self.paths.cache_data,
            outputs=base / self.paths.outputs,
            external=base / self.paths.external,
            baselines_csv=base / self.paths.baselines_csv,
            prompt_file=base / self.prompts.file,
        )


class ResolvedConfig(BaseModel):
    root: Path
    raw_data: Path
    processed_data: Path
    cache_data: Path
    outputs: Path
    external: Path
    baselines_csv: Path
    prompt_file: Path

    model_config = {"arbitrary_types_allowed": True}


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    return AppConfig.model_validate(load_yaml(cfg_path))

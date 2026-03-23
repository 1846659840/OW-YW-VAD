from owywvad.config import load_config


def test_paper_config_matches_defaults() -> None:
    config = load_config("configs/paper_main.yaml")
    assert config.model.fps == 12
    assert config.model.input_size == (640, 640)
    assert config.optimizer.name == "AdamW"
    assert config.optimizer.lr == 1e-4
    assert config.optimizer.weight_decay == 1e-4
    assert config.optimizer.batch_size == 16
    assert config.model.trajectory_length == 16
    assert config.model.tcn_dilations == (1, 2, 4)
    assert config.model.hidden_dim == 256
    assert config.model.memory_prototypes == 512
    assert config.model.memory_neighbors == 5
    assert config.model.uncertainty_temperature == 0.5
    assert config.model.memory_temperature == 0.07
    assert config.model.fusion_weights.alpha == 0.4
    assert config.model.fusion_weights.beta == 0.25
    assert config.model.fusion_weights.gamma == 0.2
    assert config.model.fusion_weights.delta == 0.15
    assert config.model.decision_thresholds.tau_a == 0.5
    assert config.model.decision_thresholds.tau_c == 0.6
    assert config.training.loss_weights.lambda1 == 1.0
    assert config.training.loss_weights.lambda5 == 0.2
    assert config.tracker.high_conf == 0.6
    assert config.tracker.low_conf == 0.1
    assert config.tracker.matching_threshold == 0.8
    assert config.training.stage2_epochs == 20
    assert config.training.stage3_epochs == 30
    assert config.training.stage4_epochs == 20


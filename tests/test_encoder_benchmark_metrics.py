import numpy as np
import pytest
import torch

from callbacks.encoder_benchmark import (
    EncoderBenchmarkCallback,
    embedding_health_metrics,
    loss_stability_metrics,
)


def test_embedding_health_accepts_two_dimensional_cloud():
    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(500, 2))
    metrics = embedding_health_metrics(embedding)
    assert metrics["collapsed"] == 0.0
    assert metrics["nonfinite_fraction"] == 0.0
    assert metrics["std_min"] > 0.5
    assert metrics["axis_ratio"] > 0.5
    assert metrics["near_line"] == 0.0


def test_embedding_health_flags_point_line_and_nonfinite_collapse():
    x = np.linspace(-1.0, 1.0, 100)
    line = np.column_stack([x, np.zeros_like(x)])
    assert embedding_health_metrics(line)["collapsed"] == 1.0

    point = np.zeros((100, 2))
    assert embedding_health_metrics(point)["collapsed"] == 1.0

    cloud = np.column_stack([x, x**2])
    cloud[0, 0] = np.nan
    metrics = embedding_health_metrics(cloud)
    assert metrics["collapsed"] == 1.0
    assert metrics["nonfinite_fraction"] > 0.0


def test_embedding_health_flags_near_line_before_strict_collapse():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1000)
    near_line = np.column_stack([x, x + rng.normal(scale=0.02, size=x.shape)])
    metrics = embedding_health_metrics(near_line)

    assert 1.0e-3 < metrics["axis_ratio"] < 5.0e-2
    assert metrics["near_line"] == 1.0
    assert metrics["collapsed"] == 0.0


def test_embedding_health_reports_extreme_norm_and_trimmed_axis_ratio():
    rng = np.random.default_rng(7)
    bulk = rng.normal(scale=0.01, size=(999, 2))
    embedding = np.vstack([bulk, [[1000.0, -1000.0]]])
    metrics = embedding_health_metrics(embedding)

    assert metrics["norm_max_median_ratio"] > 1000.0
    assert metrics["axis_ratio_trim_top_0_1pct"] > metrics["axis_ratio"]


def test_loss_stability_distinguishes_stable_and_oscillating_series():
    stable = loss_stability_metrics([1.0, 0.9, 0.8, 0.7, 0.6])
    assert stable["oscillation_detected"] is False
    assert stable["direction_change_rate"] == 0.0

    oscillating = loss_stability_metrics([1.0, 2.0, 0.5, 2.5, 0.4, 3.0])
    assert oscillating["oscillation_detected"] is True
    assert oscillating["direction_change_rate"] > 0.5


def test_callback_writes_required_capacity_and_final_loss_summary():
    class Experiment:
        summary = {}

    class Logger:
        experiment = Experiment()

        def log_metrics(self, metrics, step=None):
            self.experiment.summary.update(metrics)

    class DataModule:
        batch_size = 4096

    class Trainer:
        logger = Logger()
        datamodule = DataModule()
        global_step = 1
        is_global_zero = True

    module = torch.nn.Linear(3, 2)
    module.encoder_capacity = {
        "params": 8,
        "baseline_params": 16,
        "param_ratio": 0.5,
        "param_in_band": True,
    }

    callback = EncoderBenchmarkCallback(check_grad_every_n_steps=0)
    trainer = Trainer()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    module._latest_training_losses = {
        "manifold_loss": torch.tensor(1.25),
        "density_loss": torch.tensor(0.5),
        "total_loss": torch.tensor(1.75),
    }
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_epoch_end(trainer, module)
    callback.on_fit_end(trainer, module)

    summary = trainer.logger.experiment.summary
    assert summary["engineering/actual_batch_size"] == 4096
    assert summary["engineering/encoder_params"] == 8
    assert summary["train_status/final_manifold_loss"] == 1.25
    assert summary["train_status/final_density_loss"] == 0.5
    assert summary["train_status/nonfinite_detected"] == 0


def test_callback_records_first_nonfinite_loss_and_fails_fast():
    class Trainer:
        logger = None
        current_epoch = 7
        global_step = 11
        should_stop = False

    callback = EncoderBenchmarkCallback(
        check_grad_every_n_steps=1, fail_on_nonfinite=True
    )
    trainer = Trainer()
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        callback.on_before_backward(
            trainer, torch.nn.Linear(2, 2), torch.tensor(float("nan"))
        )

    assert trainer.should_stop is True
    assert callback._first_nonfinite_kind == "loss"
    assert callback._first_nonfinite_name == "training_loss"
    assert callback._first_nonfinite_epoch == 7
    assert callback._first_nonfinite_step == 11


def test_callback_allows_grad_scaler_to_recover_from_amp_overflow():
    class Scaler:
        def get_scale(self):
            return 65536.0

    class PrecisionPlugin:
        scaler = Scaler()

    class Trainer:
        logger = None
        precision_plugin = PrecisionPlugin()
        current_epoch = 10
        global_step = 263
        should_stop = False

    module = torch.nn.Linear(2, 2)
    module.weight.grad = torch.full_like(module.weight, float("inf"))
    callback = EncoderBenchmarkCallback(
        check_grad_every_n_steps=1, fail_on_nonfinite=True
    )
    trainer = Trainer()

    callback.on_before_optimizer_step(trainer, module, object())

    assert trainer.should_stop is False
    assert callback._amp_overflow_count == 1
    assert callback._first_amp_overflow_parameter == "weight"
    assert callback._nonfinite_gradient is False


def test_callback_still_fails_on_non_amp_nonfinite_gradient():
    class Trainer:
        logger = None
        precision_plugin = object()
        current_epoch = 3
        global_step = 9
        should_stop = False

    module = torch.nn.Linear(2, 2)
    module.weight.grad = torch.full_like(module.weight, float("nan"))
    callback = EncoderBenchmarkCallback(
        check_grad_every_n_steps=1, fail_on_nonfinite=True
    )
    trainer = Trainer()

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        callback.on_before_optimizer_step(trainer, module, object())

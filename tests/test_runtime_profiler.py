from types import SimpleNamespace

from callbacks.runtime_profiler import RuntimeProfilerCallback


def test_runtime_row_separates_training_and_data_seeds(monkeypatch):
    monkeypatch.setenv("PL_GLOBAL_SEED", "44")
    callback = RuntimeProfilerCallback()
    callback._epoch_times = [3.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.7]

    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            seed=42,
            sample_data_size=5000,
            batch_size=4096,
            dataset=list(range(5000)),
        ),
        logger=SimpleNamespace(
            experiment=SimpleNamespace(
                name="runtime_latent_hcl_n5000_seed44",
                config={"seed_everything": 44},
            ),
            name="project-name",
        ),
        max_epochs=150,
        current_epoch=149,
        global_step=300,
    )
    module = SimpleNamespace(hparams={})

    row = callback._row(trainer, module, fit_wall_time_sec=123.0)

    assert row["seed"] == 44
    assert row["training_seed"] == 44
    assert row["data_seed"] == 42
    assert row["method"] == "latent"
    assert row["data_name"] == "hcl"
    assert row["steady_epoch_time_sec"] == "0.600000"

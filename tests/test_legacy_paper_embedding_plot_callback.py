from pathlib import Path
from types import SimpleNamespace

import numpy as np

import callbacks.legacy_paper_embedding_plot_callback as paper_plot


def test_original_paper_renderer_saves_png_and_npz(tmp_path):
    callback = paper_plot.PaperEmbeddingPlotCallback(
        output_dir=str(tmp_path),
        dataset_name="TEST",
        method_name="latent-bn",
        every_n_epochs=1000,
        max_plot_samples=50000,
        save_formats=("png",),
        save_npz=True,
        scale_down_factor=3.0,
        dpi=120,
        base_point_size=3.5,
        min_point_size=1.5,
        alpha=0.85,
        edge_color="none",
    )
    embedding = np.asarray(
        [[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "a", "b", "b"])

    saved = callback._save_paper_figure(
        embedding=embedding,
        labels=labels,
        epoch_num=1000,
        layer=0,
    )

    assert {path.suffix for path in saved} == {".png", ".npz"}
    assert all(path.exists() and path.stat().st_size > 0 for path in saved)
    archive = np.load(next(path for path in saved if path.suffix == ".npz"))
    np.testing.assert_allclose(archive["embedding"], embedding)
    np.testing.assert_array_equal(archive["labels"], labels)


def test_original_paper_renderer_logs_expected_wandb_panel(monkeypatch, tmp_path):
    image_path = Path(tmp_path) / "paper.png"
    image_path.write_bytes(b"png")
    npz_path = Path(tmp_path) / "paper.npz"
    npz_path.write_bytes(b"npz")
    logged = []

    class FakeRun:
        step = 12

        def log(self, payload, step):
            logged.append((payload, step))

    fake_run = FakeRun()
    trainer = SimpleNamespace(
        logger=SimpleNamespace(experiment=fake_run),
        global_step=10,
    )
    monkeypatch.setattr(
        paper_plot,
        "wandb",
        SimpleNamespace(Image=lambda path: ("image", path)),
    )
    callback = paper_plot.PaperEmbeddingPlotCallback(
        output_dir=str(tmp_path),
        log_to_wandb=True,
    )

    callback._log_to_wandb(trainer, [image_path, npz_path], epoch_num=1000)

    assert len(logged) == 1
    payload, step = logged[0]
    assert payload["paper_embedding/figure"] == ("image", str(image_path))
    assert payload["paper_embedding/npz_path"] == str(npz_path)
    assert step == 12

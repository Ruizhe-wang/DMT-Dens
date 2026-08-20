from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


PAPER_CONFIG_ROOTS = {
    Path("configs/publication"),
    Path("configs/encoder_bench/sweep_e18/runs"),
    Path("configs/encoder_bench/act_bs4096_5seed_1000/runs"),
    Path("configs/encoder_bench/ablation_single_factor_loo_3seed_1000/runs"),
    Path("configs/encoder_bench/case_study_latent_5seed/runs"),
    Path("configs/baseline_with_best_hyperparameter"),
    Path("configs/runtime/latent_transformer_150epoch/runs"),
}


def test_public_smoke_config_has_relative_data_path():
    config_path = ROOT / "configs" / "publication" / "toy_cpu.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data"]["init_args"]["data_path"])
    assert not data_path.is_absolute()
    assert config["data"]["init_args"]["uselabel"] is False
    assert config["model"]["init_args"]["encoder_type"] == "latent_transformer"
    assert config["trainer"]["accelerator"] == "cpu"
    assert config["trainer"]["logger"] is False


def test_citation_metadata_lists_manuscript_authors():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["title"].startswith("DMT-Dens:")
    assert len(citation["authors"]) == 6
    assert citation["repository-code"].endswith("/DMT-Dens")


def test_public_repository_metadata_and_ci_are_present():
    required = [
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0, relative


def test_documented_resolved_config_families_are_complete():
    bench = ROOT / "configs" / "encoder_bench"
    assert len(list((bench / "sweep_e18" / "runs").glob("*_latent_bn_seed*.yaml"))) == 45
    assert len(list((bench / "act_bs4096_5seed_1000" / "runs").glob("*.yaml"))) == 5
    assert len(
        list((bench / "ablation_single_factor_loo_3seed_1000" / "runs").glob("*.yaml"))
    ) == 54
    assert len(list((bench / "case_study_latent_5seed" / "runs").glob("*.yaml"))) == 20


def test_retained_configs_are_portable_and_use_local_logging():
    config_paths = sorted((ROOT / "configs").rglob("*.yaml"))
    assert len(config_paths) == 322

    for config_path in config_paths:
        relative = config_path.relative_to(ROOT)
        assert any(relative.is_relative_to(root) for root in PAPER_CONFIG_ROOTS)

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data_path = config.get("data", {}).get("init_args", {}).get("data_path")
        if data_path is not None:
            assert not Path(str(data_path)).is_absolute(), relative

        logger = config.get("trainer", {}).get("logger")
        if isinstance(logger, dict):
            assert logger["class_path"] == "lightning.pytorch.loggers.CSVLogger"


def test_retained_configs_have_public_facing_run_names():
    internal_run_suffix = re.compile(r"_c\d{2}_20\d{6}")

    for config_path in sorted((ROOT / "configs").rglob("*.yaml")):
        relative = config_path.relative_to(ROOT)
        assert internal_run_suffix.search(relative.as_posix()) is None, relative
        assert internal_run_suffix.search(
            config_path.read_text(encoding="utf-8")
        ) is None, relative


def test_public_tree_has_no_experiment_tracking_integration():
    text_paths = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "main.py",
        *sorted((ROOT / "callbacks").glob("*.py")),
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    forbidden = "".join(path.read_text(encoding="utf-8").lower() for path in text_paths)
    tracker_package = "wan" + "db"
    assert tracker_package not in forbidden


def test_mnist_example_is_download_to_figure_pipeline():
    script = ROOT / "scripts" / "run_mnist_example.py"
    config = yaml.safe_load(
        (ROOT / "configs" / "publication" / "mnist_example.yaml").read_text(
            encoding="utf-8"
        )
    )
    script_text = script.read_text(encoding="utf-8")

    assert "MNIST(root=RAW_DATA_DIR, train=True, download=True)" in script_text
    assert "subprocess.run(command" in script_text
    assert "dmt_dens_mnist_layer0_final_annotation_final.png" in script_text
    assert config["data"]["init_args"]["uselabel"] is False
    assert config["model"]["class_path"].endswith("DMTEVT_model")
    callback = config["trainer"]["callbacks"][0]
    assert callback["class_path"].endswith("PaperEmbeddingCallback")


def test_toy_generator_produces_valid_npz(tmp_path):
    output = tmp_path / "toy.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_toy_data.py"),
            "--output",
            str(output),
            "--samples",
            "90",
            "--features",
            "12",
        ],
        cwd=ROOT,
        check=True,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["X"].shape == (90, 12)
        assert archive["y"].shape == (90,)
        assert archive["latent"].shape == (90, 3)
        assert np.isfinite(archive["X"]).all()

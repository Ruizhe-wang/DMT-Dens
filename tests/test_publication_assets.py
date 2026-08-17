from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_smoke_config_has_relative_data_path():
    config_path = ROOT / "configs" / "publication" / "toy_cpu.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data"]["init_args"]["data_path"])
    assert not data_path.is_absolute()
    assert config["data"]["init_args"]["uselabel"] is False
    assert config["model"]["init_args"]["encoder_type"] == "latent_transformer"
    assert config["trainer"]["accelerator"] == "cpu"


def test_citation_metadata_lists_manuscript_authors():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["title"].startswith("DMT-Dens:")
    assert len(citation["authors"]) == 6
    assert citation["repository-code"].endswith("/DMT-Dens")


def test_documented_resolved_config_families_are_complete():
    bench = ROOT / "configs" / "encoder_bench"
    assert len(list((bench / "sweep_e18" / "runs").glob("*_latent_bn_seed*.yaml"))) == 45
    assert len(list((bench / "act_bs4096_5seed_1000" / "runs").glob("*.yaml"))) == 5
    assert len(
        list((bench / "ablation_single_factor_loo_3seed_1000" / "runs").glob("*.yaml"))
    ) == 54
    assert len(list((bench / "case_study_latent_5seed" / "runs").glob("*.yaml"))) == 20


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

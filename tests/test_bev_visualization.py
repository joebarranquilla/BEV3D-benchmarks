from pathlib import Path

from bevbench.bev_visualization import _ensure_nuscenes_visualization_alias, default_output_dir


def test_default_output_dir() -> None:
    out = default_output_dir(Path("results"), "bevdepth")
    assert out == Path("results/visualizations/bevdepth")


def test_nuscenes_visualization_alias_creation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "nuscenes"
    dataset_root.mkdir(parents=True)
    info_file = dataset_root / "nuscenes_infos_val.pkl"
    info_file.write_text("stub", encoding="utf-8")

    repo_dir = tmp_path / "third_party" / "BEVDepth"
    repo_dir.mkdir(parents=True)

    _ensure_nuscenes_visualization_alias(repo_dir, tmp_path)

    alias = repo_dir / "data" / "nuScenes" / "nuscenes_12hz_infos_val.pkl"
    assert alias.exists()
    assert alias.is_symlink() or alias.read_text(encoding="utf-8") == "stub"

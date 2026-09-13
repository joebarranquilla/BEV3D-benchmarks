from pathlib import Path

from bevbench.bev_visualization import (
    _ensure_nuscenes_visualization_alias,
    default_output_dir,
    render_bevdepth_detections,
)


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


def test_render_uses_absolute_output_path(monkeypatch, tmp_path: Path) -> None:
    config = {
        "model": {"name": "BEVDepth", "repo_dir": "third_party/BEVDepth", "python_executable": "python"},
        "experiment": {"name": "demo-exp"},
    }

    dataset_root = tmp_path / "data" / "nuscenes"
    dataset_root.mkdir(parents=True)
    (dataset_root / "nuscenes_infos_val.pkl").write_text("stub", encoding="utf-8")

    repo_dir = tmp_path / "third_party" / "BEVDepth"
    repo_dir.mkdir(parents=True)
    script_path = repo_dir / "scripts" / "visualize_nusc.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('ok')\n", encoding="utf-8")
    (repo_dir / "outputs" / "demo-exp").mkdir(parents=True)
    result_file = repo_dir / "outputs" / "demo-exp" / "results_nusc.json"
    result_file.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_run(cmd, cwd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["cwd"] = str(cwd)
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("bevbench.bev_visualization.subprocess.run", fake_run)

    payload = render_bevdepth_detections(config, workdir=tmp_path, out_dir=tmp_path / "results" / "visualizations" / "bevdepth")

    assert payload["status"] == "ok"
    assert captured["cmd"][-1].startswith(str((tmp_path / "results" / "visualizations" / "bevdepth").resolve()))

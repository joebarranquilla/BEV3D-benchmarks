from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any


def default_output_dir(results_root: Path, model_name: str) -> Path:
    return results_root / "visualizations" / model_name.lower()


def _resolve_model_repo(config: dict[str, Any], workdir: Path) -> Path:
    repo_dir = Path(config["model"]["repo_dir"])
    if not repo_dir.is_absolute():
        repo_dir = (workdir / repo_dir).resolve()
    return repo_dir


def _ensure_nuscenes_visualization_alias(repo_dir: Path, workdir: Path) -> None:
    dataset_root = (workdir / "data" / "nuscenes").resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"NuScenes dataset not found at {dataset_root}. Run the dataset download step first.")

    repo_data_dir = repo_dir / "data"
    repo_data_dir.mkdir(parents=True, exist_ok=True)

    target_root = repo_data_dir / "nuScenes"
    if not target_root.exists():
        try:
            target_root.symlink_to(dataset_root)
        except OSError:
            # Fallback for systems that do not allow symlinks.
            import shutil

            if target_root.exists():
                return
            shutil.copytree(dataset_root, target_root, dirs_exist_ok=True)

    legacy_info = target_root / "nuscenes_12hz_infos_val.pkl"
    real_info = dataset_root / "nuscenes_infos_val.pkl"
    if real_info.exists() and not legacy_info.exists():
        legacy_info.symlink_to(real_info)


def find_bevdepth_result_json(repo_dir: Path, exp_name: str | None = None) -> Path:
    search_root = repo_dir / "outputs"
    if exp_name:
        search_root = search_root / exp_name

    matches = sorted(search_root.rglob("results_nusc.json")) if search_root.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Could not find BEVDepth detection JSON under {search_root}. "
            "Run BEVDepth evaluation first so it writes results_nusc.json."
        )
    return matches[-1]


def render_bevdepth_detections(
    config: dict[str, Any],
    workdir: Path,
    sample_idx: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    model_name = str(config["model"]["name"]).lower()
    if "bevdepth" not in model_name:
        return {
            "model_name": config["model"]["name"],
            "status": "skipped",
            "reason": "Visualization is currently implemented for BEVDepth only.",
        }

    repo_dir = _resolve_model_repo(config, workdir)
    _ensure_nuscenes_visualization_alias(repo_dir, workdir)
    result_json = find_bevdepth_result_json(repo_dir, config["experiment"].get("name"))

    target_dir = out_dir if out_dir is not None else default_output_dir(workdir / "results", config["model"]["name"])
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / f"{config['experiment']['name']}_{sample_idx}.jpg"
    script_path = repo_dir / "scripts" / "visualize_nusc.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find visualization script at {script_path}.")

    python_bin = str(config["model"].get("python_executable", "python"))
    cmd = [
        python_bin,
        str(script_path),
        str(sample_idx),
        str(result_json),
        str(target_path),
    ]
    proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, check=False)

    return {
        "model_name": config["model"]["name"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "result_json": str(result_json),
        "image_path": str(target_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a BEVDepth detection snapshot from a saved NuScenes result JSON")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/bevdepth.yaml"),
        help="Path to the BEVDepth model config used to locate the repo and output dir.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Optional explicit result JSON path. If omitted, it is discovered under the model output directory.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="NuScenes sample index to visualize.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/visualizations/bevdepth"),
        help="Directory where the generated .jpg image will be saved.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Optional shared config for merged model metadata.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    base_cfg = _load_yaml(args.base_config)
    model_cfg = _load_yaml(args.model_config)
    cfg = _merge_dicts(base_cfg, model_cfg)

    result_json = args.result_json
    workdir = Path.cwd()
    model_name = str(cfg["model"]["name"]).lower()
    if result_json is None:
        repo_dir = _resolve_model_repo(cfg, workdir)
        result_json = find_bevdepth_result_json(repo_dir, cfg["experiment"].get("name"))
    else:
        result_json = result_json.resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (workdir / out_dir).resolve()

    target_path = out_dir / f"{cfg['experiment']['name']}_{args.sample_idx}.jpg"
    repo_dir = _resolve_model_repo(cfg, workdir)
    _ensure_nuscenes_visualization_alias(repo_dir, workdir)
    script_path = repo_dir / "scripts" / "visualize_nusc.py"
    python_bin = str(cfg["model"].get("python_executable", "python"))
    cmd = [
        python_bin,
        str(script_path),
        str(args.sample_idx),
        str(result_json),
        str(target_path),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, check=False)

    print(f"Result JSON: {result_json}")
    print(f"Image output: {target_path}")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)

    print(proc.stdout.strip())
    print(f"Saved BEV visualization to {target_path}")


if __name__ == "__main__":
    main()

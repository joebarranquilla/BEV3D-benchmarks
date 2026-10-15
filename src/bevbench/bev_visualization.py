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


def _ensure_nuscenes_visualization_alias(repo_dir: Path, workdir: Path, dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_root}. Run the dataset preparation step first.")

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


def find_result_json_generic(workdir: Path, repo_dir: Path, exp_name: str | None = None) -> Path:
    # Search common locations for results_nusc.json for different model repos
    candidates = []
    search_paths = [repo_dir / "outputs", workdir / "test", workdir / "results", repo_dir]
    for p in search_paths:
        if not p.exists():
            continue
        for match in p.rglob("results_nusc.json"):
            if exp_name is None or exp_name in str(match):
                candidates.append(match)

    # Fallback: global scan under workdir
    if not candidates:
        for match in (workdir).rglob("results_nusc.json"):
            if exp_name is None or exp_name in str(match):
                candidates.append(match)

    if not candidates:
        # Try to find any results_nusc.json and pick the newest
        all_matches = list((workdir).rglob("results_nusc.json"))
        if not all_matches:
            raise FileNotFoundError(
                f"Could not find a results_nusc.json for experiment {exp_name} under {repo_dir} or workspace."
            )
        return sorted(all_matches)[-1]
    return sorted(candidates)[-1]


def render_bevdepth_detections(
    config: dict[str, Any],
    workdir: Path,
    sample_idx: int = 2,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    model_name = str(config["model"]["name"]).lower()
    is_bevdepth = "bevdepth" in model_name
    is_bevformer = "bevformer" in model_name
    if not (is_bevdepth or is_bevformer):
        return {
            "model_name": config["model"]["name"],
            "status": "skipped",
            "reason": "Visualization is implemented for BEVDepth and BEVFormer only.",
        }

    repo_dir = _resolve_model_repo(config, workdir)
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = (workdir / dataset_root).resolve()
    _ensure_nuscenes_visualization_alias(repo_dir, workdir, dataset_root)
    # Locate the results JSON. BEVDepth places it under its repo outputs; BEVFormer may place it under test/ dirs.
    if is_bevdepth:
        result_json = find_bevdepth_result_json(repo_dir, config["experiment"].get("name"))
    else:
        result_json = find_result_json_generic(workdir, repo_dir, config["experiment"].get("name"))

    target_dir = out_dir if out_dir is not None else default_output_dir(workdir / "results", config["model"]["name"])
    target_dir = Path(target_dir)
    if not target_dir.is_absolute():
        target_dir = (workdir / target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = (target_dir / f"{config['experiment']['name']}_{sample_idx}.jpg").resolve()
    # Choose visualization script. Prefer BEVDepth's visualizer which understands results_nusc.json format.
    if is_bevdepth:
        script_path = repo_dir / "scripts" / "visualize_nusc.py"
    else:
        # If using BEVDepth visualizer to render BEVFormer outputs, ensure BEVDepth repo has dataset alias
        bevdepth_repo = (workdir / "third_party" / "BEVDepth").resolve()
        _ensure_nuscenes_visualization_alias(bevdepth_repo, workdir, dataset_root)
        script_path = (bevdepth_repo / "scripts" / "visualize_nusc.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find visualization script at {script_path}.")

    python_bin = str(config["model"].get("python_executable", "python"))
    # If visualizing BEVFormer outputs using BEVDepth visualizer, prefer BEVDepth's python to satisfy deps
    if is_bevformer:
        bevdepth_cfg_path = workdir / "configs" / "models" / "truckscenes" / "bevdepth.yaml"
        if bevdepth_cfg_path.exists():
            try:
                bevdepth_cfg = _load_yaml(bevdepth_cfg_path)
                python_bin = str(bevdepth_cfg.get("model", {}).get("python_executable", python_bin))
            except Exception:
                pass
    val_info_name = "nuscenes_infos_val.pkl"
    dataset_name = str(config.get("dataset", {}).get("name", "")).lower()
    if dataset_name == "nuscenes":
        # Existing BEVDepth script expects this alias name for nuScenes visualization.
        val_info_name = "nuscenes_12hz_infos_val.pkl"

    cmd = [
        python_bin,
        str(script_path),
        str(sample_idx),
        str(result_json),
        str(target_path),
        "--data-root",
        str(dataset_root),
        "--val-info",
        val_info_name,
    ]
    # If we're using BEVDepth visualizer for BEVFormer outputs, ensure its repo is on PYTHONPATH
    env = None
    proc_cwd = repo_dir
    if is_bevformer:
        bevdepth_repo = (workdir / "third_party" / "BEVDepth").resolve()
        env = dict(**__import__("os").environ)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(bevdepth_repo) + (":" + prev if prev else "")
        proc_cwd = bevdepth_repo

    proc = subprocess.run(cmd, cwd=proc_cwd, env=env, capture_output=True, text=True, check=False)

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
    is_bevdepth = "bevdepth" in model_name
    is_bevformer = "bevformer" in model_name
    repo_dir = _resolve_model_repo(cfg, workdir)
    if result_json is None:
        if is_bevdepth:
            result_json = find_bevdepth_result_json(repo_dir, cfg["experiment"].get("name"))
        else:
            result_json = find_result_json_generic(workdir, repo_dir, cfg["experiment"].get("name"))
    else:
        result_json = result_json.resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (workdir / out_dir).resolve()

    target_path = (out_dir / f"{cfg['experiment']['name']}_{args.sample_idx}.jpg").resolve()
    dataset_root = Path(cfg["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = (workdir / dataset_root).resolve()
    _ensure_nuscenes_visualization_alias(repo_dir, workdir, dataset_root)
    if is_bevdepth:
        script_path = repo_dir / "scripts" / "visualize_nusc.py"
    else:
        bevdepth_repo = (workdir / "third_party" / "BEVDepth").resolve()
        _ensure_nuscenes_visualization_alias(bevdepth_repo, workdir, dataset_root)
        script_path = (bevdepth_repo / "scripts" / "visualize_nusc.py").resolve()
    python_bin = str(cfg["model"].get("python_executable", "python"))
    if is_bevformer:
        bevdepth_cfg_path = workdir / "configs" / "models" / "truckscenes" / "bevdepth.yaml"
        if bevdepth_cfg_path.exists():
            try:
                bevdepth_cfg = _load_yaml(bevdepth_cfg_path)
                python_bin = str(bevdepth_cfg.get("model", {}).get("python_executable", python_bin))
            except Exception:
                pass
    val_info_name = "nuscenes_infos_val.pkl"
    dataset_name = str(cfg.get("dataset", {}).get("name", "")).lower()
    if dataset_name == "nuscenes":
        val_info_name = "nuscenes_12hz_infos_val.pkl"

    cmd = [
        python_bin,
        str(script_path),
        str(args.sample_idx),
        str(result_json),
        str(target_path),
        "--data-root",
        str(dataset_root),
        "--val-info",
        val_info_name,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = None
    proc_cwd = repo_dir
    if not is_bevdepth:
        bevdepth_repo = (workdir / "third_party" / "BEVDepth").resolve()
        env = dict(**__import__("os").environ)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(bevdepth_repo) + (":" + prev if prev else "")
        proc_cwd = bevdepth_repo

    proc = subprocess.run(cmd, cwd=proc_cwd, env=env, capture_output=True, text=True, check=False)

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

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def _resolve_path(path_str: str, workdir: Path, repo_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    from_workdir = (workdir / p).resolve()
    if from_workdir.exists():
        return from_workdir
    return (repo_dir / p).resolve()


def _ensure_bevdepth_data_link(config: dict[str, Any], workdir: Path, repo_dir: Path) -> None:
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = (workdir / dataset_root).resolve()

    data_dir = repo_dir / "data"
    target = data_dir / "nuScenes"
    if target.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    target.symlink_to(dataset_root)


def _build_eval_command(config: dict[str, Any], workdir: Path) -> tuple[list[str], Path]:
    model = config["model"]
    repo_dir = Path(model["repo_dir"])
    repo_dir = repo_dir if repo_dir.is_absolute() else (workdir / repo_dir).resolve()

    config_path = _resolve_path(str(model["config_path"]), workdir, repo_dir)
    checkpoint = _resolve_path(str(model["checkpoint"]), workdir, repo_dir)

    model_name = str(model["name"]).lower()
    batch_size = int(model.get("batch_size", 1))
    exp_name = str(config["experiment"].get("name", model_name))
    python_bin = str(model.get("python_executable", "python"))

    if "bevdepth" in model_name:
        _ensure_bevdepth_data_link(config, workdir, repo_dir)
        cmd = [
            python_bin,
            str(config_path),
            "--ckpt_path",
            str(checkpoint),
            "-e",
            "-b",
            str(batch_size),
            "--gpus",
            "1",
            "--default_root_dir",
            str(repo_dir / "outputs" / exp_name),
        ]
        return cmd, repo_dir

    test_script = repo_dir / "tools" / "test.py"
    if not test_script.exists():
        raise FileNotFoundError(
            f"Could not find eval entrypoint at {test_script}. "
            "Set a custom command strategy for this model in runners.py."
        )

    cmd = [
        python_bin,
        str(test_script),
        str(config_path),
        str(checkpoint),
        "--eval",
        "bbox",
    ]
    return cmd, repo_dir


def _read_metrics_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = {}
    for key in ("mean_ap", "nd_score", "mAP", "NDS"):
        if key in payload:
            metrics[key] = float(payload[key])
    return metrics or None


def run_model_eval(config: dict[str, Any], workdir: Path, execute: bool = False) -> dict[str, Any]:
    """Build and optionally execute model evaluation command."""
    metrics_cfg = str(config["evaluation"].get("metrics_file", ""))
    metrics_file = Path(metrics_cfg) if metrics_cfg else None

    try:
        cmd, repo_dir = _build_eval_command(config, workdir)
    except Exception as exc:
        return {
            "model_name": config["model"]["name"],
            "config_name": config["experiment"]["name"],
            "status": "failed",
            "return_code": -1,
            "elapsed_sec": 0.0,
            "cwd": "",
            "command": "",
            "metrics": None,
            "metrics_file": metrics_cfg,
            "log_file": "",
            "error": str(exc),
        }

    cmd_str = " ".join(shlex.quote(part) for part in cmd)

    if metrics_file is not None and not metrics_file.is_absolute():
        metrics_file = (workdir / metrics_file).resolve()

    if not execute:
        return {
            "model_name": config["model"]["name"],
            "config_name": config["experiment"]["name"],
            "status": "planned",
            "cwd": str(repo_dir),
            "command": cmd_str,
            "notes": "Run again with --execute to launch external evaluation.",
        }

    start = time.perf_counter()
    repo_dir.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        cmd,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.perf_counter() - start, 2)

    logs_dir = workdir / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{config['model']['name'].lower()}_last.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(run.stdout)
        if run.stderr:
            f.write("\n\n[stderr]\n")
            f.write(run.stderr)

    metrics = _read_metrics_summary(metrics_file) if metrics_file is not None else None

    hint = ""
    if "No module named 'bevdepth'" in run.stderr:
        hint = (
            "BEVDepth is not installed in this environment. "
            "BEVDepth needs torch==1.9.0/Python<=3.9/CUDA 11.1, which is incompatible with "
            "this project's .venv. Create a dedicated conda env (see configs/models/bevdepth.yaml "
            "'python_executable' and README.md 'BEVDepth environment' section) and point "
            "python_executable at that env's python binary."
        )

    return {
        "model_name": config["model"]["name"],
        "config_name": config["experiment"]["name"],
        "status": "ok" if run.returncode == 0 else "failed",
        "return_code": run.returncode,
        "elapsed_sec": elapsed,
        "cwd": str(repo_dir),
        "command": cmd_str,
        "metrics": metrics,
        "metrics_file": str(metrics_file) if metrics_file is not None else "",
        "log_file": str(log_path),
        "hint": hint,
    }


def run_model_eval_legacy(config: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Compatibility wrapper for older calls."""
    return {
        "model": config["model"]["name"],
        "workdir": str(workdir),
        "status": "not_implemented",
    }

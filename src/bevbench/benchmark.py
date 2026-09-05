from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from bevbench.bev_visualization import default_output_dir, render_bevdepth_detections
from bevbench.runners import run_model_eval


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEV 3D benchmark harness")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to the shared base config",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model-specific config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/last_run.json"),
        help="Where to write benchmark output JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print merged config without running benchmark",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute external model eval command",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After evaluation, render a BEVDepth prediction image into results/visualizations/<model>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    model_cfg = load_yaml(args.model_config)
    cfg = merge_dicts(base_cfg, model_cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    payload = run_model_eval(cfg, workdir=Path.cwd(), execute=args.execute)

    if args.visualize and "bevdepth" in str(cfg["model"]["name"]).lower():
        viz_out = default_output_dir(args.out.parent, cfg["model"]["name"])
        viz_payload = render_bevdepth_detections(cfg, workdir=Path.cwd(), out_dir=viz_out)
        payload["visualization"] = viz_payload
        print(f"Saved BEV visualization metadata for {cfg['model']['name']} to {args.out}")

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved benchmark result to {args.out}")


if __name__ == "__main__":
    main()

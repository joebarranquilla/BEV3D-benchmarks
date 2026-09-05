from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "leaderboard.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("metrics") or {}
        row = {
            "run_file": path.name,
            "model_name": payload.get("model_name", "unknown"),
            "config_name": payload.get("config_name", "unknown"),
            "status": payload.get("status", "unknown"),
            "elapsed_sec": payload.get("elapsed_sec"),
            "mAP": metrics.get("mAP", metrics.get("mean_ap")),
            "NDS": metrics.get("NDS", metrics.get("nd_score")),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["run_file", "model_name", "config_name", "status", "elapsed_sec", "mAP", "NDS"])
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leaderboard CSV from benchmark results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/leaderboard.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = collect_results(args.results_dir)
    if not df.empty:
        df = df.sort_values(by=["NDS", "mAP"], ascending=False, na_position="last")
    df.to_csv(args.out_csv, index=False)
    print(f"Saved leaderboard: {args.out_csv}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

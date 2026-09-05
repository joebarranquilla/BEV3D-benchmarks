from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BEV benchmark leaderboard")
    parser.add_argument("--leaderboard", type=Path, default=Path("results/leaderboard.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def _plot_metric(df: pd.DataFrame, metric: str, out_file: Path) -> None:
    clean = df.dropna(subset=[metric])
    if clean.empty:
        print(f"Skip {metric}: no values available")
        return

    clean = clean.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(clean["model_name"], clean[metric])
    plt.title(f"nuScenes-mini {metric} by model")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()
    print(f"Saved plot: {out_file}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.leaderboard)
    _plot_metric(df, "mAP", args.out_dir / "map.png")
    _plot_metric(df, "NDS", args.out_dir / "nds.png")


if __name__ == "__main__":
    main()

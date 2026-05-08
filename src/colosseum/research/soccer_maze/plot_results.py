"""Visualize HumanoidSoccerMaze benchmark results.

Usage:
    pixi run python -m colosseum.research.soccer_maze.plot_results \
        --input benchmark_results/stage1_results.json

Produces:
    - TSR / SES bar chart
    - Success rate vs training step (cumulative over checkpoints)
    - Per-goal success breakdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_stage1(results_path: Path, output_dir: Path) -> None:
  with open(results_path) as f:
    data = json.load(f)

  output_dir.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  # ── Plot 1: TSR + SES bar chart ────────────────────────────────────────
  labels = ["TSR", "SES"]
  values = [data["TSR"], data["SES"]]
  colors = ["#2196F3", "#4CAF50"]
  axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
  axes[0].set_ylim(0, 1.05)
  axes[0].set_ylabel("Score")
  axes[0].set_title(
    f"Stage {data['stage']}  |  {data.get('map', data.get('difficulty', ''))}"
  )
  for i, v in enumerate(values):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
  axes[0].text(
    0.5, -0.15,
    f"T = {data['T']:,}  |  Λ = {data['Lambda']:,.0f}  |  "
    f"SR = {data['SR_count']}/{data['total_triples']}",
    transform=axes[0].transAxes, ha="center", fontsize=9, color="gray",
  )

  # ── Plot 2: Cumulative success over steps ─────────────────────────────
  if data.get("sr_min_steps"):
    steps = np.array(data["sr_min_steps"], dtype=np.float64)
    steps_sorted = np.sort(steps)
    y = np.arange(1, len(steps_sorted) + 1) / data["total_triples"]
    axes[1].step(steps_sorted / 1e6, y, where="post", color="#2196F3", linewidth=2)
    axes[1].axhline(y=data["TSR"], color="#4CAF50", linestyle="--", alpha=0.7,
                    label=f"TSR = {data['TSR']:.3f}")
    axes[1].set_xlabel("Training steps (millions)")
    axes[1].set_ylabel("Cumulative success fraction")
    axes[1].set_title("Success vs training progress")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

  plt.tight_layout()
  out = output_dir / "benchmark_summary.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out}")


def plot_stage2(results_path: Path, output_dir: Path) -> None:
  with open(results_path) as f:
    data = json.load(f)

  per_map = data.get("per_map", [])
  if not per_map:
    print("No per-map data in Stage 2 results")
    return

  output_dir.mkdir(parents=True, exist_ok=True)

  maps = [m.get("map", "?") for m in per_map]
  tsr_vals = [m["TSR"] for m in per_map]
  ses_vals = [m["SES"] for m in per_map]

  x = np.arange(len(maps))
  w = 0.35

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.bar(x - w/2, tsr_vals, w, label="TSR", color="#2196F3", edgecolor="white")
  ax.bar(x + w/2, ses_vals, w, label="SES", color="#4CAF50", edgecolor="white")
  ax.set_xticks(x)
  ax.set_xticklabels(maps)
  ax.set_ylim(0, 1.05)
  ax.set_ylabel("Score")
  ax.set_title(
    f"Stage 2  |  TSR={data['TSR']:.3f}  |  SES={data['SES']:.3f}"
  )
  ax.legend()
  ax.grid(axis="y", alpha=0.3)

  plt.tight_layout()
  out = output_dir / "stage2_summary.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out}")


def main():
  parser = argparse.ArgumentParser(description="Plot benchmark results")
  parser.add_argument("--input", type=Path, required=True,
                      help="Path to stage*_results.json")
  parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_plots"))
  args = parser.parse_args()

  if "stage2" in args.input.name:
    plot_stage2(args.input, args.output_dir)
  else:
    plot_stage1(args.input, args.output_dir)


if __name__ == "__main__":
  main()

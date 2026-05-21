#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspaces/luminal")
RESULT_DIR = REPO / "scratch/egglog_repros/results_parallelism_120s"
CSV_PATH = RESULT_DIR / "results.csv"
PNG_PATH = REPO / "scratch/egglog_repros/egglog_parallelism_summary.png"
TIMEOUT_S = 120.0

MODELS = [
    "llama",
    "paged_llama",
    "qwen",
    "qwen3_moe",
    "gemma",
    "gemma4_moe",
    "whisper",
]

REVS = [
    "0a8cc35a6c68d0460c20449d5fa19ca3caba2923",
    "2e5657bbb2c1a90fba31002da61381815f891b6f",
    "345fa8d93ff904865c1b69cffbaeeedf6b88cc09",
]

THREADS = ["1", "8", "default-30"]


def split_hash(rev: str) -> str:
    return f"{rev[:20]}\n{rev[20:]}"


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = load_rows()
    by_key = {
        (row["model"], row["egglog_rev"], row["rayon_threads"]): row
        for row in rows
    }

    fig, axes = plt.subplots(1, len(REVS), figsize=(18, 7), constrained_layout=True)
    cmap = plt.get_cmap("viridis_r")
    vmin = math.log10(1.0)
    vmax = math.log10(TIMEOUT_S)

    image = None
    for ax, rev in zip(axes, REVS):
        values = np.zeros((len(MODELS), len(THREADS)))
        labels = [["" for _ in THREADS] for _ in MODELS]

        for i, model in enumerate(MODELS):
            for j, threads in enumerate(THREADS):
                row = by_key[(model, rev, threads)]
                if row["seconds"]:
                    seconds = float(row["seconds"])
                    labels[i][j] = f"{seconds:.1f}s"
                else:
                    seconds = TIMEOUT_S
                    labels[i][j] = f">{int(TIMEOUT_S)}s"
                values[i, j] = math.log10(max(1.0, min(seconds, TIMEOUT_S)))

        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(split_hash(rev), fontsize=10, family="monospace")
        ax.set_xticks(range(len(THREADS)), THREADS)
        ax.set_yticks(range(len(MODELS)), MODELS if ax is axes[0] else [])
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Rayon threads")

        for i in range(len(MODELS)):
            for j in range(len(THREADS)):
                timeout = labels[i][j].startswith(">")
                ax.text(
                    j,
                    i,
                    labels[i][j],
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > 1.4 else "black",
                    fontsize=8,
                    fontweight="bold" if timeout else "normal",
                )

    fig.suptitle("Egglog parse_and_run_program runtime by Rayon parallelism (120s cap)", fontsize=14)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.78)
        cbar.set_label("log10(seconds), capped at 120s")
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=180)
    print(PNG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

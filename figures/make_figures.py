"""Regenerate every figure from stored results. No simulation, no fitting.

PROJECT.md section 26: figures must regenerate from a clean checkout plus the
committed JSON in results/. This script reads only those files, so CI can
rebuild the figures without NEURON, without hnn-core, and in seconds.

    python figures/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures"


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def fig_data_budget() -> bool:
    """Recovery against recording duration, per estimator."""
    runs = load("calibration_n20_t*.json")
    if not runs:
        return False

    by_estimator: dict[str, dict[float, list[float]]] = {}
    for run in runs:
        duration = float(run["config"]["duration"])
        for name, res in run["results"].items():
            by_estimator.setdefault(name, {}).setdefault(duration, []).append(
                res["weight_spearman"]
            )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, series in sorted(by_estimator.items()):
        xs = sorted(series)
        means = [np.mean(series[x]) for x in xs]
        sds = [np.std(series[x]) for x in xs]
        ax.errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("recording duration (s)")
    ax.set_ylabel(r"weight recovery  $\rho$")
    ax.set_title("Data budget: synthetic ground truth")
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    # The synthetic model is exactly well specified, so this curve should
    # approach 1.0 rather than plateau (section 5). The flattening to look for
    # belongs to the HNN curve.
    ax.annotate("no plateau expected:\nmodel is exactly specified",
                xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(OUT / "data_budget.png", dpi=160)
    plt.close(fig)
    return True


def fig_hnn_per_class() -> bool:
    """Per-connection-class recovery on HNN, never pooled."""
    runs = load("hnn_first_*.json")
    runs = [r for r in runs if r.get("per_class")]
    if not runs:
        return False
    run = max(runs, key=lambda r: float(r["config"]["tstop"]))

    names = list(run["per_class"])
    rhos = [run["per_class"][n]["weight_spearman"] for n in names]
    order = np.argsort(rhos)
    names = [names[i] for i in order]
    rhos = [rhos[i] for i in order]
    # Colour by source sign: the E/I split is the axis that matters.
    colours = ["#c2543d" if n.startswith(("L2_basket", "L5_basket")) else "#3d6ec2"
               for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.6))
    ax.barh(range(len(names)), rhos, color=colours)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("|", "  ") for n in names], fontsize=7)
    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel(r"weight recovery  $\rho$  (within connection class)")
    ax.set_title(f"HNN recovery by connection class "
                 f"({float(run['config']['tstop'])/1000:g}s, "
                 f"{run['config']['mesh'][0]}x{run['config']['mesh'][1]})")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#3d6ec2", "#c2543d")]
    ax.legend(handles, ["excitatory source", "inhibitory source"],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "hnn_per_class.png", dpi=160)
    plt.close(fig)
    return True


def fig_grid() -> bool:
    """Section 31's degradation figure: drive observability x observed population."""
    runs = load("grid_*.json")
    if not runs:
        return False
    records = [r for run in runs for r in run["records"]]

    drives = ["true", "proxy", "hidden"]
    observations = sorted({r["observation"] for r in records})

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for obs in observations:
        means, sds = [], []
        for drive in drives:
            vals = [r["weight_spearman"] for r in records
                    if r["observation"] == obs and r["drive"] == drive]
            means.append(np.mean(vals) if vals else np.nan)
            sds.append(np.std(vals) if vals else 0.0)
        ax.errorbar(range(len(drives)), means, yerr=sds, marker="o",
                    capsize=3, label=obs.replace("_", " "))

    ax.set_xticks(range(len(drives)))
    ax.set_xticklabels(["true drive", "observable proxy", "drive hidden"])
    ax.set_ylabel(r"weight recovery  $\rho$")
    ax.set_title("Recovery degradation: drive observability x observed population")
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    built = {
        "data_budget.png": fig_data_budget(),
        "hnn_per_class.png": fig_hnn_per_class(),
        "degradation.png": fig_grid(),
    }
    for name, ok in built.items():
        print(f"  {'wrote' if ok else 'SKIPPED (no results yet)'}: {name}")
    if not any(built.values()):
        print("\nNo results found in results/. Run the experiments first.")
        sys.exit(1)


if __name__ == "__main__":
    main()

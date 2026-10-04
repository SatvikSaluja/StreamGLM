"""Render recorded benchmarks and model-selection results; no fitted curves."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/v2"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    benchmarks = [json.loads(path.read_text()) for path in sorted(args.artifacts.glob("benchmark_*.json"))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for impl, color in [("streamed", "#267a68"), ("materialized", "#b35c37")]:
        rows = [r for r in benchmarks if r.get("implementation", "streamed") == impl]
        if rows:
            axes[0].scatter([r["hypothetical_design_gib"] for r in rows], [r["peak_rss_gib"] for r in rows], label=impl, c=color)
    axes[0].set(xlabel="Materialized design size (GiB)", ylabel="Measured whole-process peak RSS (GiB)", title="Fixed-parameter objective/gradient passes")
    if benchmarks:
        axes[0].legend()
    report_path = args.artifacts/"real_mouse32"/"report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        rows = report["selection"]["candidates"]
        axes[1].scatter([r["rank"] for r in rows], [r["validation"]["gain_nats_per_bin"] for r in rows], c="#267a68")
        axes[1].set(xlabel="Coupling rank (0 = self-history only)", ylabel="Validation gain over intercept (nats/bin)", title="Mouse32: validation-selected models")
    else:
        axes[1].text(.5, .5, "Real-data report not available", ha="center")
    fig.savefig(args.artifacts/"overview.png", dpi=180)
    svg = args.artifacts/"overview.svg"
    fig.savefig(svg)
    # Matplotlib emits trailing spaces inside multiline paths; normalize the
    # generated text so the checked-in scientific figure passes diff checks.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


if __name__ == "__main__":
    main()

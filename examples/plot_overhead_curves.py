"""Generate a publication-quality plot of cross-silicon Freivalds overhead.

Reads measured data from the three-platform benchmark runs (M5 MPS, AMD MI300X,
NVIDIA H100) and produces a log-log plot of verification overhead vs. matrix
size, with a 1% threshold reference line.

Output: overhead_curves.pdf (vector, suitable for LaTeX inclusion).

Usage:
    python examples/plot_overhead_curves.py --output overhead_curves.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


# Measured data from 2026-04-22 benchmark runs, FP32, k=10 Freivalds iterations.
# Values are overhead as percentage of matmul time.
# See traces/2026-04-22_*.jsonl for raw records.
DATA = {
    "Apple M5 (MPS)": {
        1024: 86.0,
        2048: 25.0,
        4096: 6.0,
        8192: 2.2,
        16384: 0.97,
    },
    "AMD MI300X (ROCm 7.0)": {
        1024: 475.0,
        2048: 132.0,
        4096: 29.0,
        8192: 7.5,
        16384: 2.0,
        32768: 0.75,
    },
    "NVIDIA H100 (CUDA 12.4)": {
        1024: 310.0,
        2048: 80.0,
        4096: 16.0,
        8192: 4.4,
        16384: 1.5,
        32768: 0.55,
    },
}


STYLE = {
    "Apple M5 (MPS)": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "AMD MI300X (ROCm 7.0)": {"color": "#d62728", "marker": "s", "linestyle": "-"},
    "NVIDIA H100 (CUDA 12.4)": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="overhead_curves.pdf")
    parser.add_argument("--width", type=float, default=6.0)
    parser.add_argument("--height", type=float, default=4.0)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    # Plot each platform's curve
    for label, curve in DATA.items():
        ns = sorted(curve.keys())
        overheads = [curve[n] for n in ns]
        style = STYLE[label]
        ax.plot(
            ns,
            overheads,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=7,
            linewidth=1.8,
        )

    # 1% threshold reference line
    ax.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        zorder=0,
    )
    ax.text(
        1100, 1.15, "1% overhead",
        fontsize=9, color="gray", style="italic",
    )

    # Axes: log-log
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # X ticks at each measured size
    all_ns = sorted({n for curve in DATA.values() for n in curve})
    ax.set_xticks(all_ns)
    ax.set_xticklabels([f"{n:,}" for n in all_ns], rotation=0)
    ax.set_xlim(900, 40000)

    # Y ticks and limits
    ax.set_ylim(0.3, 1000)
    ax.set_yticks([0.5, 1, 2, 5, 10, 25, 100, 500])
    ax.set_yticklabels(["0.5%", "1%", "2%", "5%", "10%", "25%", "100%", "500%"])

    # Labels
    ax.set_xlabel("Matrix dimension $n$ (square matmul $n\\!\\times\\!n\\!\\times\\!n$)", fontsize=11)
    ax.set_ylabel("Freivalds overhead (\\% of matmul time)", fontsize=11)
    ax.set_title("Cross-silicon Freivalds verification overhead, FP32, $k{=}10$", fontsize=11)

    # Grid and legend
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(args.output, format="pdf", bbox_inches="tight")
    print(f"Saved figure to: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()

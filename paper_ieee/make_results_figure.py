"""Build the cross-scale Dice-delta figure from completed machine artifacts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["BUSI", "ClinicDB", "ColonDB", "ISIC18", "DSB18", "EM"]
NETWORKS = ["MK_UNet_T", "MK_UNet_S", "MK_UNet"]
NETWORK_LABELS = ["Tiny", "Small", "Standard"]
ARMS = ["pruned", "adaptive", "sparse"]
ARM_LABELS = [r"Static $\{3,5\}$", "Adaptive soft", "Sparse top-1"]


def load_completed_runs():
    rows = {}
    for result_path in ROOT.glob("dist/machine*/path A/runs/*/result.json"):
        config_path = result_path.with_name("config.json")
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        kernels = "".join(str(k) for k in config["kernel_sizes"])
        if config["routing_mode"] == "fixed":
            arm = "fixed" if kernels == "135" else "pruned"
        elif config["routing_mode"] == "adaptive_soft":
            arm = "adaptive"
        elif config["routing_mode"] == "sparse_top1":
            arm = "sparse"
        else:
            continue
        key = (config["network"], config["dataset"], arm)
        if key in rows:
            raise RuntimeError(f"duplicate completed run for {key}")
        rows[key] = 100.0 * result["test_dice_at_best_val"]
    return rows


def main():
    rows = load_completed_runs()
    expected = len(NETWORKS) * len(DATASETS) * 4
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} completed runs, found {len(rows)}")

    fig, axes = plt.subplots(
        1, 3, figsize=(10.8, 3.0), constrained_layout=True,
        gridspec_kw={"wspace": 0.10}
    )
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    for ax, arm, title in zip(axes, ARMS, ARM_LABELS):
        values = np.array([
            [rows[(network, dataset, arm)] - rows[(network, dataset, "fixed")]
             for dataset in DATASETS]
            for network in NETWORKS
        ])
        image = ax.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xticks(range(len(DATASETS)), DATASETS, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(NETWORKS)), NETWORK_LABELS, fontsize=9)
        ax.set_xticks(np.arange(-.5, len(DATASETS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(NETWORKS), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                color = "white" if abs(value) >= 2.25 else "#111111"
                ax.text(col, row, f"{value:+.2f}", ha="center", va="center",
                        fontsize=8.1, color=color)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.012)
    colorbar.set_label("Dice change from fixed baseline (points)", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5, length=2)
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "cross_scale_dice_delta.pdf", bbox_inches="tight")
    fig.savefig(out / "cross_scale_dice_delta.png", dpi=220, bbox_inches="tight")
    print(f"wrote {out / 'cross_scale_dice_delta.pdf'}")


if __name__ == "__main__":
    main()

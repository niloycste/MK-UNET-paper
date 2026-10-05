"""Generate simplified vector diagrams for the MK-UNet conditional-routing study."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures"
COLORS = {
    "blue": "#dceeff",
    "blue_edge": "#2878b5",
    "green": "#e5f4df",
    "green_edge": "#4f8f45",
    "orange": "#fff0cc",
    "orange_edge": "#d98900",
    "purple": "#eadcff",
    "purple_edge": "#7753a6",
    "red": "#ffe1de",
    "red_edge": "#c74b46",
    "gray": "#f2f3f4",
    "gray_edge": "#667078",
}


def box(ax, xy, width, height, text, fc="gray", ec="gray_edge", size=9,
        weight="normal"):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2, facecolor=COLORS[fc], edgecolor=COLORS[ec]
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=size, fontweight=weight,
            linespacing=1.05)
    return patch


def arrow(ax, start, end, color="#333333", style="-", width=1.2,
          mutation=10):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=width, linestyle=style, color=color,
        shrinkA=1.5, shrinkB=1.5
    )
    ax.add_patch(patch)


def panel_title(ax, x, text):
    ax.text(x, 0.955, text, ha="center", va="center",
            fontsize=11, fontweight="bold")


def architecture_figure():
    fig, ax = plt.subplots(figsize=(12.4, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_title(ax, 0.16, "(a) MK-UNet topology retained")
    panel_title(ax, 0.52, "(b) Conditional MKDC block")
    panel_title(ax, 0.855, "(c) Inference execution")

    # (a) compact U-shaped topology
    enc_x, dec_x = 0.035, 0.225
    ys = [0.78, 0.62, 0.46, 0.30]
    enc = []
    dec = []
    for i, y in enumerate(ys):
        enc.append(box(ax, (enc_x, y), 0.075, 0.085, f"Enc {i+1}\nR", "blue", "blue_edge", 8.3))
        dec.append(box(ax, (dec_x, y), 0.075, 0.085, f"Dec {i+1}\nR", "green", "green_edge", 8.3))
        if i < len(ys) - 1:
            arrow(ax, (enc_x + 0.0375, y), (enc_x + 0.0375, ys[i+1] + 0.085))
            arrow(ax, (dec_x + 0.0375, ys[i+1] + 0.085), (dec_x + 0.0375, y))
        arrow(ax, (enc_x + 0.075, y + 0.043), (dec_x, y + 0.043),
              color="#777777", style="--", width=0.9)
    bottleneck = box(ax, (0.13, 0.14), 0.075, 0.085, "Enc 5\nR", "purple", "purple_edge", 8.3)
    arrow(ax, (enc_x + 0.0375, ys[-1]), (0.1675, 0.225))
    arrow(ax, (0.205, 0.182), (dec_x + 0.0375, ys[-1]))
    ax.text(0.167, 0.065, "R replaces only MKDC aggregation",
            ha="center", fontsize=8.2, color="#444444")

    # (b) branches and router
    box(ax, (0.358, 0.49), 0.058, 0.10, "feature\n$x$", "gray", "gray_edge", 9.2)
    branch_y = [0.72, 0.50, 0.28]
    kernels = ["DW $1\\times1$", "DW $3\\times3$", "DW $5\\times5$"]
    for y, name in zip(branch_y, kernels):
        box(ax, (0.468, y), 0.105, 0.105, name, "green", "green_edge", 9.4)
        arrow(ax, (0.416, 0.54), (0.468, y + 0.052))
        arrow(ax, (0.573, y + 0.052), (0.632, 0.54))
    box(ax, (0.632, 0.48), 0.095, 0.12, "weighted sum\nor select", "orange", "orange_edge", 8.4)
    box(ax, (0.382, 0.10), 0.125, 0.11, "GAP $\\rightarrow$ MLP", "purple", "purple_edge", 9.2)
    box(ax, (0.555, 0.10), 0.128, 0.11, "softmax\n$a_1,a_3,a_5$", "purple", "purple_edge", 8.4)
    arrow(ax, (0.387, 0.49), (0.445, 0.21), color=COLORS["purple_edge"])
    arrow(ax, (0.507, 0.155), (0.555, 0.155), color=COLORS["purple_edge"])
    arrow(ax, (0.683, 0.21), (0.683, 0.48), color=COLORS["purple_edge"])

    # (c) explicit execution modes
    modes = [
        (0.70, "Fixed", "1, 3, 5", "3/3 branches", "gray", "gray_edge"),
        (0.47, "Adaptive soft", "$a_1,a_3,a_5$", "3/3 branches", "blue", "blue_edge"),
        (0.24, "Sparse top-1", "$k^*=\\arg\\max a_k$", "1/3 branches", "orange", "orange_edge"),
    ]
    for y, name, rule, count, fc, ec in modes:
        box(ax, (0.758, y), 0.128, 0.115, name, fc, ec, 9.2, "bold")
        box(ax, (0.896, y), 0.088, 0.115, rule, fc, ec, 8.1)
        ax.text(0.872, y - 0.036, count, ha="center", va="center",
                fontsize=8.5, color=COLORS[ec], fontweight="bold")
    ax.text(0.872, 0.075, "Top-1 dispatch invokes only\none selected branch",
            ha="center", va="center", fontsize=8.5, color="#444444")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "conditional_routing_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "conditional_routing_architecture.png", dpi=240,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def diagnostic_figure():
    fig, ax = plt.subplots(figsize=(12.1, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.020, 0.165, "trained adaptive\ncheckpoint", "blue", "blue_edge"),
        (0.225, 0.165, "3 ClinicDB images\n10 routed blocks", "gray", "gray_edge"),
        (0.430, 0.185, "branch norm shares\nand cosine similarity", "purple", "purple_edge"),
        (0.650, 0.185, "diagnostic hypothesis:\nremove $1\\times1$ branch", "orange", "orange_edge"),
        (0.875, 0.105, "train static\n$\\{3,5\\}$", "green", "green_edge"),
    ]
    centers = []
    for x, width, text_value, fc, ec in steps:
        box(ax, (x, 0.38), width, 0.28, text_value, fc, ec, 9.6,
            "bold" if "hypothesis" in text_value else "normal")
        centers.append((x, width))
    for (x1, w1), (x2, _) in zip(centers[:-1], centers[1:]):
        arrow(ax, (x1 + w1, 0.52), (x2, 0.52), width=1.4, mutation=12)
    ax.text(0.50, 0.16,
            "Purpose: a limited routing diagnostic motivates one fixed-branch ablation; it is not an optimal-pruning rule.",
            ha="center", va="center", fontsize=9.2, color="#444444")

    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "diagnostic_static_ablation.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "diagnostic_static_ablation.png", dpi=240,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    architecture_figure()
    diagnostic_figure()


if __name__ == "__main__":
    main()

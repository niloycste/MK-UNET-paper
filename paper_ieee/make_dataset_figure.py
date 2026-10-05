"""Create a deterministic dataset-overview figure from prepared test splits.

For each dataset, the sample whose foreground fraction is closest to the
split median is selected. This avoids cherry-picking unusually easy or large
targets while keeping the figure reproducible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"

DATASETS = [
    ("BUSI", ROOT / "data/busi/target/BUSI/test"),
    ("ClinicDB", ROOT / "data/polyp/target/ClinicDB/test"),
    ("ColonDB", ROOT / "data/polyp/target/ColonDB/test"),
    ("ISIC18", ROOT / "data/isic/target/ISIC18/test"),
    ("DSB18", ROOT / "data/cell/target/DSB18/test"),
    ("EM", ROOT / "data/cell/target/EM/test"),
]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def representative_pair(split: Path):
    image_dir, mask_dir = split / "images", split / "masks"
    candidates = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            mask = read_mask(mask_path)
            candidates.append((float(mask.mean()), image_path, mask_path))
    if not candidates:
        raise RuntimeError(f"No matched image/mask pairs in {split}")
    fractions = np.asarray([row[0] for row in candidates])
    median = float(np.median(fractions))
    return min(candidates, key=lambda row: abs(row[0] - median))[1:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(12.0, 3.75))

    for column, (name, split) in enumerate(DATASETS):
        image_path, mask_path = representative_pair(split)
        image = read_rgb(image_path)
        mask = read_mask(mask_path)

        axes[0, column].imshow(image)
        axes[0, column].set_title(name, fontsize=10, fontweight="bold", pad=4)
        axes[1, column].imshow(mask, cmap="gray", vmin=0, vmax=1)

        for row in range(2):
            axes[row, column].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Ground truth", fontsize=10, fontweight="bold")
    for row in range(2):
        axes[row, 0].yaxis.set_label_coords(-0.12, 0.5)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.025,
                        wspace=0.035, hspace=0.08)
    fig.savefig(OUT / "dataset_examples.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "dataset_examples.png", dpi=240,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

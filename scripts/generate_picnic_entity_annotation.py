#!/usr/bin/env python3
"""
Generate annotated picnic image showing entity mention differences between HC and svPPA.
Creates a two-panel figure similar to eye-tracking comparison (HC vs lvPPA style).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

# Entity mention rates from heatmap (HC, svPPA) - proportion who mentioned
ENTITY_RATES = {
    "family": (0.92, 0.71),
    "man": (0.80, 0.57),
    "woman": (0.52, 0.31),
    "boy": (0.64, 0.33),
    "girl": (0.40, 0.24),
    "dog": (0.68, 0.52),
    "kite": (0.80, 0.33),
    "blanket": (0.40, 0.05),
    "picnic basket": (0.44, 0.10),
    "book": (0.36, 0.21),
    "tree": (0.80, 0.45),
    "lake": (0.80, 0.48),
    "boat": (0.24, 0.24),
    "house": (0.72, 0.57),
}

# Approximate bounding boxes: (x_frac, y_frac, w_frac, h_frac) - fraction of image
# Layout based on WAB Picnic: man left on blanket, woman pouring, boy with kite, girl in sand, dog, house left, lake/boat right
BOXES = {
    "family": (0.02, 0.45, 0.55, 0.52),   # covers picnic group
    "man": (0.05, 0.52, 0.18, 0.35),
    "woman": (0.12, 0.50, 0.18, 0.38),
    "boy": (0.38, 0.50, 0.18, 0.42),
    "girl": (0.52, 0.58, 0.15, 0.35),
    "dog": (0.32, 0.62, 0.18, 0.28),
    "kite": (0.48, 0.02, 0.22, 0.18),
    "blanket": (0.02, 0.70, 0.40, 0.26),
    "picnic basket": (0.15, 0.68, 0.12, 0.12),
    "book": (0.06, 0.58, 0.08, 0.12),
    "tree": (0.02, 0.12, 0.20, 0.40),
    "lake": (0.22, 0.32, 0.70, 0.42),
    "boat": (0.68, 0.20, 0.24, 0.30),
    "house": (0.00, 0.00, 0.24, 0.38),
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate annotated picnic (HC vs svPPA)")
    ap.add_argument("--input", type=Path, default=None,
                    help="Path to picnic image (e.g. picnic.png). If not found, creates placeholder.")
    ap.add_argument("--output", type=Path, default=Path("outputs/picnic_entity_annotation.png"))
    args = ap.parse_args()

    # Find input image
    candidates = [
        args.input,
        Path("picnic.png"),
        Path("data/picnic.png"),
    ]
    img_path = None
    for c in candidates:
        if c and c.exists():
            img_path = c
            break

    if img_path is None:
        # Create gray placeholder if no image
        arr = np.ones((400, 600, 3)) * 0.9
        arr[50:350, 80:520] = 0.85
        img = Image.fromarray((arr * 255).astype(np.uint8))
        print("No picnic image found; using placeholder. Provide --input for actual image.")
    else:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

    h, w = arr.shape[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.02, right=0.98, wspace=0.08)

    for ax, panel_name, use_hc in [(ax1, "A) HC", True), (ax2, "B) svPPA", False)]:
        ax.imshow(arr)
        ax.set_title(panel_name, fontsize=13, fontweight="bold")
        ax.axis("off")

        for entity, (hc_rate, sv_rate) in ENTITY_RATES.items():
            if entity not in BOXES:
                continue
            xf, yf, wf, hf = BOXES[entity]
            x0 = int(xf * w)
            y0 = int(yf * h)
            x1 = int((xf + wf) * w)
            y1 = int((yf + hf) * h)
            rate = hc_rate if use_hc else sv_rate
            # Color: green = high, orange = medium, red = low
            if rate >= 0.5:
                color = (0.2, 0.65, 0.35)
            elif rate >= 0.25:
                color = (0.95, 0.55, 0.2)
            else:
                color = (0.8, 0.25, 0.25)
            rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor=color, facecolor=(*color[:3], 0.08))
            ax.add_patch(rect)
            # Label with subtle bbox for readability
            cx, cy = (x0 + x1) / 2, y0 + (y1 - y0) / 2
            ax.text(cx, cy, entity, ha="center", va="center", fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"))

    # Legend below panels
    legend_elements = [
        Line2D([0], [0], color=(0.2, 0.65, 0.35), linewidth=5, label="≥50%"),
        Line2D([0], [0], color=(0.95, 0.55, 0.2), linewidth=5, label="25–50%"),
        Line2D([0], [0], color=(0.8, 0.25, 0.25), linewidth=5, label="<25%"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11, frameon=True, fancybox=True)

    fig.suptitle("Proportion of participants naming each entity", fontsize=13, fontweight="bold", y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

"""Export Week 9–10 PR and matched error figures as PNG."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from report_fullxy_pr import curve


def font(size):
    return ImageFont.load_default(size=size)


def main():
    output = Path("samples/week9_10_figures")
    output.mkdir(exist_ok=True)
    for seed in (42, 43):
        summary = json.loads(Path(f"samples/hann_seed{seed}_pr_01/summary.json").read_text())
        canvas = Image.new("RGB", (960, 580), "#18232b")
        draw = ImageDraw.Draw(canvas)
        draw.text((70, 16), f"Hann validation precision–recall | Seed {seed}", fill="white", font=font(23))
        for i in range(6):
            x, y = 80 + 160 * i, 500 - 84 * i
            draw.line((x, 80, x, 500), fill="#425465")
            draw.line((80, y, 880, y), fill="#425465")
            draw.text((x, 510), f"{i / 5:.1f}", fill="white", font=font(16), anchor="mt")
            draw.text((65, y), f"{i / 5:.1f}", fill="white", font=font(16), anchor="rm")
        draw.text((480, 545), "Recall", fill="white", font=font(20), anchor="mt")
        draw.text((10, 55), "Precision", fill="white", font=font(16))
        for index, (variant, color) in enumerate((("raw", "#afbdca"), ("nlm", "#69debb"))):
            path = Path(f"samples/blend_{variant}_hann_seed{seed}_val_pr_01/pr_scores.npz")
            with np.load(path) as data:
                _, precision, recall = curve(data["probability"], data["truth"])
            keep = np.unique(np.searchsorted(recall, np.linspace(0, 1, 3000)))
            draw.line([(80 + 800 * recall[i], 500 - 420 * precision[i]) for i in keep], fill=color, width=3)
            row = summary[variant]
            x = 80 + 800 * row["recall_at_candidate"]
            y = 500 - 420 * row["precision_at_60_recall"]
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline="white")
            draw.text((610 + 137 * index, 53), f"{variant.upper()} AP {row['ap']:.3f}", fill=color, font=font(16))
        canvas.save(output / f"pr_seed{seed}.png")

    for sample, z in ((11, 64), (12, 64)):
        stem = f"sample_{sample:02d}_z{z:03d}"
        canvas = Image.new("RGB", (1536, 1120), "#18232b")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"Seed 42 | sample_{sample:02d} | Z={z} | Hann stitching", fill="white", font=font(22))
        for row_index, cards in enumerate((
            (("raw", 1, "Reference mask"), ("raw", 2, "Raw prediction"), ("nlm", 2, "NLM prediction")),
            (("raw", 0, "Raw EM"), ("raw", 3, "Raw errors"), ("nlm", 3, "NLM errors")),
        )):
            for col, (variant, panel, label) in enumerate(cards):
                image = Path(f"samples/blend_{variant}_hann_seed42_test_r60_01/assets/{stem}_whole_section.png")
                x, y = col * 512, 34 + row_index * 542
                draw.text((x + 8, y + 4), label, fill="white", font=font(18))
                with Image.open(image) as source:
                    canvas.paste(source.crop((panel*512, 46, (panel+1)*512, 558)), (x, y + 30))
        canvas.save(output / f"{stem}_comparison.png", optimize=True)


if __name__ == "__main__":
    main()

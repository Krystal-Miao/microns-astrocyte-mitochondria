"""Build a local side-by-side viewer from the three full-XY evaluation runs."""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=[500, 3000], default=500)
    parser.add_argument("--hann", action="store_true", help="Use the 3,000-step Hann runs")
    args = parser.parse_args()
    if args.hann and args.steps != 3000:
        parser.error("--hann requires --steps 3000")
    variants = ("raw", "clahe", "nlm") if args.steps == 500 else ("raw", "nlm")
    suffix = "" if args.steps == 500 else "3000"
    tail = "_pr_01" if args.steps == 3000 and args.split == "val" else "_01"
    roots = {v: Path(f"samples/unet_mip2_{v}{suffix}_fullxy_{args.split}{tail}") for v in variants}
    if args.hann:
        roots = {v: Path(f"samples/blend_{v}_hann_{args.split}_01") for v in variants}
    results = {v: json.loads((root / "result.json").read_text()) for v, root in roots.items()}
    args.out.mkdir(parents=True, exist_ok=False)
    panels = []
    scores = ["average_precision", "dice", "precision", "recall"]

    def row(label, metrics):
        return f"<tr><th>{label}</th>" + "".join(f"<td>{metrics[k]:.3f}</td>" for k in scores) + "</tr>"

    header = "<tr><th>Input</th><th>AP</th><th>Dice</th><th>Precision</th><th>Recall</th></tr>"
    sections = {(r["sample"], r["z"]) for r in results["raw"]["per_section"]}
    by_section = {v: {(r["sample"], r["z"]): r["metrics"] for r in d["per_section"]}
                  for v, d in results.items()}
    if any(set(d) != sections for d in by_section.values()):
        raise ValueError("Evaluation runs cover different sections")
    for sample, z in sorted(sections):
        stem = f"{sample}_z{z:03d}"
        panels.append(f"<h2>{sample}, Z={z}</h2><table>{header}" +
                      "".join(row(v.upper(), by_section[v][sample, z]) for v in variants) + "</table>")
        for group in ("prediction", "errors"):
            panels.append('<div class="grid">')
            cards = [("raw", 0, "Raw EM")] if group == "prediction" else [("raw", 1, "Reference")]
            cards += [(v, 2 if group == "prediction" else 3, f"{v.upper()} {group}") for v in variants]
            for v, index, label in cards:
                with Image.open(roots[v] / "assets" / f"{stem}_whole_section.png") as source:
                    width = source.width // 5
                    filename = f"{stem}_{v}_{index}.png"
                    source.crop((index * width, 46, (index + 1) * width, source.height)).save(args.out / filename)
                panels.append(f'<figure><figcaption>{label}</figcaption><a href="{filename}">'
                              f'<img loading="lazy" src="{filename}" alt="{label}"></a></figure>')
            panels.append("</div>")
    split = "Validation" if args.split == "val" else "Previously evaluated test cells"
    summary = "".join(row(v.upper(), results[v]["aggregate"]) for v in variants)
    sample_tables = "".join(
        f"<h2>{sample} summary</h2><table>{header}"
        + "".join(row(v.upper(), results[v]["per_sample"][sample]) for v in variants)
        + "</table>"
        for sample in results["raw"]["per_sample"]
    )
    checkpoint_steps = ", ".join(f"{v.upper()} step {results[v]['checkpoint_step']}" for v in variants)
    inference = "50% XY overlap with Hann blending" if args.hann else "single full-XY forward pass"
    (args.out / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{split} · MIP2 preprocessing</title><style>
body{{max-width:1300px;margin:2rem auto;padding:0 1rem;background:#18232b;color:#eee;font:16px system-ui}}
.grid{{display:grid;grid-template-columns:repeat({len(variants) + 1},minmax(0,1fr));gap:10px;margin:1rem 0}}
figure{{margin:0}}img{{width:100%;display:block}}td,th{{padding:.3rem .8rem;text-align:right}}
figcaption{{padding:.4rem 0}}h2{{margin-top:2rem}}
</style><h1>{split} · MIP2 preprocessing</h1>
<p>{args.steps:,} training steps, seed 42. Selected checkpoints: {checkpoint_steps}. Inference: {inference}. Same full XY sections and scored label mask. Dice, precision and recall use each checkpoint's frozen validation-patch threshold.</p>
<p>Prediction: cyan = foreground. Errors: green = true positive, red = false positive on scored negatives, blue = false negative. Click a panel to view it at full size.</p>
<table>{header}{summary}</table>{sample_tables}{''.join(panels)}</html>""", encoding="utf-8")


if __name__ == "__main__":
    main()

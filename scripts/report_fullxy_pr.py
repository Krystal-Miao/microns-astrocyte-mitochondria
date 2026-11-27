"""Plot pooled full-XY validation precision–recall curves."""

import argparse
import json
from pathlib import Path

import numpy as np


def curve(probability, truth):
    order = np.argsort(-probability, kind="stable")
    probability, truth = probability[order], truth[order]
    ends = np.r_[np.flatnonzero(probability[:-1] != probability[1:]), len(truth) - 1]
    tp = np.cumsum(truth)[ends]
    return probability[ends], tp / (ends + 1), tp / truth.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hann-seed", type=int, choices=[42, 43])
    parser.add_argument("--raw3000", action="store_true", help="Include the 3,000-step Raw run")
    parser.add_argument("--nlm3000", action="store_true", help="Include the 3,000-step NLM run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths, markers, rows, legend = [], [], [], []
    summary = {}
    reference_truth = None
    variants = [("raw", "#a9b8c9"), ("clahe", "#f5ae5c"), ("nlm", "#69debb")]
    if args.raw3000:
        variants.append(("raw3000", "#83b8ff"))
    if args.nlm3000:
        variants.append(("nlm3000", "#ef94e8"))
    if args.hann_seed is not None:
        variants = [("raw", "#a9b8c9"), ("nlm", "#69debb")]
    target_recall = 0.70 if args.hann_seed is None else 0.60
    recall_key = round(100 * target_recall)
    for name, color in variants:
        root = (Path(f"samples/unet_mip2_{name}_fullxy_val_pr_01") if args.hann_seed is None else
                Path(f"samples/blend_{name}_hann_seed{args.hann_seed}_val_pr_01"))
        with np.load(root / "pr_scores.npz") as scores:
            probability, truth, locked = scores["probability"], scores["truth"], float(scores["threshold"])
        if reference_truth is None:
            reference_truth = truth
        elif not np.array_equal(reference_truth, truth):
            raise ValueError("Scored labels differ across variants")
        thresholds, precision, recall = curve(probability, truth)
        ap = float(np.dot(np.diff(np.r_[0, recall]), precision))
        predicted = probability >= locked
        tp = np.count_nonzero(predicted & truth)
        lp, lr = tp / max(1, predicted.sum()), tp / truth.sum()
        eligible = np.flatnonzero(recall >= target_recall)
        i = eligible[np.argmax(precision[eligible])]
        summary[name] = dict(ap=ap, prevalence=float(truth.mean()), locked_threshold=locked,
                             locked_precision=lp, locked_recall=lr,
                             recall_at_candidate=float(recall[i]),
                             **{f"threshold_at_{recall_key}_recall": float(thresholds[i]),
                                f"precision_at_{recall_key}_recall": float(precision[i])})
        keep = np.unique(np.r_[0, np.searchsorted(recall, np.linspace(0, 1, 2000)), len(recall) - 1])
        points = " ".join(f"{70 + 680*r:.2f},{440 - 400*p:.2f}" for r, p in zip(recall[keep], precision[keep]))
        paths.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        legend_x = 72 + 120 * len(legend)
        legend.append(f'<text x="{legend_x}" y="24" fill="{color}">{name.upper()} · {ap:.3f}</text>')
        markers.append(f'<circle cx="{70+680*lr:.2f}" cy="{440-400*lp:.2f}" r="5" fill="{color}" stroke="white"><title>{name.upper()} locked threshold {locked:.3f}</title></circle>')
        markers.append(f'<rect x="{65+680*recall[i]:.2f}" y="{435-400*precision[i]:.2f}" width="10" height="10" fill="{color}" stroke="white"><title>{name.upper()} candidate threshold {thresholds[i]:.3f}</title></rect>')
        rows.append(f'<tr><th style="color:{color}">{name.upper()}</th><td>{ap:.3f}</td><td>{locked:.3f}</td><td>{lp:.1%}</td><td>{lr:.1%}</td><td>{thresholds[i]:.3f}</td><td>{precision[i]:.1%}</td><td>{recall[i]:.1%}</td></tr>')
    prevalence_y = 440 - 400 * reference_truth.mean()
    ticks = []
    for i in range(6):
        x, y = 70 + 136*i, 440 - 80*i
        ticks.append(f'<path d="M{x} 40 V440 M70 {y} H750" stroke="#334656"/><text x="{x}" y="464" text-anchor="middle">{i/5:.1f}</text><text x="58" y="{y+4}" text-anchor="end">{i/5:.1f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 510" font-family="system-ui" fill="#eee">
<title>Validation precision–recall curves</title><rect width="780" height="510" fill="#18232b"/>
{''.join(legend)}{''.join(ticks)}<path d="M70 {prevalence_y} H750" stroke="#8f9ca5" stroke-dasharray="6 5"/>
{''.join(paths)}{''.join(markers)}
<text x="410" y="492" text-anchor="middle">Recall</text><text x="18" y="275" transform="rotate(-90 18 275)">Precision</text></svg>"""
    (args.out / "curve.svg").write_text(svg)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Full-section validation PR curves</title><style>body{{max-width:950px;margin:2rem auto;padding:0 1rem;background:#18232b;color:#eee;font:16px system-ui}}td,th{{padding:.4rem;text-align:right}}svg{{max-width:850px}}</style>
<h1>Full-section validation PR curves</h1>
<p>Pooled scored pixels from sample_09/10, Z = 32, 64, 96. Dashed line: labeled foreground prevalence.</p>
{svg}
<p>Circles mark the existing patch-validation thresholds. Squares mark the highest validation precision with at least {recall_key}% recall.</p>
<table><tr><th>Input</th><th>AP</th><th>Locked threshold</th><th>Precision</th><th>Recall</th><th>Candidate threshold</th><th>Precision</th><th>Recall</th></tr>{''.join(rows)}</table></html>""")


if __name__ == "__main__":
    main()

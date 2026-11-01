"""Evaluate fixed-patch checkpoints on one shared held-out Z block."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from diagnose_astro_unet import save_overlay
from train_astro_unet import UNet, digest, loss_fn, metrics


RUNS = {
    "1 patch": Path("samples/unet_patch_lr0003_01"),
    "4 patches": Path("samples/unet_four_patches_lr0003_01"),
    "16 patches": Path("samples/unet_sixteen_patches_lr0003_01"),
}


def predict(model, image):
    x = torch.from_numpy(image.copy())[None, None].float() / 255
    with torch.no_grad():
        return model(x).sigmoid()[0, 0].numpy()


def predict_tiled(model, image, size=256, stride=128):
    """Average probabilities from overlapping tiles; every pixel is covered."""
    height, width = image.shape
    ys = list(range(0, height - size + 1, stride))
    xs = list(range(0, width - size + 1, stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)
    total = np.zeros(image.shape, np.float32)
    count = np.zeros(image.shape, np.uint16)
    for y in ys:
        for x in xs:
            total[y:y + size, x:x + size] += predict(model, image[y:y + size, x:x + size])
            count[y:y + size, x:x + size] += 1
    if not np.all(count):
        raise ValueError("Tiled inference left uncovered pixels")
    return total / count


def score(probability, target):
    target_bool = target.astype(bool)
    prediction = probability >= 0.5
    counts = (
        int((prediction & target_bool).sum()),
        int((prediction & ~target_bool).sum()),
        int((~prediction & target_bool).sum()),
    )
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = torch.from_numpy(np.log(clipped / (1 - clipped)))[None, None].float()
    truth = torch.from_numpy(target.copy())[None, None].float()
    return dict(
        **metrics(*counts),
        loss=float(loss_fn(logits, truth)),
        predicted_foreground_fraction=float(prediction.mean()),
        labeled_foreground_fraction=float(target_bool.mean()),
        probability_min=float(probability.min()),
        probability_max=float(probability.max()),
    )


def pooled_score(probabilities, targets):
    return score(np.stack(probabilities), np.stack(targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("samples/astro_864691136057675736_run1/sample_01"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)

    image = tifffile.imread(args.source / "image.tif")
    mask = tifffile.imread(args.source / "mask.tif")
    valid_z = np.flatnonzero(np.ptp(image, axis=(1, 2)) > 0).tolist()
    validation_z = [z for z in valid_z if z >= 112]
    if validation_z != [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127]:
        raise ValueError("Unexpected validation Z block")
    source_hashes = {name: digest(args.source / name) for name in ["image.tif", "mask.tif", "metadata.json"]}

    rows = []
    summary = {}
    provenance = {}
    for label, run in RUNS.items():
        run_config = json.loads((run / "config.json").read_text())
        if run_config["source_sha256"] != source_hashes:
            raise ValueError(f"{label} was trained from different source files")
        checkpoint = run / "last.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = UNet(saved["width"])
        model.load_state_dict(saved["model"])
        model.eval()
        outputs = {"full_512": [], "tiled_256": []}
        targets = []
        run_dir = args.out / label.replace(" ", "_")
        run_dir.mkdir()
        for z in validation_z:
            target = mask[z]
            probabilities = {
                "full_512": predict(model, image[z]),
                "tiled_256": predict_tiled(model, image[z]),
            }
            targets.append(target)
            save_overlay(run_dir / f"z{z:03d}_reference.png", image[z], target)
            for mode, probability in probabilities.items():
                result = score(probability, target)
                rows.append(dict(model=label, checkpoint_step=saved["step"], mode=mode, z=z, **result))
                outputs[mode].append(probability)
                save_overlay(run_dir / f"z{z:03d}_{mode}.png", image[z], probability >= 0.5)
        summary[label] = {mode: pooled_score(values, targets) for mode, values in outputs.items()}
        provenance[label] = dict(
            run=str(run), checkpoint=str(checkpoint), checkpoint_step=saved["step"],
            checkpoint_sha256=digest(checkpoint), width=saved["width"],
        )

    protocol = dict(
        status="within-sample validation; not independent test or biological replication",
        validation_z=validation_z,
        excluded_constant_z=[124], threshold=0.5,
        full_512="single 512x512 forward pass",
        tiled_256="nine overlapping 256x256 tiles at XY starts 0,128,256; mean probability in overlaps",
        parameters_updated=False, source=str(args.source), source_sha256=source_hashes,
        checkpoints=provenance,
    )
    (args.out / "results.json").write_text(json.dumps(dict(protocol=protocol, summary=summary, slices=rows), indent=2))
    with (args.out / "slice_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    models = list(RUNS)
    html = '''<!doctype html><meta charset="utf-8"><title>Shared validation comparison</title>
<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px;background:#18232b;color:#eee}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #647482}img{width:32%}select{margin:8px}pre{white-space:pre-wrap}a{color:#8dbaff}</style>
<h1>1 / 4 / 16 fixed-patch models: shared validation</h1>
<p>同一组 sample_01 Z≥112 validation slices（Z=124 constant slice 排除），相同 threshold 0.5。只做 frozen inference，不更新 parameters。Full 512 与 overlapping tiled 256 均在整张 reference mask 上评分。</p>
<table><tr><th>Model</th><th>Mode</th><th>Loss</th><th>Dice</th><th>IoU</th><th>Predicted / labeled foreground</th></tr>'''
    for label in models:
        for mode in ["full_512", "tiled_256"]:
            result = summary[label][mode]
            html += f'<tr><td>{label}</td><td>{mode}</td><td>{result["loss"]:.4f}</td><td>{result["dice"]:.4f}</td><td>{result["iou"]:.4f}</td><td>{result["predicted_foreground_fraction"]:.2%} / {result["labeled_foreground_fraction"]:.2%}</td></tr>'
    html += '''</table><h2>Slice overlays</h2><select id="model"></select><select id="z"></select><p>左：full 512；中：tiled 256；右：reference。</p><img id="full"><img id="tiled"><img id="ref"><p><a href="slice_metrics.csv">Per-slice metrics CSV</a></p><script>'''
    html += "const models=" + json.dumps([label.replace(" ", "_") for label in models]) + ";const zs=" + json.dumps(validation_z) + ";"
    html += '''const m=document.getElementById("model"),z=document.getElementById("z");models.forEach(x=>m.add(new Option(x,x)));zs.forEach(x=>z.add(new Option("Z="+x,x)));function show(){const base=m.value+"/z"+String(z.value).padStart(3,"0");document.getElementById("full").src=base+"_full_512.png";document.getElementById("tiled").src=base+"_tiled_256.png";document.getElementById("ref").src=m.value+"/z"+String(z.value).padStart(3,"0")+"_reference.png"}m.onchange=show;z.onchange=show;show();</script>'''
    (args.out / "index.html").write_text(html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Fixed training-patch memorization diagnostic; no validation/test scoring."""
import argparse
import csv
import json
import time
import struct
import zlib
from pathlib import Path

import numpy as np
import tifffile
import torch
from train_astro_unet import UNet, digest, loss_fn, metrics


@torch.no_grad()
def measure(model, x, y):
    model.eval()
    logits = model(x)
    prob = logits.sigmoid()
    target = y.bool()
    pred = prob >= 0.5
    score = metrics(int((pred & target).sum()), int((pred & ~target).sum()),
                    int((~pred & target).sum()))
    score.update(loss=float(loss_fn(logits, y)),
                 predicted_foreground_fraction=float(pred.float().mean()),
                 labeled_foreground_fraction=float(y.mean()),
                 probability_min=float(prob.min()), probability_max=float(prob.max()),
                 foreground_probability_mean=float(prob[target].mean()),
                 background_probability_mean=float(prob[~target].mean()))
    return score, prob[0, 0].cpu().numpy()


def save_overlay(path, image, mask):
    """Write a lossless RGB PNG without extra dependencies."""
    rgb = np.repeat(image[..., None], 3, axis=2).astype(float)
    selected = mask.astype(bool)
    rgb[selected] = .6*rgb[selected] + .4*np.array([255, 0, 0])
    rgb = rgb.astype(np.uint8)
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind+data))
    h, w = image.shape
    scanlines = b''.join(b'\x00'+row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w,h,8,2,0,0,0))
                     + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('samples/astro_864691136057675736_run1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--full-slice', action='store_true', help='Train on the entire sample_01 Z=47 slice')
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate must be finite and positive')
    if args.steps < 1:
        parser.error('steps must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    # Same patch as the earlier smoke check: training region for folds 2/3 only.
    folder = args.data / 'sample_01'
    sources = {name: digest(folder/name) for name in ['image.tif', 'mask.tif', 'metadata.json']}
    a, m = (tifffile.imread(folder/name) for name in ['image.tif', 'mask.tif'])
    meta = json.loads((folder/'metadata.json').read_text())
    if a.shape != m.shape or a.shape != (128, 512, 512) or a.dtype != np.uint8:
        raise ValueError('Expected aligned uint8 image and matching 128x512x512 mask')
    if meta['saved_axis_order'] != 'ZYX' or meta['em_resolution_nm'] != meta['seg_resolution_nm']:
        raise ValueError('Incompatible axes/spacing')
    y0, y1, x0, x1 = (0,512,0,512) if args.full_slice else (256,512,128,384)
    a, m = a[47,y0:y1,x0:x1].copy(), m[47,y0:y1,x0:x1].copy()
    if set(np.unique(m)) != {0, 1} or np.ptp(a) == 0:
        raise ValueError('Patch must contain image contrast and both label classes')
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(sample='sample_01', z=47, y_bounds=[y0,y1], x_bounds=[x0,x1], full_slice=args.full_slice,
                  bounds_max_exclusive=True, device=device, steps=args.steps, seed=42,
                  width=16, learning_rate=args.learning_rate, threshold=0.5, target_dice=0.95,
                  normalization='uint8 / 255', augmentation=False, batch_size=1,
                  loss='BCEWithLogits + soft Dice', source_sha256=sources,
                  script_sha256=digest(Path(__file__)), model_script_sha256=digest(Path(__file__).with_name('train_astro_unet.py')),
                  torch_version=torch.__version__, image_range=[int(a.min()),int(a.max())],
                  foreground_fraction=float(m.mean()),
                  scope='Development patch for folds 2/3; not generalization evidence; fold 1 no longer untouched by development')
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    tifffile.imwrite(args.out/'image.tif', a)
    tifffile.imwrite(args.out/'mask.tif', m)
    x = torch.from_numpy(a)[None,None].float().to(device)/255
    y = torch.from_numpy(m)[None,None].float().to(device)
    model = UNet(16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    snapshots = args.out/'snapshots'
    snapshots.mkdir()
    save_overlay(args.out/'reference_overlay.png', a, m)
    history = []
    start = time.time()
    grad_norm = update_max = 0.0
    for step in range(args.steps+1):
        if step:
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(x),y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            grad_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
            if not np.isfinite(grad_norm):
                raise ValueError('Nonfinite gradient')
            before = model.head.weight.detach().clone()
            opt.step()
            update_max = float((model.head.weight.detach()-before).abs().max())
        if step % 50 == 0 or step == args.steps:
            score, prob = measure(model,x,y)
            tifffile.imwrite(snapshots/f'step_{step:04d}_probability.tif', prob)
            save_overlay(snapshots/f'step_{step:04d}_overlay.png', a, prob >= .5)
            row = dict(step=step, elapsed_seconds=time.time()-start,
                       gradient_l2=grad_norm, head_update_max=update_max, **score)
            history.append(row)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
            with (args.out/'history.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(f"step {step}/{args.steps} loss={score['loss']:.4f} Dice={score['dice']:.4f} IoU={score['iou']:.4f} pred_fg={score['predicted_foreground_fraction']:.2%} label_fg={score['labeled_foreground_fraction']:.2%} P(fg|label_fg)={score['foreground_probability_mean']:.3f} P(fg|label_bg)={score['background_probability_mean']:.3f} range=[{score['probability_min']:.3f},{score['probability_max']:.3f}] grad={grad_norm:.4g} update={update_max:.4g}",flush=True)
    tifffile.imwrite(args.out/'probability.tif',prob)
    tifffile.imwrite(args.out/'prediction.tif',(prob>=0.5).astype(np.uint8))
    # PPM needs no extra imaging package; RGB overlays retain unmodified gray background.
    for name, mask in [('reference_overlay', m), ('prediction_overlay',prob>=0.5)]:
        rgb=np.repeat(a[...,None],3,axis=2).astype(float)
        rgb[mask.astype(bool)] = .6*rgb[mask.astype(bool)] + .4*np.array([255,0,0])
        (args.out/f'{name}.ppm').write_bytes(f'P6\n{a.shape[1]} {a.shape[0]}\n255\n'.encode()+rgb.astype(np.uint8).tobytes())
    save_overlay(args.out/'prediction_overlay.png', a, prob >= .5)
    torch.save({'model':model.state_dict(),'width':16,'step':args.steps},args.out/'last.pt')
    result=dict(final=history[-1],maximum_logged_dice=max(r['dice'] for r in history),
                final_target_met=history[-1]['dice']>=.95, scope=config['scope'])
    for name,h in sources.items():
        if digest(folder/name)!=h:
            raise ValueError('Source changed')
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print('Saved:',args.out,'Final Dice target met:',result['final_target_met'],flush=True)


if __name__=='__main__':
    main()

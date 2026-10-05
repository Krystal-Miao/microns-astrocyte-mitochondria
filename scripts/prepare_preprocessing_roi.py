"""Save an unchanged EM/mask ROI and a fixed-window scientific contact sheet."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, output, center, width, z_range):
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    meta = json.loads((source / 'metadata.json').read_text())
    image = tifffile.memmap(source / 'image.tif', mode='r')
    mask = tifffile.memmap(source / 'mask.tif', mode='r')
    if meta['saved_axis_order'] != 'ZYX' or image.shape != mask.shape or image.dtype != np.uint8:
        raise ValueError('Expected aligned ZYX uint8 EM and mask')
    nz, ny, nx = image.shape
    x, y, z = center
    z0, z1 = z_range
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz and 0 <= z0 <= z < z1 <= nz):
        raise ValueError('Center or exclusive Z bounds invalid')
    if not 1 <= width <= min(nx, ny):
        raise ValueError('XY width must fit source')
    x0 = min(max(x-width//2, 0), nx-width)
    y0 = min(max(y-width//2, 0), ny-width)
    lo = np.array([x0, y0, z0]); hi = np.array([x0+width, y0+width, z1])
    crop = np.array(image[z0:z1, y0:y0+width, x0:x0+width])
    labels = np.array(mask[z0:z1, y0:y0+width, x0:x0+width])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary target-cell mask')
    before = {name: sha256(source/name) for name in ['image.tif', 'mask.tif', 'metadata.json']}
    output.mkdir(parents=True)
    tifffile.imwrite(output/'image.tif', crop, photometric='minisblack', metadata={'axes': 'ZYX'})
    tifffile.imwrite(output/'mask.tif', labels, photometric='minisblack', metadata={'axes': 'ZYX'})
    # Nearest-neighbor magnification, fixed 0..255 window for every panel.
    scale = 2; tile = width*scale; columns = 4; rows = (len(crop)+columns-1)//columns
    sheet = Image.new('RGB', (columns*tile, rows*(tile+28)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, plane in enumerate(crop):
        left = (i % columns)*tile; top = (i//columns)*(tile+28)
        draw.text((left+8, top+7), f'Source Z={z0+i} | raw EM', fill='black')
        panel = Image.fromarray(plane).resize((tile,tile), Image.Resampling.NEAREST).convert('RGB')
        pd = ImageDraw.Draw(panel)
        # One scale bar per slice in the original physical resolution.
        bar = round(500/meta['em_resolution_nm'][0]*scale)
        if bar < tile-20:
            pd.line((10,tile-12,10+bar,tile-12),fill='white',width=3)
            pd.text((10,tile-30),'0.5 um',fill='white',stroke_width=1,stroke_fill='black')
        sheet.paste(panel,(left,top+28))
    sheet.save(output/'raw_contact_sheet.png')
    origin = np.array(meta['voxel_bbox']['min'])
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sample': str(source), 'source_sha256': before,
        'root_id': meta['root_id'], 'dataset': meta['dataset'],
        'materialization_version': meta['materialization_version'],
        'selected_point_source_xyz': list(center),
        'selected_point_inside_target_mask': bool(mask[z,y,x]),
        'source_local_bbox_xyz': {'min': lo.tolist(), 'max': hi.tolist()},
        'voxel_bbox': {'min': (origin+lo).tolist(), 'max': (origin+hi).tolist()},
        'bbox_max_is_exclusive': True, 'saved_axis_order': 'ZYX',
        'em_resolution_nm': meta['em_resolution_nm'], 'image_shape': list(crop.shape),
        'occupancy': float(labels.mean()), 'mask_is_mitochondria_ground_truth': False,
        'processing': 'None. Exact subarray of original EM; binary astrocyte mask unchanged.',
        'display': {'window': [0,255], 'magnification': 2, 'interpolation': 'nearest',
                    'note': 'Scale bars appear only in contact sheet, never in TIFF.'},
        'per_slice': [{'source_z': z0+i, 'mean_intensity': float(a.mean()),
                       'std_intensity_not_noise_estimate': float(a.std()),
                       'target_occupancy': float(labels[i].mean())} for i,a in enumerate(crop)],
        'scope': 'Exploratory user-selected ROI, no confirmed mitochondria annotation or evaluation ground truth.',
    }
    (output/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    assert np.array_equal(tifffile.imread(output/'image.tif'), image[z0:z1,y0:y0+width,x0:x0+width])
    assert np.array_equal(tifffile.imread(output/'mask.tif'), mask[z0:z1,y0:y0+width,x0:x0+width])
    assert before == {name: sha256(source/name) for name in before}, 'Source changed'
    print(f'Verified exact crop and unchanged source files: {output}')
    print(f'Source XYZ bounds [{lo.tolist()}, {hi.tolist()}); ZYX shape {crop.shape}')
    print(f'Target occupancy: {labels.mean():.2%}; selected point in mask: {bool(mask[z,y,x])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--center-xyz', type=int, nargs=3, default=[213,430,47])
    parser.add_argument('--width-xy', type=int, default=192)
    parser.add_argument('--z-range', type=int, nargs=2, default=[42,53], help='Source Z range, exclusive upper bound')
    args = parser.parse_args()
    prepare(args.sample_dir.resolve(), args.output_dir.resolve(), args.center_xyz,args.width_xy,args.z_range)

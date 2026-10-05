"""Build an offline slice viewer from downloaded samples; source TIFFs stay intact."""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def build(data_dir, output):
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose another --output-dir")
    records = []
    sources = sorted(data_dir.glob('sample_*/metadata.json'))
    if not sources:
        raise ValueError(f"No sample metadata under {data_dir}")
    # Validate inputs before writing output.
    for path in sources:
        meta = json.loads(path.read_text())
        if meta['saved_axis_order'] != 'ZYX':
            raise ValueError('Expected ZYX input')
        image = tifffile.memmap(path.parent / 'image.tif', mode='r')
        mask = tifffile.memmap(path.parent / 'mask.tif', mode='r')
        if image.shape != mask.shape or image.ndim != 3 or image.dtype != np.uint8:
            raise ValueError(f'Invalid shape or image dtype: {path.parent}')
        records.append(dict(name=path.parent.name, shape=list(image.shape),
                            start=meta['preview_local_z'], origin=meta['voxel_bbox']['min'],
                            resolution=meta['em_resolution_nm'], root_id=meta['root_id']))
        del image, mask
    output.mkdir(parents=True)
    for record, path in zip(records, sources):
        image = tifffile.memmap(path.parent / 'image.tif', mode='r')
        mask = tifffile.memmap(path.parent / 'mask.tif', mode='r')
        assets = output / record['name']
        assets.mkdir()
        record['occupancy'] = []
        for z in range(image.shape[0]):
            selected = np.asarray(mask[z]) != 0
            record['occupancy'].append(float(selected.mean()))
            Image.fromarray(np.asarray(image[z])).save(assets / f'em_{z:03d}.png')
            rgba = np.zeros((*selected.shape, 4), dtype=np.uint8)
            rgba[selected] = [255, 0, 0, 255]
            Image.fromarray(rgba).save(assets / f'mask_{z:03d}.png')
        del image, mask
        print(f"Built {record['name']}: {record['shape'][0]} slices", flush=True)
    template = Path(__file__).with_name('sample_viewer.html').read_text()
    payload = json.dumps(records).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('__SAMPLE_DATA__', payload))
    print(f"Open in your browser: {output / 'index.html'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.data_dir.resolve(), args.output_dir.resolve())

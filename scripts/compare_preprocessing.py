"""Run a recorded 2D CLAHE/NLM pilot on source slices, then crop the fixed ROI."""
import argparse
import html as html_utils
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means

from prepare_preprocessing_roi import sha256


def run(roi_dir, output):
    if output.exists():
        raise ValueError('Choose a new output directory; existing results are never overwritten')
    meta = json.loads((roi_dir/'metadata.json').read_text())
    source = Path(meta['source_sample'])
    hashes = {name: sha256(source/name) for name in meta['source_sha256']}
    if hashes != meta['source_sha256']:
        raise ValueError('Source files changed since ROI preparation')
    image = tifffile.memmap(source/'image.tif', mode='r')
    mask = tifffile.imread(roi_dir/'mask.tif')
    x0,y0,z0 = meta['source_local_bbox_xyz']['min']
    x1,y1,z1 = meta['source_local_bbox_xyz']['max']
    raw_uint8 = tifffile.imread(roi_dir/'image.tif')
    assert np.array_equal(raw_uint8,image[z0:z1,y0:y1,x0:x1])
    assert np.array_equal(mask,tifffile.memmap(source/'mask.tif',mode='r')[z0:z1,y0:y1,x0:x1])
    roi_hashes = {name: sha256(roi_dir/name) for name in ['image.tif','mask.tif','metadata.json']}
    valid = np.array([np.ptp(s) > 0 for s in raw_uint8])
    if not valid.any():
        raise ValueError('All ROI slices are constant; inspect the source before preprocessing')
    invalid_z = [z0+i for i,ok in enumerate(valid) if not ok]
    clahe = dict(kernel_size=(64,64),clip_limit=0.01,nbins=256)
    nlm = dict(patch_size=5,patch_distance=6,fast_mode=True,channel_axis=None,
               sigma=0.0,preserve_range=True)
    arrays = {'raw': raw_uint8.astype(np.float32)/255,
              'clahe': [], 'nlm_h2': [], 'nlm_h3': []}
    for z in range(z0,z1):
        plane = np.asarray(image[z],dtype=np.float32)/255
        if not valid[z-z0]:
            for name in ['clahe','nlm_h2','nlm_h3']:
                arrays[name].append(plane[y0:y1,x0:x1].copy())
            print(f'QC: constant ROI at Z={z}; kept unchanged, excluded from statistics',flush=True)
            continue
        arrays['clahe'].append(equalize_adapthist(plane,**clahe)[y0:y1,x0:x1])
        for factor,name in [(2.0,'nlm_h2'),(3.0,'nlm_h3')]:
            arrays[name].append(denoise_nl_means(plane,h=factor/255,**nlm)[y0:y1,x0:x1])
        print(f'Processed full source slice Z={z}; saved ROI in memory',flush=True)
    for name in arrays:
        arrays[name] = np.asarray(arrays[name],dtype=np.float32)
        assert arrays[name].shape == raw_uint8.shape and np.isfinite(arrays[name]).all()
        assert arrays[name].min() >= 0 and arrays[name].max() <= 1
    output.mkdir(parents=True)
    residuals = {name: arrays['raw']-arrays[name] for name in ['nlm_h2','nlm_h3']}
    for name,array in {**arrays, **{f'residual_{k}':v for k,v in residuals.items()}}.items():
        tifffile.imwrite(output/f'{name}.tif',array,photometric='minisblack',metadata={'axes':'ZYX'})
        assert np.array_equal(tifffile.imread(output/f'{name}.tif'),array)

    low,high = np.percentile(arrays['raw'][valid],[1,99]).tolist()
    if high <= low:
        low,high=0.,1.
    panels = [('raw','Raw',0,1), ('raw','Raw: linear display',low,high),
              ('clahe','CLAHE (contrast only)',0,1),
              ('nlm_h2','NLM h=2/255',0,1), ('nlm_h3','NLM h=3/255',0,1)]
    for i,z in enumerate(range(z0,z1)):
        fig,axes = plt.subplots(1,5,figsize=(16,3.8),layout='constrained')
        for ax,(name,title,vmin,vmax) in zip(axes,panels):
            ax.imshow(arrays[name][i],cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_title(title); ax.set_axis_off()
        note = 'QC: CONSTANT ROI, NOT PROCESSED; excluded from statistics' if not valid[i] else 'Linear control: valid raw ROI 1st/99th percentiles, fixed across slices; display only'
        fig.suptitle(f'{source.name} | source Z={z} | display 0..1 except linear control [{low:.3f}, {high:.3f}]\n{note}')
        fig.savefig(output/f'comparison_z{z:03d}.png',dpi=150);plt.close(fig)

    focus = meta['selected_point_source_xyz'][2]-z0
    # A separate shared-window view makes subtle NLM changes easier to inspect.
    fig,axes = plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
    for ax,name in zip(axes,['raw','nlm_h2','nlm_h3']):
        ax.imshow(arrays[name][focus],cmap='gray',vmin=low,vmax=high,interpolation='nearest')
        ax.set_title(name);ax.set_axis_off()
    fig.suptitle(f'Z={z0+focus} | shared raw-derived display window [{low:.3f}, {high:.3f}]\nDisplay only; TIFF values unchanged')
    fig.savefig(output/'nlm_detail.png',dpi=160);plt.close(fig)
    limit = max(float(np.max(np.abs(a))) for a in residuals.values()) or 1/255
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,(name,a) in zip(axes,residuals.items()):
        im=ax.imshow(a[focus],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_title(f'raw - {name}');ax.set_axis_off()
    fig.colorbar(im,ax=axes,label='Signed change (normalized intensity); same scale')
    fig.suptitle(f'Z={z0+focus}: residual is changed signal, NOT proven noise')
    fig.savefig(output/'residual.png',dpi=160);plt.close(fig)

    changes = {}
    for name,a in arrays.items():
        if name=='raw':continue
        diff=(a-arrays['raw'])[valid]
        changes[name]={'mean_absolute_change_0_255': float(np.mean(np.abs(diff))*255),
                       'target_mask_mean_absolute_change_0_255':float(np.mean(np.abs(diff)[mask[valid]!=0])*255) if np.any(mask[valid]) else None,
                       'max_absolute_change_0_255':float(np.max(np.abs(diff))*255)}
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),roi_metadata=meta,
                  source_sha256=hashes,roi_sha256=roi_hashes,
                  versions={p:version(p) for p in ['numpy','scipy','scikit-image','tifffile','matplotlib']},
                  python=platform.python_version(),normalization='float32(raw_uint8) / 255; no per-slice normalization',
                  processing='Independent 2D full sample XY slices, then ROI crop; no mask applied before filtering',
                  noise_estimation='Not performed: optional PyWavelets dependency absent.',
                  parameter_caveat='Fixed exploratory h values, not estimated noise or optimized settings. sigma=0 disables noise-variance correction.',
                  parameters={'clahe':clahe,'nlm_common':nlm,'h_values_0_255':[2.0,3.0],'h_values_normalized':[2/255,3/255]},
                  qc={'constant_roi_source_z':invalid_z,'policy':'Keep unchanged, skip processing, exclude from intensity window and change statistics. Cause unconfirmed.'},
                  display={'main_except_linear':[0,1],'raw_linear_control':[low,high],
                           'linear_window_selection':'1st/99th percentiles of nonconstant raw ROI slices; fixed across Z; clipped outside window; no filtering or TIFF changes',
                           'nlm_detail':[low,high],'signed_residual':[-limit,limit]},
                  changes_not_quality_scores=changes,
                  limitations=['No clean EM or mitochondria ground truth; no quality ranking or segmentation claim.',
                               'ROI touches sample Y boundary: source padding still affects nearby pixels.',
                               'One selected ROI only; independent slices may differ in appearance across Z.'])
    assert hashes=={name:sha256(source/name) for name in hashes}
    assert roi_hashes=={name:sha256(roi_dir/name) for name in roi_hashes}
    report['verification']='Source and ROI hashes unchanged; all six TIFFs read back exactly; aligned shapes and finite ranges verified.'
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Offline report: no server, dependencies, or data uploads.
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>预处理比较 · SAMPLENAME</title>
<style>body{font:17px system-ui;background:#14202b;color:#edf4fa;margin:24px;line-height:1.6}img{max-width:100%;background:white}button,input{font:inherit}a{color:#a8daff}</style>
<h1>SAMPLENAME · 预处理比较</h1><p>五列依次：原图、原图仅线性调整显示、CLAHE（局部对比度增强）、NLM h=2/255、NLM h=3/255。不是线粒体标注，也不是质量排名。</p>
<p style="color:#ffcf86">QCNOTE</p>
<p>第二列没有降噪，只按该 ROI 原图的 1%–99% 分位数拉伸显示范围；窗口在所有层中固定，范围外显示为黑或白，TIFF 不改动。其余四列显示范围固定为 0–1。比较第二、三列，观察简单显示调整与 CLAHE 的差别；比较 NLM 细节请看下方共同窗口的三列图。</p>
<button onclick="step(-1)">上一层</button> <input id="z" type="range" min="ZMIN" max="ZMAX" value="ZFOCUS" oninput="show()"> <button onclick="step(1)">下一层</button> <span id="label"></span>
<p><img id="comparison" alt="原图、线性显示、CLAHE及两档NLM比较"></p>
<h2>同一窗口放大观察 NLM 改变</h2><p>下面固定为 Z=ZFOCUS；三列用同一个来自原图的显示窗口，未改变 TIFF。</p><img src="nlm_detail.png" alt="NLM细节对照">
<h2>原图减去 NLM 的残差</h2><p>下面固定为 Z=ZFOCUS。红/蓝表示正/负变化，接近白色表示变化较小。出现结构轮廓提示真实细节可能被改变；不能把残差直接叫作噪声。</p><img src="residual.png" alt="有符号残差">
<p><a href="report.json">参数、来源与检查记录</a>。原始数据未改动。暂无干净图像或线粒体真值，不能据此报告分割精度改善。</p>
<script>const z=document.getElementById('z');function show(){document.getElementById('label').textContent='源 sample Z = '+z.value;document.getElementById('comparison').src='comparison_z'+String(z.value).padStart(3,'0')+'.png'}function step(d){z.value=Math.max(Number(z.min),Math.min(Number(z.max),Number(z.value)+d));show()}show();</script></html>'''
    qc_note = f'异常提醒：源 Z={invalid_z} 为常量图层，未处理，已排除在显示窗口估算及变化统计之外；原始文件保留，原因待确认。' if invalid_z else '本 ROI 未发现全层恒定灰度；这不等于完整图像质量认证。'
    html=html.replace('ZMIN',str(z0)).replace('ZMAX',str(z1-1)).replace('ZFOCUS',str(z0+focus)).replace('SAMPLENAME',html_utils.escape(source.name)).replace('QCNOTE',qc_note)
    (output/'index.html').write_text(html)
    print(json.dumps({'changes':changes,'output':str(output)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roi-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    run(args.roi_dir.resolve(),args.output_dir.resolve())

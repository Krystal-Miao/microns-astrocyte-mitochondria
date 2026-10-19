"""Compare fixed exploratory NLM/bilateral settings on fixed sample regions."""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.restoration import denoise_nl_means, denoise_bilateral
from prepare_preprocessing_roi import sha256


def run(manifest_path, output, sample="sample_01"):
    if output.exists():
        raise ValueError('Output exists; use a new directory')
    manifest=json.loads(manifest_path.read_text())
    candidates=[r for r in manifest['candidates'] if r['sample']==sample]
    assert len(candidates)==3
    roi=Path(candidates[0]['source_roi']); meta=json.loads((roi/'metadata.json').read_text())
    source=Path(meta['source_sample'])
    hashes={str(p):sha256(p) for folder in [source,roi] for p in [folder/'image.tif',folder/'mask.tif',folder/'metadata.json']}
    for p,h in hashes.items():
        expected=manifest['source_sha256'].get(p)
        if expected: assert h==expected
    assert {name:sha256(source/name) for name in meta['source_sha256']}==meta['source_sha256']
    image=tifffile.memmap(source/'image.tif',mode='r')
    zs=candidates[0]['source_z_neighbors']; assert len(zs)==3 and (sample!='sample_02' or 108 not in zs)
    x0,y0,z0=meta['source_local_bbox_xyz']['min'];x1,y1,z1=meta['source_local_bbox_xyz']['max']
    assert np.array_equal(tifffile.imread(roi/'image.tif'),image[z0:z1,y0:y1,x0:x1])
    params={
        'raw':{},
        'nlm_h2':dict(h=2/255,patch_size=5,patch_distance=6,fast_mode=True,sigma=0,preserve_range=True,channel_axis=None),
        'nlm_h3':dict(h=3/255,patch_size=5,patch_distance=6,fast_mode=True,sigma=0,preserve_range=True,channel_axis=None),
        'bilateral_c2':dict(sigma_color=2/255,sigma_spatial=1,win_size=7,bins=10000,mode='reflect',cval=0,channel_axis=None),
        'bilateral_c4':dict(sigma_color=4/255,sigma_spatial=1,win_size=7,bins=10000,mode='reflect',cval=0,channel_axis=None)}
    if sample != 'sample_01':
        params={k:params[k] for k in ['raw','nlm_h2','bilateral_c2']}
    arrays={name:[] for name in params}; timings={name:0. for name in params}
    for z in zs:
        plane=np.asarray(image[z],dtype=np.float32)/255
        assert np.ptp(plane)>0
        for name,kw in params.items():
            t=time.perf_counter()
            result=plane if name=='raw' else (denoise_nl_means(plane,**kw) if name.startswith('nlm') else denoise_bilateral(plane,**kw))
            timings[name]+=time.perf_counter()-t
            arrays[name].append(result[y0:y1,x0:x1].copy())
        print(f'Processed full 512x512 source Z={z}',flush=True)
    arrays={k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()}
    output.mkdir(parents=True)
    for name,a in arrays.items():
        assert a.shape==(3,192,192) and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
        tifffile.imwrite(output/f'{name}.tif',a,metadata={'axes':'ZYX'},photometric='minisblack')
        assert np.array_equal(tifffile.imread(output/f'{name}.tif'),a)
    residuals={k:arrays['raw']-a for k,a in arrays.items() if k!='raw'}
    for k,a in residuals.items():
        tifffile.imwrite(output/f'residual_{k}.tif',a,metadata={'axes':'ZYX'},photometric='minisblack')
        assert np.array_equal(tifffile.imread(output/f'residual_{k}.tif'),a)
    lo,hi=np.array(candidates[0]['display_window'])/255
    limit=max(float(np.abs(a).max()) for a in residuals.values())
    rows=[]; profiles=[]; sections=[]
    names=list(arrays)
    for zi,z in enumerate(zs):
        sections.append(f'<section class="layer" id="z{z}"><h2>Source Z={z}</h2>')
        for r in candidates:
            bx,by=r['source_local_xy_bbox']['min']; ex,ey=r['source_local_xy_bbox']['max'];sl=(slice(by-y0,ey-y0),slice(bx-x0,ex-x0))
            fig,axes=plt.subplots(1,len(names),figsize=(12,3),layout='constrained')
            for ax,k in zip(axes,names):
                patch=arrays[k][zi][sl]
                ax.imshow(patch,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(k);ax.axis('off')
                rows.append(dict(candidate=r['id'],source_z=z,method=k,mean_255=float(patch.mean()*255),std_255_not_noise=float(patch.std()*255),mae_change_255=float(np.abs(patch-arrays['raw'][zi][sl]).mean()*255)))
            fig.suptitle(f"{r['id']} | Z={z} | identical raw-derived display window")
            filename=f"{r['id']}_z{z}.png";fig.savefig(output/filename,dpi=150);plt.close(fig)
            sections.append(f'<h3>{r["id"]}</h3><img src="{filename}">')
            fig,axes=plt.subplots(1,len(residuals),figsize=(12,3),layout='constrained')
            for ax,(k,a) in zip(axes,residuals.items()):
                im=ax.imshow(a[zi][sl]*255,cmap='RdBu_r',vmin=-limit*255,vmax=limit*255,interpolation='nearest');ax.set_title(k);ax.axis('off')
            fig.colorbar(im,ax=axes,label='Raw minus output (0–255 units)')
            fig.suptitle('Residual = changed content, NOT proven noise; same scale across all panels and Z')
            filename=f"{r['id']}_residual_z{z}.png";fig.savefig(output/filename,dpi=150);plt.close(fig)
            sections.append(f'<details><summary>View residuals (changed content, not proven noise)</summary><img src="{filename}"></details>')
        r=next(r for r in candidates if r['purpose']=='boundary');(ax0,ay0),(ax1,ay1)=r['line_source_xy']
        assert ay0==ay1 or ax0==ax1
        n=max(abs(ax1-ax0),abs(ay1-ay0))+1
        xs=np.linspace(ax0,ax1,n).astype(int);ys=np.linspace(ay0,ay1,n).astype(int)
        distance=np.sqrt(((xs-xs[0])*meta['em_resolution_nm'][0])**2+((ys-ys[0])*meta['em_resolution_nm'][1])**2)
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for k in names:
            vals=arrays[k][zi,ys-y0,xs-x0]*255
            ax.plot(distance,vals,label=k,marker='.',linewidth=1)
            profiles.extend(dict(source_z=z,method=k,distance_nm=int(d),intensity_255=float(v)) for d,v in zip(distance,vals))
        ax.set(xlabel='Distance along fixed line (nm)',ylabel='Intensity (0–255 units)',title=f'Source XY ({ax0},{ay0}) to ({ax1},{ay1}), Z={z}; exploratory profile');ax.legend();ax.grid(alpha=.2)
        fig.savefig(output/f'profile_z{z}.png',dpi=150);plt.close(fig)
        sections.append(f'<h3>Boundary intensity profile</h3><img src="profile_z{z}.png"></section>')
    for filename,data in [('diagnostics.csv',rows),('profiles.csv',profiles)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),sample=sample,source_z=zs,
        parameters=params,seconds_full_slice_processing=timings,versions={p:version(p) for p in ['numpy','scikit-image','tifffile','matplotlib']},
        source_sha256=hashes,candidate_manifest_sha256=sha256(manifest_path),candidates=candidates,
        authorization='User approved proceeding after reviewing candidate positions and experimental design; frozen for exploratory tuning, not biological identity confirmation.',
        display_window_255=candidates[0]['display_window'],residual_limit_255=limit*255,
        limitations=['Standard deviation is intensity variation, not noise sigma','No clean-reference SNR/PSNR/SSIM or accuracy ranking','Candidate patches overlap','Only one astrocyte','Full-slice processing before ROI crop; no mask-zeroing','Only the named sample processed; no independent biological validation','No chosen best method','Bilateral and NLM strengths are not equivalent'],diagnostics=rows)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    options=''.join(f'<option>{z}</option>' for z in [zs[1],zs[0],zs[2]])
    stage='Exploratory parameter comparison.' if sample=='sample_01' else 'Regional evaluation with fixed sample_01 parameters; no retuning.'
    labels=', '.join(names)
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{sample} - Denoising comparison</title><style>body{{font:17px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#14212a;color:#eef3f7}}img{{width:100%}}select{{padding:12px}}details{{margin:15px 0}}p{{line-height:1.6}}</style><h1>{sample}: Raw / NLM / Bilateral</h1><p>{stage} Three fixed candidate regions; these are not verified mitochondria labels. Original EM is unchanged.</p><p>Raw-derived display window: {candidates[0]['display_window']} (0-255 units, display only), fixed across methods and slices within this sample. Left to right: {labels}. NLM h2/h3 use h=2/255 or 3/255; Bilateral c2/c4 use sigma_color=2/255 or 4/255, with sigma_spatial=1 pixel. These are not equivalent strengths.</p><p>Inspect boundary and texture preservation alongside residuals. Uniform-region SD measures intensity variation, not noise. There is no validated best method. Profiles show sampled intensities, not fitted boundary widths. Adjacent sections contain different anatomy. Display and residual scales may differ between samples.</p><label>Source Z: <select id="z">{options}</select></label>'''+''.join(sections)+'''<script>function show(){document.querySelectorAll('.layer').forEach(e=>e.hidden=e.id!=='z'+document.getElementById('z').value)}document.getElementById('z').onchange=show;show();</script></html>'''
    (output/'index.html').write_text(html)
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    print('Verified TIFF readbacks, ranges and unchanged input hashes:',output)
    print(json.dumps([r for r in rows if r['source_z']==zs[1] and r['candidate'].endswith('_uniform')],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--sample',choices=['sample_01','sample_02','sample_03'],default='sample_01');a=p.parse_args();run(a.manifest,a.output_dir,a.sample)

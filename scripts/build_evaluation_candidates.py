"""Build a read-only-source candidate review page; no denoising or labels inferred."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw

# Visually proposed ROI-local XY centers. All require human review.
PROPOSALS = [
    [(112,104,'boundary'), (94,100,'texture'), (100,80,'uniform')],
    [(76,120,'boundary'), (96,112,'texture'), (148,60,'uniform')],
    [(90,145,'boundary'), (110,120,'texture'), (132,132,'uniform')],
]
LABELS = {'boundary':'边界候选', 'texture':'内部纹理候选', 'uniform':'相对均匀候选（不是纯噪声）'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(base, output):
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    output.mkdir(parents=True)
    records=[]; sections=[]; hashes={}
    for sample, proposals in enumerate(PROPOSALS,1):
        source=base/f'roi_{sample:02d}'
        meta=json.loads((source/'metadata.json').read_text())
        for name in ('image.tif','mask.tif','metadata.json'):
            p=source/name; hashes[str(p)]=sha(p)
        image=tifffile.imread(source/'image.tif'); mask=tifffile.imread(source/'mask.tif')
        assert image.shape==mask.shape and image.dtype==np.uint8
        origin=np.array(meta['source_local_bbox_xyz']['min'])
        world=np.array(meta['voxel_bbox']['min']); resolution=np.array(meta['em_resolution_nm'])
        z=meta['selected_point_source_xyz'][2]-int(origin[2])
        valid=np.array([a.min()!=a.max() for a in image])
        lo,hi=np.percentile(image[valid],[1,99]); assert hi>lo
        sections.append(f'<h2>sample_{sample:02d} — '+('调参候选' if sample==1 else '预留评价候选')+'</h2>')
        sections.append('<p>只在原图上确认位置，不根据降噪结果挑区域。同一细胞内的区域不构成独立生物学重复。</p>')
        for number,(x,y,kind) in enumerate(proposals,1):
            ident=f's{sample:02d}_{kind}'; x0=x-16;y0=y-16;x1=x+16;y1=y+16
            assert 0<=x0<x1<=image.shape[2] and 0<=y0<y1<=image.shape[1]
            local=(origin+np.array([x,y,z])).tolist()
            line=([[x,y-12],[x,y+12]] if sample==3 else [[x-12,y],[x+12,y]]) if kind=='boundary' else None
            record={'id':ident,'sample':f'sample_{sample:02d}','purpose':kind,'status':'pending',
                'selection_basis':('Lowest central-slice intensity SD among 32px windows on a 4px grid with target occupancy >=99%; not a noise-only region' if kind=='uniform' else 'Visual proposal on raw EM; unconfirmed identity'),'role':'tuning' if sample==1 else 'held_out_candidate','roi_local_center_xyz':[x,y,z],
                'source_local_center_xyz':local,'global_center_nm':((world+[x,y,z])*resolution).tolist(),
                'source_local_xy_bbox':{'min':[int(origin[0])+x0,int(origin[1])+y0],'max':[int(origin[0])+x1,int(origin[1])+y1]},
                'bbox_max_exclusive':True,'source_z_neighbors':[int(origin[2])+z+d for d in [-1,0,1]],
                'target_mask_fraction':float(mask[z,y0:y1,x0:x1].mean()),
                'line_source_xy':[[int(origin[0])+a,int(origin[1])+b] for a,b in line] if line else None,
                'display_window':[float(lo),float(hi)],'source_roi':str(source)}
            records.append(record)
            imgs=[]
            for delta in [-1,0,1]:
                zi=z+delta; assert 0<=zi<len(image) and valid[zi]
                assert not(sample==2 and int(origin[2])+zi==108)
                plane=image[zi]; scaled=np.uint8(np.clip((plane-lo)/(hi-lo),0,1)*255)
                for mode in ['raw','mask']:
                    rgb=np.repeat(scaled[...,None],3,axis=2)
                    if mode=='mask':
                        target=mask[zi].astype(bool); rgb[target]=np.uint8(rgb[target]*.65+np.array([255,0,0])*.35)
                    panel=Image.fromarray(rgb).resize((576,576),Image.Resampling.NEAREST)
                    draw=ImageDraw.Draw(panel);draw.rectangle((x0*3,y0*3,x1*3-1,y1*3-1),outline='yellow',width=2)
                    if line: draw.line(tuple(v*3 for point in line for v in point),fill='cyan',width=2)
                    bar=round(500/resolution[0]*3); draw.line((12,558,12+bar,558),fill='white',width=3); draw.text((12,540),'0.5 um',fill='white',stroke_width=1,stroke_fill='black')
                    panel.save(output/f'{ident}_{delta+1}_{mode}.png')
                imgs.append(f'<figure><img data-base="{ident}_{delta+1}" src="{ident}_{delta+1}_raw.png"><figcaption>Source Z={int(origin[2])+zi}</figcaption></figure>')
            sections.append(f'<article data-id="{ident}"><h3>{ident} · {LABELS[kind]}</h3><p>中心 sample XYZ={local}；黄色框 32 × 32 px（0.256 × 0.256 µm）；中心层框内 astrocyte 占比 {record["target_mask_fraction"]:.1%}。</p>')
            if kind=='uniform': sections.append('<p>此框按目标 mask 内较低灰度标准差选出，只表示相对均匀，仍可能位于细胞器内部；不能用它定义真实噪声。</p>')
            if line: sections.append('<p>青色线为拟议灰度剖面线，需要确认是否穿过目标边界。</p>')
            sections.append('<div class="row">'+''.join(imgs)+'</div><label>审阅：<select><option value="pending">待确认</option><option value="accept">保留候选</option><option value="reject">排除</option><option value="revise">需要调整</option></select></label> <input placeholder="备注：结构、位置或需要调整的原因"></article>')
    manifest={'status':'candidate_only','no_denoising':True,'source_sha256':hashes,'candidates':records,
        'limitations':['No confirmed mitochondria labels','Uniform patches are not pure noise','Candidates may overlap within a sample','No clean reference or validated noise model','All crops from one astrocyte'],
        'display':'1st/99th percentile window per ROI from nonconstant slices; fixed across neighbors. Nearest-neighbor magnification. Source arrays unchanged.'}
    (output/'candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>预处理评价区域候选</title><style>body{font:16px system-ui;background:#14212a;color:#eef3f7;max-width:1500px;margin:30px auto;padding:16px}article{background:#20313d;padding:18px;margin:24px 0;border-radius:12px}.row{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%;image-rendering:pixelated}button,select,input{padding:10px;margin:6px}input{width:50%}figcaption{text-align:center}p{line-height:1.6}</style><h1>预处理评价区域候选</h1><p>所有位置均待人工确认。黄色框是评价候选，不是分割标签；红色表示 astrocyte mask，不是接触区域或线粒体。左右图为相邻截面，不是同一结构的重复拍摄。</p><p>当前显示为原始 EM 的固定线性窗口（每个 ROI 非恒定层的 1%–99%）；未降噪、未改变 TIFF。不同 ROI 的窗口不同。sample_02 的异常 Z=108 不参与显示窗口计算或候选图。候选可能重叠，不能按独立样本统计。</p><button id="mask">显示 astrocyte mask</button><button id="export">导出审阅 JSON</button><p>选择不会自动保存；关闭前请导出审阅 JSON。离线页面不会上传数据。</p>'''+''.join(sections)+'''<script>const manifest=MANIFEST;let overlay=false;document.getElementById('mask').onclick=()=>{overlay=!overlay;document.querySelectorAll('img[data-base]').forEach(e=>e.src=e.dataset.base+(overlay?'_mask.png':'_raw.png'));document.getElementById('mask').textContent=overlay?'隐藏 astrocyte mask':'显示 astrocyte mask'};document.getElementById('export').onclick=()=>{const result=JSON.parse(JSON.stringify(manifest));document.querySelectorAll('article').forEach(e=>{const r=result.candidates.find(x=>x.id===e.dataset.id);r.status=e.querySelector('select').value;r.review_note=e.querySelector('input').value});result.reviewed_at=new Date().toISOString();const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='candidate_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></html>'''.replace('MANIFEST',json.dumps(manifest).replace('<','\\u003c'))
    (output/'index.html').write_text(page)
    assert all(sha(Path(p))==h for p,h in hashes.items()),'Source changed'
    assert len(records)==9 and len(list(output.glob('*.png')))==54
    print(f'Created {output}/index.html; 9 pending candidates; 54 previews; source hashes unchanged.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();build(args.data_dir,args.output_dir)

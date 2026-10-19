# MICrONS preprocessing 小范围实验记录

日期：2026-10-02。当前状态：原始对照 ROI 已建立；CLAHE 与两档固定强度 NLM 的 11 层比较已完成。未训练模型，未评估线粒体分割精度。

## 固定原始对照

- 来源：`samples/astro_864691136057675736_run1/sample_01`。
- 用户选点：sample 局部 XYZ `(213, 430, 47)`，该点在目标 astrocyte mask 内。
- ROI 边界（sample 局部 XYZ，最大值不包含）：`[117,320,42]` 到 `[309,512,53]`。
- 形状：ZYX `(11,192,192)`，原始分辨率 XYZ `8 × 8 × 40 nm`。
- Y 范围靠近 sample 边缘，裁剪框向内平移以保留 192 像素宽度；选点并非严格位于裁剪中心。
- 输出：`samples/astro_864691136057675736_run1/preprocessing/roi_01/`。
- `image.tif` 和 `mask.tif` 是原始数组的精确子数组；没有重采样、对比度增强或降噪。
- `metadata.json` 保存空间转换、选点、逐层强度描述统计、来源文件 SHA-256。
- `raw_contact_sheet.png` 按 Z=42–52 排列，统一 0–255 显示范围，2 倍最近邻显示，比例尺 0.5 µm。比例尺不写入 TIFF。

运行方式（输出目录必须不存在）：

```bash
python -B scripts/prepare_preprocessing_roi.py \
  --sample-dir samples/astro_864691136057675736_run1/sample_01 \
  --output-dir samples/astro_864691136057675736_run1/preprocessing/roi_01
```

已验证输出 EM/mask 与输入切片逐像素相等，原始三个来源文件的哈希在执行前后不变。ROI 内目标细胞约占 38.37%。这不是线粒体占比。

## 方法初筛：原理与当前选择

以下是待验证方案，不是对本数据已取得效果的声明。

| 对照/方法 | 作用 | 本轮定位 |
|---|---|---|
| Raw | 不处理 | 必须保留的参照；处理结果未必优于原图 |
| 固定线性显示窗口 | 改变观察时的明暗映射 | 分清“看得更清楚”与“降噪”；同一比较中使用同一窗口 |
| CLAHE | 局部直方图均衡化，增强局部对比度 | 单独比较，不与降噪组合后才展示；增强纹理也可能让噪声更显眼 |
| Non-local means (NLM) | 按周围小图案相似程度加权平均 | 首轮传统降噪候选；弱参数起步，检查细节损失 |
| Bilateral filter | 同时考虑位置距离和灰度相似性 | 第二个传统候选；参数过强仍可能损害细节 |
| Noise2Void | 从单次含噪图像学习去噪，无需干净配对目标 | 后续阅读候选，不在这一个小 ROI 上训练后宣称有效 |

NLM 的 `h` 控制平滑强度；官方示例使用 `estimate_sigma` 给出初始量级，但它不是本数据真实噪声的证明。[NLM 官方示例](https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_nonlocal_means.html)

`estimate_sigma` 是针对高斯噪声的估计器；bilateral 同时按空间和灰度相似性加权。EM 细纹理可能影响估计，不能把图像整体标准差直接当作噪声标准差。[restoration 官方文档](https://scikit-image.org/docs/0.25.x/api/skimage.restoration.html)

CLAHE 属于对比度处理，`kernel_size`、`clip_limit` 等参数需要明确记录。[exposure 官方文档](https://scikit-image.org/docs/0.25.x/api/skimage.exposure.html#skimage.exposure.equalize_adapthist)

Noise2Void 的原论文提出从单次含噪观测训练，不要求干净目标。是否适合这里的噪声与结构，需要额外验证，不能由“无需标签”推出“没有假设或风险”。[Krull et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.html)

## 初始实验方案（执行时的参数调整见后文）

1. 先做逐层 2D 比较，避免把 Z 向 40 nm 与 XY 向 8 nm 当成各向同性网格。
2. 使用同一原图分别生成 Raw、CLAHE、弱 NLM；不先串联处理。
3. 从完整 sample XY 切片处理后再裁出 ROI，减少仅裁小块处理带来的边界效应。不要先把 mask 外像素清零，这会人为引入边界。
4. 原图转浮点时固定除以 255；输出保留浮点值。原始 TIFF 与 mask 不改动。
5. 初始探索参数：NLM `patch_size=5`、`patch_distance=6`、`fast_mode=True`、`channel_axis=None`，`h` 比较 `0.4` 与 `0.6` 倍初步 sigma 估计；这不是 MICrONS 推荐参数或已优化参数。
6. sigma 可先按完整来源切片估计并记录各层值，再以这 11 层的中位数作为本次固定强度，避免逐层参数波动。该估计需要和组织纹理一起审查。
7. CLAHE 初始试验固定 `kernel_size=(64,64)`、`clip_limit=0.01`、`nbins=256`；单独显示结果，不能把对比度改变称作去噪。
8. 每个变体记录函数、全部参数、依赖版本、输入哈希、坐标和时间。比较图使用一致显示窗口，不为每张图单独自动拉伸。

## 怎样评估而不夸大结论

- 在 Z=46、47、48、49 及完整 11 层中观察：细边界是否断裂/消失，内部纹理是否被抹平，相邻层变化是否仍合理。
- 看 `raw - denoised` 残差：如果出现清晰的结构轮廓，提示真实结构可能被去掉；残差看似随机也不等于已经证明处理正确。
- 单独检查目标 astrocyte 内部，但保留附近组织作为上下文。mask 不是线粒体标注。
- 当前没有干净 EM 真值，因此不要把相对原始含噪图像的 PSNR/SSIM 当成降噪质量排名。
- 当前没有线粒体真值或固定分割模型，因此不能报告 mitochondria segmentation 的 Dice/IoU 改善。
- 这是一个用户选取、靠近边界的小 ROI，只用于探索。若要推广，应增加其他位置及 sample_02/03，并将参数选择区域与最终评估区域分开。
- 需要专家确认结构身份或独立线粒体标注后，才能开展可靠的生物学定量评价。

不需要为本轮安装新包或下载更多数据。深度学习方法暂不进入实施阶段。

## 本轮实际执行与初步观察

原计划调用 `estimate_sigma`，但运行发现当前环境没有其可选依赖 PyWavelets。
本轮没有安装包，也没有以别的统计量假冒噪声估计；改为固定探索参数：

- NLM：`h=2/255` 和 `h=3/255`；`sigma=0`（不进行噪声方差校正）。其他参数仍为 `patch_size=5`、`patch_distance=6`、`fast_mode=True`、`channel_axis=None`、`preserve_range=True`。
- CLAHE：`kernel_size=(64,64)`、`clip_limit=0.01`、`nbins=256`。
- 每个源 sample 的完整 512×512 XY 切片先处理，再取出已固定的 192×192 ROI；共 Z=42–52。
- 所有处理分支独立从原图输入；没有串联 CLAHE 与 NLM，也没有遮掉 mask 外组织。
- ROI 到达 sample 的 Y 边界，因此完整切片处理仍不能消除该处的边界填充效应。

复现命令（输出目录必须不存在）：

```bash
python -B scripts/compare_preprocessing.py \
  --roi-dir samples/astro_864691136057675736_run1/preprocessing/roi_01 \
  --output-dir samples/astro_864691136057675736_run1/preprocessing/comparison_01
```

输出目录含 `index.html` 离线比较页、11 张 `comparison_z*.png`、`nlm_detail.png`、`residual.png`、四组 float32 TIFF（包括 raw/255）、两组有符号残差 TIFF，以及 `report.json` 参数/版本/哈希记录。原始 uint8 EM 保留在 source 和 roi_01 中。

浏览页顶部可切换 11 层；下方细节和残差图固定在用户所选 Z=47。主对比图统一显示 0–1。额外 NLM 细节图使用全部原始 ROI 的 1%–99% 分位数作为所有面板共同窗口，只改变显示。

| 方法 | 相对原图的平均绝对变化（0–255 单位） |
|---|---:|
| CLAHE | 31.723 |
| NLM h=2/255 | 0.858 |
| NLM h=3/255 | 1.480 |

这些是变化量，不是降噪质量分数，也不用于在 CLAHE 与 NLM 之间排名。

Z=47 的视觉检查：CLAHE 增强局部明暗差异；NLM 两档均削弱细颗粒纹理，h=3 更明显。暂不能断言被削弱的纹理全是噪声。保留两档作为探索对照，不选择“最佳方法”，也不将当前参数推广到整个细胞。

验证：来源 EM/mask/metadata 与固定 ROI 的三个文件执行前后哈希不变；四组处理 TIFF 和两组残差 TIFF 写出后逐像素读回一致；形状与有效数值范围通过检查。后续需要人工检查内部细节及边界，再在其他独立 ROI 上验证。

## sample_02 与线性显示对照

用户对第一轮的反馈：NLM 两档肉眼差别不大；CLAHE 让细节与颗粒感都更突出；NLM 仍能辨认部分边界，但内部纹理不如 CLAHE 显眼。这是视觉观察，颗粒感尚未确认为噪声。

随后用户在 sample_02 选择局部 XYZ `(137,426,105)`。已确认该点在目标 mask 内。新增 roi_02：局部边界 `[41,320,100]` 到 `[233,512,111]`（右端不含），ZYX `(11,192,192)`，目标占比约 25.51%。未确认任何线粒体身份。

沿用相同 CLAHE/NLM 参数，新增第二列“原图仅线性调整显示”：采用本 ROI 非恒定切片合并后的 1%–99% 原图分位数，在所有 Z 层固定；不降噪、不改 TIFF。范围外的值显示为黑/白。不同 ROI 使用同样窗口选择规则，但数值窗口不同，不能当作跨 ROI 的绝对亮度比较。

**QC 发现：sample_02 原始完整 512×512 的 Z=108 层全为 0。** 原因未确认，不能当作普通低对比度或噪声图像。该层保留原样、不处理、不插值，排除在窗口估算和变化统计之外。最初生成的 `comparison_02/` 未做这项排除，只作为先前运行留存；请使用带提醒的 `comparison_02_qc/index.html`。异常层仍可在页面查看，但不是有效方法比较样本。

当前查看入口（相对于 `samples/astro_864691136057675736_run1/preprocessing/`）：

- sample_02：`comparison_02_qc/index.html`，五列，默认 Z=105。
- sample_01：`comparison_01_linear/index.html`，补上线性对照，默认 Z=47。已验证四组处理 TIFF 与原 `comparison_01/` 逐像素相同。

sample_02 常量层以外的平均绝对变化（0–255 单位）：CLAHE 30.039、NLM h=2/255 为 0.918、NLM h=3/255 为 1.522。仍不是质量分数。浏览页下方的 NLM 共同窗口细节图和残差图固定在用户选中的 Z=105，不随顶部滑块变化。

用户决定：sample_02 的 Z=108 仅标记异常并排除，不再重新下载追查。原因继续记为未知，不补值、不覆盖原始数据。

## sample_03 固定区域与比较

用户补充截图提供局部 XYZ `(232,440,18)`，已确认该点在目标 astrocyte mask 内。新增 `roi_03/`：局部边界 `[136,320,13]` 到 `[328,512,24]`（右端不含），ZYX `(11,192,192)`，目标占比约 65.33%。Y 框平移至 sample 边界以保持尺寸，仍需留意边界处理效应。

入口：`samples/astro_864691136057675736_run1/preprocessing/comparison_03/index.html`，默认 Z=18，可切换 Z=13–23。五列为原图、原图线性显示、CLAHE、NLM h=2/255、NLM h=3/255；下方细节图与残差图固定 Z=18。

沿用 sample_02 的全部处理参数和线性窗口选择规则。检查的 11 个 ROI 切片未发现恒定灰度层，不代表排除了所有图像异常。原始及 ROI 来源哈希保持不变，处理 TIFF 读回一致，残差恒等式和参数一致性核验通过。

本区域平均绝对变化（0–255 单位）：CLAHE 26.636、NLM h=2/255 为 0.883、NLM h=3/255 为 1.558；这些仍然不是质量分数。三块数据均来自同一个 astrocyte，不构成跨细胞验证。sample_03 的人工比较反馈尚待记录，没有确定最佳方法或线粒体身份。

## Phase 2：评价区域候选（待确认）

新脚本 `scripts/build_evaluation_candidates.py` 从原始 ROI 构建离线审阅页，不执行降噪。
当前入口：`samples/astro_864691136057675736_run1/preprocessing/evaluation_candidates_02/index.html`。
`evaluation_candidates_01/` 为选点调整前的初稿，请使用 02。

每个 sample 提供边界、内部纹理和相对均匀候选，共 9 个；所有状态为 pending。
边界/纹理位置是原图视觉提议，不是经确认的线粒体。均匀候选为中心层目标 mask
占比至少 99% 的 32×32 窗口中，以 4 像素步长搜索最低灰度标准差的区域；
这个选择会偏向低变化区域，不能把其标准差当作无偏噪声估计，也可能选中细胞器内部。
候选可能在同一 sample 内重叠，不作为独立重复。sample_01 计划调参，02/03 预留评价，均待确认。

每项显示中心层及前后各一层、黄色候选框、边界项的拟议青色剖面线，可切换红色 astrocyte mask。
源 sample Z 分别为 46–48、104–106、17–19；不含异常 Z=108。
每个 ROI 的显示窗口由非恒定层的原始强度 1%–99% 分位数固定确定；只改变预览。
审阅选项为待确认/保留/排除/调整，备注和选项需点击导出 JSON 保存，不会自动持久化。

`candidates.json` 记录 source-local 坐标、全局 nm 中心坐标、框范围、mask 占比、显示窗口和来源哈希。
生成过程核验 9 个来源文件哈希不变、候选范围有效、切片非恒定，生成 54 张预览。
尚未运行新方法或取得任何评价结果；浏览器交互仍需人工确认。

## Phase 2：sample_01 固定区域降噪试验（2026-10-18）

后续人工反馈：Z47 边界 h2 比 h3 清楚、c4 比 c2 清楚、h2 与 c4 接近；内部纹理 c2 比 h2 多；uniform 中 h2 更均匀但差距不大，最初综合偏好 c2。Z48 认为 h2 平滑更明显，h2/c2 残差差别不大；Z46 也差不多。保留 h2/c2 两个候选，未确定最佳方法。

已固定全部参数运行 sample_02 Z104–106 和 sample_03 Z17–19 的 Raw/h2/c2 三列比较。入口为同一 preprocessing 目录下 evaluation_sample02_01/index.html 和 evaluation_sample03_01/index.html。sample_02 Z108 排除。脚本支持 --sample 参数及水平/垂直剖面。每个目录含输出 TIFF、残差、剖面、diagnostics.csv、profiles.csv、report.json。输入哈希与 TIFF 读回验证通过。

中心层 uniform 标准差（0–255）：sample02 Z105 为 Raw 4.436、h2 3.713、c2 3.974；sample03 Z18 为 Raw 4.164、h2 3.342、c2 3.537。仅表示局部波动，不是噪声估计或质量排名。显示窗口与残差色标在每个样本内部固定，跨样本可能不同。两块区域此前已被视觉探索，且来自同一细胞，因此是固定参数的区域复核，不是盲测或独立生物学验证。新结果人工审阅待进行，结构身份及剖面适用性仍未确认。

在用户同意开始后，将 sample_01 的三个候选位置固定用于探索性调参；这不代表已确认细胞器身份。原候选 manifest 保留 pending 历史状态，本次选择与授权记录在运行的 report.json。sample_02/03 未参与本次调参，之后只能作为同一细胞内的预留区域检查，不能称为独立生物学验证。

脚本：`scripts/run_denoising_tuning.py`。
结果入口：`samples/astro_864691136057675736_run1/preprocessing/tuning_sample01_01/index.html`。

固定 Z=46、47、48，在完整 512×512 原始切片上分别处理后裁出 ROI，不先将 mask 外清零。输入为 float32 /255，五组为 Raw、NLM h=2/255、NLM h=3/255、Bilateral sigma_color=2/255、Bilateral sigma_color=4/255。NLM 使用 patch_size=5、patch_distance=6、fast_mode=True、sigma=0；Bilateral 使用 sigma_spatial=1 px、win_size=7、bins=10000、reflect 边界。两类算法的参数不是等效强度，当前不是最优参数搜索。

所有方法与切片共用原始强度 112–160 的显示窗口；本轮不加 CLAHE。输出包含五组 TIFF、四组有符号残差 TIFF、三个区域的局部比较图、共同色标残差图、边界灰度剖面、diagnostics.csv、profiles.csv 和包含参数及来源哈希的 report.json。

中心层 Z=47 的相对均匀区域灰度标准差（0–255 单位）：Raw 3.335、NLM h2 2.313、NLM h3 1.798、Bilateral c2 2.667、Bilateral c4 2.336。这只表示局部灰度波动下降，不是噪声标准差或 SNR，不用于直接排名。该区域按低标准差筛选，存在选择偏差。尚无干净参考图或已确认线粒体标签，不能证明真实细节全部保留，也未选择最佳方法。

验证：来源与 ROI 文件哈希未改变；输出 TIFF 读回一致；形状、有限数值和范围检查通过；页面图片链接存在，45 行诊断记录已核验。浏览器人工评价待进行。代码与文档尚未提交或推送，结果图保存在本机。

复现命令（输出目录必须不存在）：

```bash
MPLCONFIGDIR=/tmp/microns-mpl /opt/miniconda3/envs/microns-astrocyte-toolkit/bin/python -B scripts/run_denoising_tuning.py --manifest samples/astro_864691136057675736_run1/preprocessing/evaluation_candidates_02/candidates.json --output-dir samples/astro_864691136057675736_run1/preprocessing/tuning_sample01_01
```

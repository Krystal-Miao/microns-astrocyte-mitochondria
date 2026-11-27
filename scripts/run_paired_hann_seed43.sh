#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python=.venv-unet/bin/python
roots=(samples/astro_multicell_mip2_pilot_01 samples/astro_multicell_mip2_remaining_01)
prior=samples/hann_seed42_pr_01/summary.json

if [[ ! -s "$prior" ]]; then
    echo "Generate the seed 42 Hann validation PR summary first." >&2
    exit 1
fi

evaluate() {
    "$python" -u scripts/evaluate_multicell3d_whole_sections.py \
        --roots "${roots[@]}" --stride-xy 64 --blend hann --sections 32 64 96 "$@"
}

for variant in raw nlm; do
    if [[ "$variant" == raw ]]; then
        run=samples/unet_multicell_controlled_random_nnstyle_seed42_3000_01
    else
        run=samples/unet_mip2nlm_controlled_random_seed42_3000_01
    fi
    evaluate --run "$run" --split test --threshold-from "$prior" \
        --out "samples/blend_${variant}_hann_seed42_test_r60_01"
done

for variant in raw nlm; do
    run="samples/unet_mip2${variant}_controlled_random_seed43_3000_01"
    "$python" -u scripts/train_multicell3d_controlled_random.py \
        --roots "${roots[@]}" --out "$run" --steps 3000 --seed 43 \
        --eval-every 250 --preprocessing "$variant"
    evaluate --run "$run" --split validation --save-pr-scores \
        --out "samples/blend_${variant}_hann_seed43_val_pr_01"
done

report=samples/hann_seed43_pr_01
"$python" scripts/report_fullxy_pr.py --hann-seed 43 --out "$report"
for variant in raw nlm; do
    evaluate --run "samples/unet_mip2${variant}_controlled_random_seed43_3000_01" \
        --split test --threshold-from "$report/summary.json" \
        --out "samples/blend_${variant}_hann_seed43_test_r60_01"
done

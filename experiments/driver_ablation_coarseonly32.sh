#!/bin/bash
# B2 half1 coarse-only with a SMALLER coarse crop (radius 32 -> 64px) vs coarseonly (radius 48 -> 96px).
# Tests how small the low-res pose input can go before accuracy drops. Compare time + pose(stacked) to:
#   norefine 100min/50%<5, localrefine 87min/46%, coarseonly(r48) 78min/50%.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half1.star; NPZ=results/2026_Jul_06/ablation_npz; mkdir -p $NPZ
echo "===== B2 coarse-only radius 32 start ====="
t0=$(date +%s)
bash run_ablation_train.sh coarseonly32 > LOG/ablation_coarseonly32.log 2>&1
t1=$(date +%s); secs=$((t1-t0))
echo "TRAIN_SECONDS coarseonly32 ${secs}" | tee -a LOG/ablation_times.log
echo "  train coarseonly32 done in $((secs/60)) min"
rm -f results/2026_Jul_06/B2_ablation_coarseonly32_poses/*.star 2>/dev/null
bash run_ablation_poseinfer.sh coarseonly32 > LOG/ablation_coarseonly32_poses.log 2>&1
python3 pose_bestof8.py "results/2026_Jul_06/B2_ablation_coarseonly32_poses/temp_pose_infer_rank*.star" \
   ${REF} ${NPZ}/ablation_coarseonly32.npz > LOG/ablation_acc_coarseonly32.log 2>&1
echo "  ABLATION coarseonly32 accuracy: $(grep -a OVERALL LOG/ablation_acc_coarseonly32.log | tail -1)"
echo COARSEONLY32_DONE

#!/bin/bash
# B2 half1 coarse-only run: coarse encoder predicts pose from the 96px downsampled input for ALL epochs,
# local full-res encoder NEVER runs (marching still drives the decoder/loss). Compare time + pose vs
# norefine (99min, 50% <5deg) and localrefine (85min, 46%). Target: accuracy held, bigger time cut.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half1.star; NPZ=results/2026_Jul_06/ablation_npz; mkdir -p $NPZ
echo "===== B2 coarse-only start ====="
t0=$(date +%s)
bash run_ablation_train.sh coarseonly > LOG/ablation_coarseonly.log 2>&1
t1=$(date +%s); secs=$((t1-t0))
echo "TRAIN_SECONDS coarseonly ${secs}" | tee -a LOG/ablation_times.log
echo "  train coarseonly done in $((secs/60)) min"
rm -f results/2026_Jul_06/B2_ablation_coarseonly_poses/*.star 2>/dev/null
bash run_ablation_poseinfer.sh coarseonly > LOG/ablation_coarseonly_poses.log 2>&1
python3 pose_bestof8.py "results/2026_Jul_06/B2_ablation_coarseonly_poses/temp_pose_infer_rank*.star" \
   ${REF} ${NPZ}/ablation_coarseonly.npz > LOG/ablation_acc_coarseonly.log 2>&1
echo "  ABLATION coarseonly accuracy: $(grep -a OVERALL LOG/ablation_acc_coarseonly.log | tail -1)"
echo COARSEONLY_DONE

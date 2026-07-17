#!/bin/bash
# (3) end_resolution 5 experiment on B2 half1, coarse-only. Q: can we still get accurate poses if the
# decoder is only ever fit to 5A (vs 3.5A)? And it runs shorter: mask_steps 22 (not 32) -> marching to 5A
# completes ~ep66, so num-epochs 68 (~24% fewer epochs). Compare pose(stacked)+time to coarseonly(3.5A):
#   coarseonly 78min/50.1%<5, norefine 100min/50.1%.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half1.star; NPZ=results/2026_Jul_06/ablation_npz; mkdir -p $NPZ
export NEPOCH=68 MSTEPS=22 ENDRES=5.0 DISTEP=44 UNIEP=68 SUFFIX=_end5
echo "===== B2 coarse-only end_res 5 start (mask_steps 22, 68 ep) ====="
t0=$(date +%s)
bash run_ablation_train.sh coarseonly > LOG/ablation_coarseonly_end5.log 2>&1
t1=$(date +%s); secs=$((t1-t0))
echo "TRAIN_SECONDS coarseonly_end5 ${secs}" | tee -a LOG/ablation_times.log
echo "  train coarseonly_end5 done in $((secs/60)) min"
rm -f results/2026_Jul_06/B2_ablation_coarseonly_end5_poses/*.star 2>/dev/null
bash run_ablation_poseinfer.sh coarseonly > LOG/ablation_coarseonly_end5_poses.log 2>&1
python3 pose_bestof8.py "results/2026_Jul_06/B2_ablation_coarseonly_end5_poses/temp_pose_infer_rank*.star" \
   ${REF} ${NPZ}/ablation_coarseonly_end5.npz > LOG/ablation_acc_coarseonly_end5.log 2>&1
echo "  ABLATION coarseonly_end5 accuracy: $(grep -a OVERALL LOG/ablation_acc_coarseonly_end5.log | tail -1)"
echo COARSEONLY_END5_DONE

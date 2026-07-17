#!/bin/bash
# Continue B2 half1 norefine from the ep90 checkpoint, pushing end_resolution 3.5A -> 3.0A over 30 more
# epochs (ep90->120). Q: does more training + higher-res marching solve MORE particles (>50% <5deg)?
# Schedule chosen for a SMOOTH continuation: mask_steps 40 (mask_epoch 3) -> frontier at ep90 ~= 3.9A
# (≈ the checkpoint's ~3.83A, no forward jump), reaching 3.0A at ep120. Resume restores the epoch counter.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; REF=/data/cryoFIFA_data/B2_act_half1.star; NPZ=results/${Date}/ablation_npz
CK=results/${Date}/B2_ablation_norefine/checkpoint.pt
[ -f "$CK" ] || { echo "MISSING ep90 checkpoint $CK"; exit 2; }
echo "===== continue B2 half1: end_res 3.0, ep90->120 (resume $CK) ====="
t0=$(date +%s)
RESUME=$CK SUFFIX=_end3 HALF=half1 NEPOCH=120 MSTEPS=40 ENDRES=3.0 DISTEP=60 UNIEP=120 \
   bash run_ablation_train.sh norefine > LOG/b2_end3_train.log 2>&1
echo "  train done in $(( ($(date +%s)-t0)/60 )) min; last: $(grep -aoE '\[[0-9]+/120\]' LOG/b2_end3_train.log|tail -1)"
rm -f results/${Date}/B2_ablation_norefine_end3_poses/*.star 2>/dev/null
SUFFIX=_end3 HALF=half1 MSTEPS=40 ENDRES=3.0 bash run_ablation_poseinfer.sh norefine > LOG/b2_end3_poses.log 2>&1
python3 pose_bestof8.py "results/${Date}/B2_ablation_norefine_end3_poses/temp_pose_infer_rank*.star" \
   ${REF} ${NPZ}/ablation_norefine_end3.npz > LOG/b2_end3_acc.log 2>&1
echo "  END3 accuracy: $(grep -a OVERALL LOG/b2_end3_acc.log|tail -1)"
echo "  (baseline ep90 3.5A: <5deg=50.1%)"
# export the new <5deg subset for reconstruction comparison
python3 export_solved_star.py "results/${Date}/B2_ablation_norefine_end3_poses/temp_pose_infer_rank*.star" \
   ${REF} results/${Date}/B2_solved_lt5_norefine_end3.star 5 > LOG/b2_end3_export.log 2>&1
grep -a wrote LOG/b2_end3_export.log
echo B2_END3_DONE

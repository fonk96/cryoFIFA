#!/bin/bash
# Cross-half generalization driver: half1 checkpoint -> predict half2 poses -> (a) pose error vs the
# half2 cryoSPARC consensus, (b) export a cryoSPARC-importable star for reconstruct-only.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA
cd $ROOT/experiments
mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half2.star
POSES="results/2026_Jul_06/B2_crosshalf_h1model_h2data_poses/temp_pose_infer_rank*.star"
echo "===== CROSSHALF (h1 model -> h2 data) start ====="
rm -f results/2026_Jul_06/B2_crosshalf_h1model_h2data_poses/*.star 2>/dev/null
t0=$(date +%s)
bash run_crosshalf_poseinfer.sh > LOG/crosshalf_poses.log 2>&1
t1=$(date +%s)
echo "  pose_infer done in $(( (t1-t0)/60 )) min"
# (a) generalization metric: predicted h2 poses vs h2 cryoSPARC consensus
python3 pose_bestof8.py "$POSES" ${REF} /tmp/crosshalf.npz > LOG/crosshalf_acc.log 2>&1
echo "  CROSSHALF accuracy (h1 model on unseen h2): $(grep -a OVERALL LOG/crosshalf_acc.log | tail -1)"
# (b) cryoSPARC-importable star for reconstruct-only (zero shift = best in prior B2 test; fit as backup)
python3 export_star_for_cryosparc.py ${REF} results/2026_Jul_06/crosshalf_h2pred_noshift.star  zero "$POSES" > LOG/crosshalf_export_zero.log 2>&1
python3 export_star_for_cryosparc.py ${REF} results/2026_Jul_06/crosshalf_h2pred_fitshift.star fit  "$POSES" > LOG/crosshalf_export_fit.log 2>&1
echo "  exported: results/2026_Jul_06/crosshalf_h2pred_{noshift,fitshift}.star"
echo "CROSSHALF_DONE"

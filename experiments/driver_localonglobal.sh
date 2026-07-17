#!/bin/bash
# Local-on-global driver: resume global-only checkpoint + add local module (enlarged epoch + end res) ->
# train (timed) -> best-of-8 pose_infer -> pose accuracy vs the half1 cryoSPARC consensus. Baseline to beat:
# global-only (norefine) = median 4.99 deg @ 3.8A.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA
cd $ROOT/experiments
mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half1.star
echo "===== LOCAL-ON-GLOBAL start ====="
t0=$(date +%s)
bash run_localonglobal_train.sh > LOG/localonglobal.log 2>&1
t1=$(date +%s)
echo "  train done in $(( (t1-t0)/60 )) min"
rm -f results/2026_Jul_06/B2_localonglobal_end25_poses/*.star 2>/dev/null
bash run_localonglobal_poseinfer.sh > LOG/localonglobal_poses.log 2>&1
echo "  pose_infer done"
python3 pose_bestof8.py "results/2026_Jul_06/B2_localonglobal_end25_poses/temp_pose_infer_rank*.star" \
   ${REF} /tmp/localonglobal.npz > LOG/localonglobal_acc.log 2>&1
echo "  LOCAL-ON-GLOBAL accuracy (vs half1 cryoSPARC): $(grep -a OVERALL LOG/localonglobal_acc.log | tail -1)"
echo "  (baseline global-only norefine = median 4.99 deg)"
echo "LOCALONGLOBAL_DONE"

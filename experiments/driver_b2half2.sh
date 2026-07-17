#!/bin/bash
# Mirror the half1 norefine pipeline on B2_half2, then export its <5deg subset star. Pairs with the
# half1 subset (25061 particles -> user got 3.17A) for a GOLD-STANDARD FSC: fully independent half
# (disjoint particles, own model, own poses, own cryoSPARC-consensus scoring).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; NPZ=results/${Date}/ablation_npz; mkdir -p $NPZ
REF=/data/cryoFIFA_data/B2_act_half2.star
if [ ! -d /data/lmdb/B2_act_half2 ]; then
  echo "===== building lmdb B2_act_half2 ====="
  PYTHONPATH=../src python3 ../tests/test_newloader.py --particles_meta ${REF} \
     --data_path /data/cryoFIFA_data/ --db_path /data/lmdb/B2_act_half2 --imgsize 228 > LOG/b2half2_lmdb.log 2>&1
  echo "  lmdb build exit=$?"
fi
echo "===== train norefine on B2_half2 (90 ep, box 228, end 3.5A) ====="
t0=$(date +%s)
HALF=half2 SUFFIX=_half2 bash run_ablation_train.sh norefine > LOG/b2half2_norefine.log 2>&1
t1=$(date +%s); echo "TRAIN_SECONDS norefine_half2 $((t1-t0))" | tee -a LOG/ablation_times.log
echo "  train done $((($t1-$t0)/60)) min"
rm -f results/${Date}/B2_ablation_norefine_half2_poses/*.star 2>/dev/null
HALF=half2 SUFFIX=_half2 bash run_ablation_poseinfer.sh norefine > LOG/b2half2_norefine_poses.log 2>&1
python3 pose_bestof8.py "results/${Date}/B2_ablation_norefine_half2_poses/temp_pose_infer_rank*.star" \
   ${REF} ${NPZ}/ablation_norefine_half2.npz > LOG/b2half2_acc.log 2>&1
echo "  ACC half2: $(grep -a OVERALL LOG/b2half2_acc.log|tail -1)"
for T in 5 3; do
  python3 export_solved_star.py "results/${Date}/B2_ablation_norefine_half2_poses/temp_pose_infer_rank*.star" \
     ${REF} results/${Date}/B2_solved_lt${T}_norefine_half2.star ${T} > LOG/b2half2_export${T}.log 2>&1
  echo "  <${T} star: $(grep -a wrote LOG/b2half2_export${T}.log|tail -1)"
done
echo B2HALF2_DONE

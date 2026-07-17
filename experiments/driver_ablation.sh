#!/bin/bash
# Local-refine ablation driver: for each arm (norefine, localrefine): train (timed) -> best-of-8
# pose_infer -> pose accuracy vs cryoSPARC consensus (B2_act_half1.star). Sequential on one 8-GPU node.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA
cd $ROOT/experiments
mkdir -p LOG
REF=/data/cryoFIFA_data/B2_act_half1.star
: > LOG/ablation_times.log
for ARM in norefine localrefine; do
  echo "===== ABLATION $ARM start ====="
  t0=$(date +%s)
  bash run_ablation_train.sh ${ARM} > LOG/ablation_${ARM}.log 2>&1
  t1=$(date +%s)
  secs=$((t1-t0))
  echo "TRAIN_SECONDS ${ARM} ${secs}" | tee -a LOG/ablation_times.log
  echo "  train $ARM done in $((secs/60)) min ${secs}s"
  # best-of-8 pose inference
  rm -f results/2026_Jul_06/B2_ablation_${ARM}_poses/*.star 2>/dev/null
  bash run_ablation_poseinfer.sh ${ARM} > LOG/ablation_${ARM}_poses.log 2>&1
  echo "  pose_infer $ARM done"
  # pose accuracy vs cryoSPARC consensus
  python3 pose_bestof8.py "results/2026_Jul_06/B2_ablation_${ARM}_poses/temp_pose_infer_rank*.star" \
     ${REF} /tmp/ablation_${ARM}.npz > LOG/ablation_acc_${ARM}.log 2>&1
  echo "  ABLATION ${ARM} accuracy: $(grep -a OVERALL LOG/ablation_acc_${ARM}.log | tail -1)"
  echo "===== ABLATION $ARM COMPLETE ====="
done
echo "ABLATION_ALL_DONE"

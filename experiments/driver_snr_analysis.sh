#!/bin/bash
# Waits for each SNR's best-of-8 poses, computes pose accuracy vs ground-truth -> /tmp/snr_TAG.npz.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments
for TAG in 005 01 02 03 04 05; do
  P=results/2026_Jul_06/ribo_snr${TAG}_poses/temp_pose_infer_rank0.star
  n=0
  while [ ! -f "$P" ] && [ $n -lt 400 ]; do sleep 45; n=$((n+1)); done
  if [ -f "$P" ]; then
    python3 pose_bestof8.py "results/2026_Jul_06/ribo_snr${TAG}_poses/temp_pose_infer_rank*.star" \
       /sddn/kongfang_work/cryoFIFA/data1/ribo_snr${TAG}.star /tmp/snr_${TAG}.npz > LOG/snr_acc_${TAG}.log 2>&1
    echo "SNR ${TAG} accuracy: $(grep -a OVERALL LOG/snr_acc_${TAG}.log | tail -1)"
  fi
done
echo ANALYSIS_ALL_DONE

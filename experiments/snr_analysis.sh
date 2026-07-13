#!/bin/bash
# Pose accuracy vs ground-truth (simulation) per SNR: cryoFIFA best-of-8 poses vs ribo_snrTAG.star.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments
for TAG in 005 01 02 03 04 05; do
  P=$(ls results/2026_Jul_06/ribo_snr${TAG}_poses/temp_pose_infer_rank0.star 2>/dev/null)
  GT=/sddn/kongfang_work/cryoFIFA/data1/ribo_snr${TAG}.star
  if [ -n "$P" ]; then
    echo "===== SNR $TAG (vs ground truth) ====="
    python3 pose_bestof8.py "results/2026_Jul_06/ribo_snr${TAG}_poses/temp_pose_infer_rank*.star" "$GT" /tmp/snr_${TAG}.npz 2>&1 | grep -aE 'matched|convention|OVERALL'
  else
    echo "SNR $TAG: poses not ready yet"
  fi
done

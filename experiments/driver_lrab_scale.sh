#!/bin/bash
# Part C: local_refine_scale sweep on ribosome sim (box 128, SNR 0.5). Fills the low end of the curve
# (0 = coarseonly done, 0.3 = coarselocal done). Tests whether a smaller scale avoids the degradation.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
echo "===== Part C local_refine_scale sweep start ====="
for ARM in scale0.05 scale0.1; do
  bash run_lrablate_ribo.sh $ARM > LOG/lrab_ribo_${ARM}.log 2>&1
  T=$(grep -aoE "\[69/70\][^#]*time: [0-9.]+ s" LOG/lrab_ribo_${ARM}.log | grep -oE "time: [0-9.]+ s" | tail -1)
  echo "LRABSCALE trained ${ARM} : ${T}"
done
echo LRABSCALE_DONE

#!/bin/bash
# Part A local-refine ablation on ribosome sim (box 128, SNR 0.5): train standard / coarseonly / coarselocal.
# Resolution (FSC-to-GT) + pose accuracy are scored offline after training; here we train + record wall-clock.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
echo "===== Part A local-refine ablation (ribosome SNR 0.5) start ====="
for ARM in standard coarseonly coarselocal; do
  bash run_lrablate_ribo.sh $ARM > LOG/lrab_ribo_${ARM}.log 2>&1
  T=$(grep -aoE "\[69/70\][^#]*time: [0-9.]+ s" LOG/lrab_ribo_${ARM}.log | grep -oE "time: [0-9.]+ s" | tail -1)
  echo "LRAB trained ${ARM} : ${T}"
done
echo LRAB_ALL_DONE

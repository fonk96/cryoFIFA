#!/bin/bash
# Wait for half2 to finish (frees ibsgpu11 + its B2 data), then run loss-export for BOTH halves:
# pose-vs-loss plot + _cryofifaLoss-annotated star (global GT-free filter) + lowest-loss-50% subset.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
echo "chain: waiting for B2HALF2_DONE ..."
for i in $(seq 1 240); do grep -q B2HALF2_DONE LOG/driver_b2half2.log 2>/dev/null && break; sleep 60; done
# ensure the training/poseinfer procs are actually gone before grabbing all 8 GPUs
for i in $(seq 1 30); do [ "$(pgrep -f cryofifa_ddpfix.py|wc -l)" -le 1 ] && break; sleep 20; done
echo "chain: half2 done, starting loss export $(date)"
bash driver_b2_lossexport.sh half1
bash driver_b2_lossexport.sh half2
echo B2_LOSSEXPORT_ALL_DONE

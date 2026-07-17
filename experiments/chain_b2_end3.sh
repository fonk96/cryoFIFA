#!/bin/bash
# Wait for b2_traj to free ibsgpu11, then run the end_res-3 continuation (ep90->120).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
echo "chain_end3: waiting for B2_TRAJ_DONE..."
for i in $(seq 1 260); do grep -q B2_TRAJ_DONE LOG/driver_b2_traj.log 2>/dev/null && break; sleep 60; done
for i in $(seq 1 30); do [ "$(pgrep -f cryofifa_ddpfix.py|wc -l)" -le 1 ] && break; sleep 20; done
echo "chain_end3: node free, starting end3 $(date)"
bash driver_b2_end3.sh
echo CHAIN_END3_ALL_DONE

#!/bin/bash
# After end3 frees ibsgpu11, re-run B2 half1 poseinfer (ORIGINAL ep90 ckpt) exporting the full 8 per-copy
# losses (full8_infer_rank*.txt) to test the handedness/symmetry signal.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
echo "chain_full8: waiting for B2_END3_DONE..."
for i in $(seq 1 120); do grep -q B2_END3_DONE LOG/driver_b2_end3.log 2>/dev/null && break; sleep 60; done
for i in $(seq 1 30); do [ "$(pgrep -f cryofifa_ddpfix.py|wc -l)" -le 1 ] && break; sleep 20; done
echo "chain_full8: node free, exporting full8 $(date)"
D=results/2026_Jul_06/B2_ablation_norefine_full8_poses
rm -f ${D}/*.txt ${D}/*.star 2>/dev/null
HALF=half1 SUFFIX="" OUTSUF=_full8 EXPORT_LOSS=1 EXPORT_CONF=1 bash run_ablation_poseinfer.sh norefine > LOG/b2_full8_infer.log 2>&1
echo "  full8_rows=$(cat ${D}/full8_infer_rank*.txt 2>/dev/null|wc -l) conf_rows=$(cat ${D}/conf_infer_rank*.txt 2>/dev/null|wc -l)"
echo "  full8 sample:"; head -1 ${D}/full8_infer_rank0.txt 2>/dev/null
echo B2_FULL8_DONE

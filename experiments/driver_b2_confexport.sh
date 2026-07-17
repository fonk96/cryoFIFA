#!/bin/bash
# Re-run B2 half1 poseinfer exporting per-particle best-of-8 CONFIDENCE (conf_infer_rank*.txt) + loss +
# poses. B2 has cryoSPARC GT -> we can VALIDATE which GT-free confidence metric predicts pose error
# (the goal: a filter usable AFTER ab-initio when no GT poses exist). Runs on ibsgpu11 (B2 data node).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06
D=results/${Date}/B2_ablation_norefine_confexp_poses
rm -f ${D}/*.star ${D}/*.txt 2>/dev/null
echo "===== B2 half1 poseinfer + conf_export ====="
HALF=half1 SUFFIX="" OUTSUF=_confexp EXPORT_LOSS=1 EXPORT_CONF=1 bash run_ablation_poseinfer.sh norefine > LOG/b2_confexp_infer.log 2>&1
echo "  poses=$(ls ${D}/temp_pose_infer_rank*.star 2>/dev/null|wc -l) conf_rows=$(cat ${D}/conf_infer_rank*.txt 2>/dev/null|wc -l) loss_rows=$(cat ${D}/loss_infer_rank*.txt 2>/dev/null|wc -l)"
echo "  conf sample:"; head -2 ${D}/conf_infer_rank0.txt 2>/dev/null
grep -aiE "Traceback|Error:|ChildFailedError" LOG/b2_confexp_infer.log 2>/dev/null | grep -v Error_ | head
echo B2_CONFEXP_DONE

#!/bin/bash
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06
D=results/${Date}/B2_ablation_norefine_half2_confexp_poses
rm -f ${D}/*.star ${D}/*.txt 2>/dev/null
echo "===== B2 half2 poseinfer + conf_export ====="
HALF=half2 SUFFIX=_half2 OUTSUF=_confexp EXPORT_LOSS=1 EXPORT_CONF=1 bash run_ablation_poseinfer.sh norefine > LOG/b2_confexp_half2_infer.log 2>&1
echo "  conf_rows=$(cat ${D}/conf_infer_rank*.txt 2>/dev/null|wc -l) poses=$(ls ${D}/temp_pose_infer_rank*.star 2>/dev/null|wc -l)"
echo B2_CONFEXP_HALF2_DONE

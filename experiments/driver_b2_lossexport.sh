#!/bin/bash
# For B2 <half>: re-run norefine poseinfer with --loss_export (per-particle loss + poses), then
#  (1) pose-vs-loss plot, (2) FULL star annotated with _cryofifaLoss + _cryofifaPoseErr for GLOBAL
#      GT-free loss filtering, + a lowest-loss-50% subset. Runs on a B2-data node (ibsgpu11).
# usage: bash driver_b2_lossexport.sh half1|half2
HALF=${1:-half1}
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG lrab_figures
Date=2026_Jul_06
SUF=""; [ "$HALF" = "half2" ] && SUF="_half2"
CS=/data/cryoFIFA_data/B2_act_${HALF}.star
D=results/${Date}/B2_ablation_norefine${SUF}_lossexp_poses
if [ ! -f results/${Date}/B2_ablation_norefine${SUF}/checkpoint.pt ]; then echo "MISSING ckpt for ${HALF}"; exit 2; fi
rm -f ${D}/*.star ${D}/*.txt 2>/dev/null
echo "===== ${HALF}: poseinfer norefine + loss_export ====="
HALF=${HALF} SUFFIX=${SUF} OUTSUF=_lossexp EXPORT_LOSS=1 bash run_ablation_poseinfer.sh norefine > LOG/b2_lossexp_${HALF}_infer.log 2>&1
echo "  poses=$(ls ${D}/temp_pose_infer_rank*.star 2>/dev/null|wc -l) loss_rows=$(cat ${D}/loss_infer_rank*.txt 2>/dev/null|wc -l)"
echo "===== ${HALF}: pose-vs-loss plot ====="
python3 plot_pose_vs_loss.py "${D}/temp_pose_infer_rank*.star" "${D}/loss_infer_rank*.txt" ${CS} \
   lrab_figures/b2_pose_vs_loss_${HALF} "B2 ${HALF} (norefine)" > LOG/b2_lossexp_${HALF}_plot.log 2>&1
grep -aE "matched|OVERALL|Spearman|median loss|keep lowest" LOG/b2_lossexp_${HALF}_plot.log
echo "===== ${HALF}: loss-annotated star (+ keep lowest-loss 50%) ====="
python3 export_star_withloss.py "${D}/temp_pose_infer_rank*.star" "${D}/loss_infer_rank*.txt" ${CS} \
   results/${Date}/B2_${HALF}_withloss.star 0.5 > LOG/b2_lossexp_${HALF}_star.log 2>&1
grep -aE "wrote|loss percentiles|keep lowest" LOG/b2_lossexp_${HALF}_star.log
echo "B2_LOSSEXPORT_${HALF}_DONE"

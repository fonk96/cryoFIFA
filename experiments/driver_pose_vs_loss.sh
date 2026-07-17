#!/bin/bash
# Relate per-particle LOSS to pose ERROR on B2 half1 (norefine). Re-run poseinfer with --loss_export to
# dump per-particle best-of-8 loss + poses, then join with pose error (vs cryoSPARC) and plot.
# Mechanistic basis for loss-ranking AND for "FIFA-easy (=low-loss) particles are better particles".
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; CS=/data/cryoFIFA_data/B2_act_half1.star
D=results/${Date}/B2_ablation_norefine_lossexp_poses
rm -f ${D}/*.star ${D}/*.txt 2>/dev/null
echo "===== poseinfer norefine half1 + loss_export ====="
HALF=half1 OUTSUF=_lossexp EXPORT_LOSS=1 bash run_ablation_poseinfer.sh norefine > LOG/pose_vs_loss_infer.log 2>&1
echo "  poses=$(ls ${D}/temp_pose_infer_rank*.star 2>/dev/null|wc -l)  loss_files=$(ls ${D}/loss_infer_rank*.txt 2>/dev/null|wc -l)  loss_rows=$(cat ${D}/loss_infer_rank*.txt 2>/dev/null|wc -l)"
python3 plot_pose_vs_loss.py "${D}/temp_pose_infer_rank*.star" "${D}/loss_infer_rank*.txt" ${CS} \
   lrab_figures/b2_pose_vs_loss "B2 half1 (norefine)" > LOG/pose_vs_loss_plot.log 2>&1
echo "----- plot log -----"; tail -8 LOG/pose_vs_loss_plot.log
echo POSE_VS_LOSS_DONE

#!/bin/bash
# B2 half1 norefine RE-RUN keeping a checkpoint at every interval (ep15/30/45/60/75/90), then export
# per-particle loss + best-of-8 confidence at EACH epoch -> loss/confidence TRAJECTORY. Tests whether
# how a particle's loss evolves over training separates well-solved (<5deg) from ill-solved better than
# a single final-epoch snapshot. seed 0 (deterministic -> same run as the original, just with the
# intermediate checkpoints kept). SUFFIX=_traj so it never clobbers the original B2_ablation_norefine.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; TD=results/${Date}/B2_ablation_norefine_traj
echo "===== B2 half1 re-run (keep_all) ====="
t0=$(date +%s)
KEEPALL=1 SUFFIX=_traj HALF=half1 bash run_ablation_train.sh norefine > LOG/b2_traj_train.log 2>&1
echo "  train done in $(( ($(date +%s)-t0)/60 )) min; checkpoints:"; ls ${TD}/checkpoint_ep*.pt 2>/dev/null | sed 's#.*/##' | tr '\n' ' '; echo
for EP in 15 30 45 60 75 90; do
  CK=${TD}/checkpoint_ep${EP}.pt
  [ -f "$CK" ] || { echo "  ep${EP}: NO ckpt, skip"; continue; }
  D=results/${Date}/B2_ablation_norefine_traj_ep${EP}_poses
  rm -f ${D}/*.txt ${D}/*.star 2>/dev/null
  CKPT_NAME=checkpoint_ep${EP}.pt OUTSUF=_ep${EP} SUFFIX=_traj HALF=half1 EXPORT_LOSS=1 EXPORT_CONF=1 \
     bash run_ablation_poseinfer.sh norefine > LOG/b2_traj_infer_ep${EP}.log 2>&1
  echo "  ep${EP}: loss_rows=$(cat ${D}/loss_infer_rank*.txt 2>/dev/null|wc -l) conf_rows=$(cat ${D}/conf_infer_rank*.txt 2>/dev/null|wc -l)"
done
echo B2_TRAJ_DONE

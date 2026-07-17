#!/bin/bash
# PART B FIX: does holding the loss-rank rejection (never dropping to keep-100%) rescue the mix?
# lrHold = keep 50% for ALL epochs (reject the junk half throughout, incl. the high-res marching phase).
# Compare against lrON (reject 50%->0% by ep40) and lrOFF (marchonly) already in driver_mix.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06
ARM=lrHold

echo "===== Part B FIX ($ARM = keep 50% throughout) start ====="
for MIX in 0103 0105; do
  case $MIX in 0103) HI=J530 ;; 0105) HI=J532 ;; esac
  GT=/sddn/kongfang_work/cryoFIFA/data1/ribo_mix${MIX}.star
  echo "----- MIX ${MIX} (high=${HI}) $ARM -----"
  bash run_mix_train.sh $ARM $MIX > LOG/mix${MIX}_${ARM}.log 2>&1
  echo "  trained mix${MIX} ${ARM}"
  rm -f results/${Date}/mix${MIX}_${ARM}_loss_ep70/temp_pose_infer_rank*.star results/${Date}/mix${MIX}_${ARM}_loss_ep70/loss_infer_rank*.txt 2>/dev/null
  bash run_mix_lossexport.sh $ARM 70 $MIX > LOG/mix${MIX}_loss_${ARM}_ep70.log 2>&1
  echo "  export mix${MIX} ${ARM} ep70"
  # overall accuracy (validated tool) + split-by-SNR
  python3 pose_bestof8.py "results/${Date}/mix${MIX}_${ARM}_loss_ep70/temp_pose_infer_rank*.star" $GT /tmp/mixfix_${MIX}_${ARM}.npz \
    > LOG/mix${MIX}_acc_${ARM}.log 2>&1
  python3 analyze_mix.py  "results/${Date}/mix${MIX}_${ARM}_loss_ep70/temp_pose_infer_rank*.star" $GT $HI \
    > LOG/mix${MIX}_accsplit_${ARM}.log 2>&1
  echo "  MIXFIX ${MIX} ${ARM} OVERALL:  $(grep -a OVERALL LOG/mix${MIX}_acc_${ARM}.log | tail -1)"
  echo "  MIXFIX ${MIX} ${ARM} SPLIT:"
  grep -aE "ALL |HIGH |LOW " LOG/mix${MIX}_accsplit_${ARM}.log | sed 's/^/    /'
done
echo MIXFIX_ALL_DONE

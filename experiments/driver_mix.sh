#!/bin/bash
# PART B (reviewer R1.4/R2.7): does loss-ranking earn its keep on MIXED-SNR data?
# marching held ON in both arms; loss-rank is the only toggle.
#   lrOFF = marchonly (baseline)   lrON = both (march + --hard_topk)
# Two mixes: 0103 = 20k SNR0.1(J528)+20k SNR0.3(J530); 0105 = 20k SNR0.1(J528)+20k SNR0.5(J532).
# Hypothesis: on a mix, lrON >= lrOFF on the HIGH-SNR subset (loss-rank discards SNR0.1 junk),
# reversing the uniform-SNR result (there lrON < lrOFF, no junk to discard).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06

echo "===== Part B (mixture) start ====="
for MIX in 0103 0105; do
  case $MIX in 0103) HI=J530 ;; 0105) HI=J532 ;; esac
  GT=/sddn/kongfang_work/cryoFIFA/data1/ribo_mix${MIX}.star
  echo "----- MIX ${MIX} (high=${HI}) -----"
  # 1) train both arms (marching ON in both)
  for ARM in lrOFF lrON; do
    bash run_mix_train.sh $ARM $MIX > LOG/mix${MIX}_${ARM}.log 2>&1
    echo "  trained mix${MIX} ${ARM}"
  done
  # 2) loss+pose export: lrON every 10 epochs (enrichment), lrOFF ep70 (accuracy poses)
  for EP in 10 20 30 40 50 60 70; do
    rm -f results/${Date}/mix${MIX}_lrON_loss_ep${EP}/loss_infer_rank*.txt results/${Date}/mix${MIX}_lrON_loss_ep${EP}/temp_pose_infer_rank*.star 2>/dev/null
    bash run_mix_lossexport.sh lrON $EP $MIX > LOG/mix${MIX}_loss_lrON_ep${EP}.log 2>&1
    echo "  export mix${MIX} lrON ep$EP"
  done
  rm -f results/${Date}/mix${MIX}_lrOFF_loss_ep70/temp_pose_infer_rank*.star 2>/dev/null
  bash run_mix_lossexport.sh lrOFF 70 $MIX > LOG/mix${MIX}_loss_lrOFF_ep70.log 2>&1
  echo "  export mix${MIX} lrOFF ep70"
  # 3) split-by-SNR pose accuracy (ep70) for both arms
  for ARM in lrOFF lrON; do
    python3 analyze_mix.py "results/${Date}/mix${MIX}_${ARM}_loss_ep70/temp_pose_infer_rank*.star" $GT $HI \
      > LOG/mix${MIX}_acc_${ARM}.log 2>&1
    echo "  MIXACC ${MIX} ${ARM}:"
    grep -aE "ALL|HIGH|LOW|convention" LOG/mix${MIX}_acc_${ARM}.log | sed 's/^/    /'
  done
  # 4) discard enrichment (lrON): is the low-loss kept set enriched in high-SNR?
  python3 analyze_discard.py $HI 0.5 \
    results/${Date}/mix${MIX}_lrON_loss_ep{10,20,30,40,50,60,70} \
    > LOG/mix${MIX}_discard.log 2>&1
  echo "  DISCARD ${MIX} (lrON):"
  cat LOG/mix${MIX}_discard.log | sed 's/^/    /'
done
echo MIX_ALL_DONE

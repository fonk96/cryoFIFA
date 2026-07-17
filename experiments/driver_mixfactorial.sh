#!/bin/bash
# PART B factorial completion: add the two missing arms so the mix has the full 2x2 (march x loss-rank):
#   lrOFF=marchonly, lrON=both  (already done)  +  lossrankonly (LR, no march), neither (both off)
# Clean per-population accuracy: strict stack-filter the prediction, eval each half vs its OWN GT star.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06
DATA=/sddn/kongfang_work/cryoFIFA/data1
LOW=J528; GTLOW=$DATA/ribo_snr01.star

echo "===== Part B factorial (lossrankonly, neither) start ====="
for MIX in 0103 0105; do
  case $MIX in
    0103) H=J530; GTHI=$DATA/ribo_snr03.star ;;
    0105) H=J532; GTHI=$DATA/ribo_snr05.star ;;
  esac
  for ARM in lossrankonly neither; do
    bash run_mix_train.sh $ARM $MIX > LOG/mix${MIX}_${ARM}.log 2>&1
    echo "  trained mix${MIX} ${ARM}"
    rm -f results/${Date}/mix${MIX}_${ARM}_loss_ep70/temp_pose_infer_rank*.star 2>/dev/null
    bash run_mix_lossexport.sh $ARM 70 $MIX > LOG/mix${MIX}_loss_${ARM}_ep70.log 2>&1
    echo "  export mix${MIX} ${ARM} ep70"
    D=results/${Date}/mix${MIX}_${ARM}_loss_ep70
    grep -h "@${H}"   $D/temp_pose_infer_rank*.star > /tmp/fac_${MIX}_${ARM}_hi.star
    grep -h "@${LOW}" $D/temp_pose_infer_rank*.star > /tmp/fac_${MIX}_${ARM}_lo.star
    AH=$(python3 pose_bestof8.py /tmp/fac_${MIX}_${ARM}_hi.star $GTHI  /tmp/fh_${MIX}_${ARM}.npz 2>&1 | grep -a OVERALL)
    AL=$(python3 pose_bestof8.py /tmp/fac_${MIX}_${ARM}_lo.star $GTLOW /tmp/fl_${MIX}_${ARM}.npz 2>&1 | grep -a OVERALL)
    echo "  FAC ${MIX} ${ARM} HIGH(${H}): $AH"
    echo "  FAC ${MIX} ${ARM} LOW(${LOW}):  $AL"
  done
done
echo MIXFAC_ALL_DONE

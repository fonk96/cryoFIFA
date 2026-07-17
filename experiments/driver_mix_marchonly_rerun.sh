#!/bin/bash
# Re-run marchonly (lrOFF) on the 0.1+0.3 mix with FRESH seeds to test whether the 0% high-SNR result
# is ab-initio basin variance or robust. New dirs (mix0103_lrOFF_seed{1,2}); original mix0103_lrOFF untouched.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; DATA=/sddn/kongfang_work/cryoFIFA/data1
echo "===== marchonly-mix re-run (fresh seeds; original dir preserved) start ====="
echo "  reference: original seed0 marchonly mix high-SNR = 0% <5deg (median 59 deg)"
for S in 1 2; do
  SUF=_seed$S
  SEED=$S SUFFIX=$SUF bash run_mix_train.sh lrOFF 0103 > LOG/mix0103_lrOFF${SUF}.log 2>&1
  echo "  trained marchonly mix seed $S -> results/${Date}/mix0103_lrOFF${SUF}"
  rm -f results/${Date}/mix0103_lrOFF${SUF}_loss_ep70/temp_pose_infer_rank*.star 2>/dev/null
  SUFFIX=$SUF bash run_mix_lossexport.sh lrOFF 70 0103 > LOG/mix0103_lrOFF${SUF}_infer.log 2>&1
  D=results/${Date}/mix0103_lrOFF${SUF}_loss_ep70
  grep -h "@J530" $D/temp_pose_infer_rank*.star > /tmp/rr_lrOFF_s${S}_hi.star
  AH=$(python3 pose_bestof8.py /tmp/rr_lrOFF_s${S}_hi.star $DATA/ribo_snr03.star /tmp/rr_s${S}.npz 2>&1 | grep -a OVERALL)
  echo "  RERUN marchonly mix seed${S} HIGH(0.3 particles): $AH"
done
echo MARCHONLY_RERUN_DONE

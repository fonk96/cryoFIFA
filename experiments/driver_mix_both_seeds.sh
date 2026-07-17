#!/bin/bash
# Run BOTH (lrON = march + loss-rank) on the 0.1+0.3 mix at seeds 1 & 2 (already have seed 0 = 84%),
# to complete the SAME-SEED comparison vs marchonly {seed0=0%, seed1=79%, seed2=0.1%}.
# If both solves the seeds where marchonly collapsed (0,2), loss-rank robustly rescues the bad basins.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; DATA=/sddn/kongfang_work/cryoFIFA/data1; NPZ=results/${Date}/ablation_npz
echo "===== both (lrON) mix seed study start ====="
for S in 1 2; do
  SUF=_seed$S
  SEED=$S SUFFIX=$SUF bash run_mix_train.sh lrON 0103 > LOG/mix0103_lrON${SUF}.log 2>&1
  echo "  trained both mix seed $S"
  rm -f results/${Date}/mix0103_lrON${SUF}_loss_ep70/temp_pose_infer_rank*.star 2>/dev/null
  SUFFIX=$SUF bash run_mix_lossexport.sh lrON 70 0103 > LOG/mix0103_lrON${SUF}_infer.log 2>&1
  D=results/${Date}/mix0103_lrON${SUF}_loss_ep70
  grep -h "@J530" $D/temp_pose_infer_rank*.star > /tmp/b_s${S}_hi.star
  grep -h "@J528" $D/temp_pose_infer_rank*.star > /tmp/b_s${S}_lo.star
  AH=$(python3 pose_bestof8.py /tmp/b_s${S}_hi.star $DATA/ribo_snr03.star $NPZ/mix_both_seed${S}_hi.npz 2>&1 | grep -a OVERALL)
  AL=$(python3 pose_bestof8.py /tmp/b_s${S}_lo.star $DATA/ribo_snr01.star $NPZ/mix_both_seed${S}_lo.npz 2>&1 | grep -a OVERALL)
  echo "  BOTHSEED ${S} HIGH(0.3): $AH"
  echo "  BOTHSEED ${S} LOW(0.1):  $AL"
done
echo BOTH_SEEDS_DONE

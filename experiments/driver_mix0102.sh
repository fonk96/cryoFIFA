#!/bin/bash
# (1) 4-arm factorial on the 0.1+0.2 ribosome mix (HARDER than 0.1+0.3: SNR0.2 sits nearer the noise
# floor than 0.3). Q: base + single arms should struggle -> does frequency MARCHING rescue it?
# arms: neither(base) | lrOFF(march-only) | lossrankonly(LR-only) | lrON(both march+LR).
# Score each subpop vs its OWN single-stack GT (critical: never vs the concatenated star):
#   J528 = SNR0.1 -> ribo_snr01.star ; J529 = SNR0.2 -> ribo_snr02.star ; matched MUST = 20000 each.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
Date=2026_Jul_06; DATA=/sddn/kongfang_work/cryoFIFA/data1; NPZ=results/${Date}/ablation_npz; mkdir -p $NPZ
MIX=0102
if [ ! -d /data/lmdb/ribo_mix0102 ]; then
  echo "===== building lmdb ribo_mix0102 ====="
  PYTHONPATH=../src python3 ../tests/test_newloader.py --particles_meta ${DATA}/ribo_mix0102.star \
     --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --db_path /data/lmdb/ribo_mix0102 --imgsize 128 \
     > LOG/mix0102_lmdb.log 2>&1
  echo "  lmdb build exit=$?"
fi
for ARM in neither lrOFF lossrankonly lrON; do
  echo "===== mix0102 ${ARM} train ====="
  t0=$(date +%s)
  bash run_mix_train.sh ${ARM} 0102 > LOG/mix0102_${ARM}.log 2>&1
  t1=$(date +%s); echo "  train ${ARM} done $((($t1-$t0)/60))min"
  rm -f results/${Date}/mix0102_${ARM}_loss_ep70/temp_pose_infer_rank*.star 2>/dev/null
  bash run_mix_lossexport.sh ${ARM} 70 0102 > LOG/mix0102_${ARM}_infer.log 2>&1
  Dd=results/${Date}/mix0102_${ARM}_loss_ep70
  grep -h "@J528" ${Dd}/temp_pose_infer_rank*.star > /tmp/m2_${ARM}_lo.star   # SNR0.1
  grep -h "@J529" ${Dd}/temp_pose_infer_rank*.star > /tmp/m2_${ARM}_hi.star   # SNR0.2
  AL=$(python3 pose_bestof8.py /tmp/m2_${ARM}_lo.star ${DATA}/ribo_snr01.star ${NPZ}/mix0102_${ARM}_lo.npz 2>&1 | grep -aE "matched|OVERALL")
  AH=$(python3 pose_bestof8.py /tmp/m2_${ARM}_hi.star ${DATA}/ribo_snr02.star ${NPZ}/mix0102_${ARM}_hi.npz 2>&1 | grep -aE "matched|OVERALL")
  echo "  MIX0102 ${ARM} LOW(SNR0.1):  ${AL}"
  echo "  MIX0102 ${ARM} HIGH(SNR0.2): ${AH}"
done
echo MIX0102_DONE

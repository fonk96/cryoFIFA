#!/bin/bash
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA; cd $ROOT/experiments; mkdir -p LOG
for TAG in 03 02 04 05 01; do
  for ARM in both marchonly lossrankonly neither; do
    echo "===== CURRIC $ARM snr$TAG start ====="
    bash run_curriculum_train.sh $ARM $TAG > LOG/curric_${ARM}_${TAG}.log 2>&1
    rm -f results/2026_Jul_06/curric_${ARM}_${TAG}_poses/*.star 2>/dev/null
    bash run_curriculum_poseinfer.sh $ARM $TAG > LOG/curric_${ARM}_${TAG}_poses.log 2>&1
    python3 pose_bestof8.py "results/2026_Jul_06/curric_${ARM}_${TAG}_poses/temp_pose_infer_rank*.star" \
       /sddn/kongfang_work/cryoFIFA/data1/ribo_snr${TAG}.star /tmp/curric_${ARM}_${TAG}.npz > LOG/curric_acc_${ARM}_${TAG}.log 2>&1
    echo "  CURRIC $ARM snr$TAG: $(grep -a OVERALL LOG/curric_acc_${ARM}_${TAG}.log | tail -1)"
  done
done
echo CURRICULUM_ALL_DONE

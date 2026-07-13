#!/bin/bash
# SNR sweep driver: for each SNR, ensure lmdb -> train (box-128, 70ep) -> best-of-8 pose_infer. Sequential.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA
cd $ROOT/experiments
mkdir -p LOG
CSPROJ=/sddn/kongfang_work/YANG/Projects/CS-yang/
for TAG in 005 01 02 03 04; do
  echo "===== SNR $TAG start ====="
  # 1) lmdb (skip if present)
  if [ ! -f /data/lmdb/ribo_snr${TAG}/data.mdb ]; then
    ( cd $ROOT && PYTHONPATH=$ROOT/src python3 tests/test_newloader.py \
        --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_snr${TAG}.star \
        --data_path $CSPROJ --db_path /data/lmdb/ribo_snr${TAG} --imgsize 128 ) > LOG/lmdb_snr${TAG}.log 2>&1
    echo "  lmdb $TAG written"
  else
    echo "  lmdb $TAG exists"
  fi
  # 2) train (blocks)
  bash run_snr_train.sh ${TAG} > LOG/ribo_snr${TAG}.log 2>&1
  echo "  train $TAG done"
  # 3) best-of-8 pose inference (blocks)
  rm -f results/2026_Jul_06/ribo_snr${TAG}_poses/*.star 2>/dev/null
  bash run_snr_poseinfer.sh ${TAG} > LOG/ribo_snr${TAG}_poses.log 2>&1
  echo "  pose_infer $TAG done"
  echo "===== SNR $TAG COMPLETE ====="
done
echo "ALL_SNR_DONE"

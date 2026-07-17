#!/bin/bash
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
ROOT=/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA
cd $ROOT/experiments; mkdir -p LOG
if [ ! -f /data/lmdb/ribo_snr05_shift/data.mdb ]; then
  ( cd $ROOT && PYTHONPATH=$ROOT/src python3 tests/test_newloader.py \
      --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_snr05_shift.star \
      --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --db_path /data/lmdb/ribo_snr05_shift --imgsize 128 ) > LOG/lmdb_j579.log 2>&1
  echo "  lmdb written"
fi
for ARM in ccoff ccon; do
  echo "===== J579 $ARM start ====="
  bash run_j579_train.sh $ARM > LOG/j579_${ARM}.log 2>&1
  echo "  train $ARM done"
  rm -f results/2026_Jul_06/j579_${ARM}_poses/*.star 2>/dev/null
  bash run_j579_poseinfer.sh $ARM > LOG/j579_${ARM}_poses.log 2>&1
  echo "===== J579 $ARM COMPLETE ====="
done
echo J579_ALL_DONE

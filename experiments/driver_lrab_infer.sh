#!/bin/bash
# pose_infer + score the 3 local-refine ablation arms vs ribo_snr05 GT (corroborates the FSC result).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments; mkdir -p LOG
GT=/sddn/kongfang_work/cryoFIFA/data1/ribo_snr05.star
for ARM in standard coarseonly coarselocal; do
  rm -f results/2026_Jul_06/lrab_ribo_${ARM}_poses/temp_pose_infer_rank*.star 2>/dev/null
  bash run_lrablate_ribo.sh $ARM infer > LOG/lrab_infer_${ARM}.log 2>&1
  python3 pose_bestof8.py "results/2026_Jul_06/lrab_ribo_${ARM}_poses/temp_pose_infer_rank*.star" $GT /tmp/lrabacc_${ARM}.npz > LOG/lrab_acc_${ARM}.log 2>&1
  echo "LRABACC ${ARM}: $(grep -a OVERALL LOG/lrab_acc_${ARM}.log | tail -1)"
done
echo LRABINFER_DONE

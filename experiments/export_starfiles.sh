#!/bin/bash
# Produce cryoSPARC-importable star files from the train2 best-of-8 poses with the CORRECTED per-copy
# shift transform (ortho = undo each winning copy's rotation/flip). Half1, half2 and merged (RandomSubset).
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
cd /sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments
CS=/data/cryoFIFA_data/B2_act.star
H1="results/2026_Jul_06/B2_h1_poses/temp_pose_infer_rank*.star"
H2="results/2026_Jul_06/B2_h2_poses/temp_pose_infer_rank*.star"
OUT=results/2026_Jul_06/star_export; mkdir -p $OUT
echo "===== MERGED (half1+half2), ortho-corrected shift ====="
python3 export_star_for_cryosparc.py $CS $OUT/b2_h1h2_ortho_merged.star ortho "$H1" "$H2"
echo "===== half1 only, ortho ====="
python3 export_star_for_cryosparc.py $CS $OUT/b2_h1_ortho.star ortho "$H1"
echo "===== half2 only, ortho ====="
python3 export_star_for_cryosparc.py $CS $OUT/b2_h2_ortho.star ortho "$H2"
echo "===== MERGED, zero-shift (baseline; best in prior test = 3.6A) ====="
python3 export_star_for_cryosparc.py $CS $OUT/b2_h1h2_zero_merged.star zero "$H1" "$H2"
echo "===== files ====="
ls -la $OUT/*.star
echo DONE

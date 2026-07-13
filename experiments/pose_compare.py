#!/usr/bin/env python3
"""
pose_compare.py -- compare cryoFIFA amortized poses vs cryoSPARC (csparc2star) poses.

cryoFIFA is ab-initio, so its poses live in an arbitrary global frame (a fixed rotation R_g,
possibly a hand flip) and use scipy 'zyz' EXTRINSIC euler; cryoSPARC/RELION uses a different
euler convention and its own frame. Rather than derive the mapping by hand, this brute-forces
the convention (euler type/order, transpose, hand) that -- after solving ONE global Kabsch
rotation R_g over all matched particles -- minimises the median per-particle angular residual.
That best-fit residual IS the real amortized-vs-refined pose disagreement.

Inputs
  --fifa  glob of cryoFIFA save_pose star(s): use the LAST epoch, all ranks, e.g.
          'results/2026_Jul_06/B2_h1_savepose/pose_130_rank*.star'
  --cs    cryoSPARC star produced by csparc2star (rlnAngleRot/Tilt/Psi + rlnOriginX/YAngst)
  --apix  A/pixel (to convert cryoSPARC OriginXAngst -> pixels for the shift comparison)

Notes
  * cryoFIFA star cols 14-15 are numpy reprs (brackets/newlines) -> we only read cols 1-13,
    which are clean, and reconstruct the matrix EXACTLY from its 'zyz' extrinsic euler.
  * Particle matching is by rlnImageName with fallbacks (after '@', basename, trailing int);
    the script prints how many matched so you can adjust if the naming differs.
"""
import argparse, glob, os, re, sys
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

M_FLIP = np.diag([1.0, 1.0, -1.0])  # z-mirror: tests the enantiomer (hand) hypothesis


# ----------------------------- star parsing -----------------------------
def parse_fifa(paths):
    """cryoFIFA save_pose star -> {name: (rot,tilt,psi,ox,oy,apix)}. Robust to the messy
    numpy-repr columns 14-15 by taking only the first 13 whitespace tokens of each data row
    and skipping numpy continuation lines (which start with '[')."""
    recs = {}
    files = []
    for p in paths:
        files += sorted(glob.glob(p))
    if not files:
        sys.exit(f"no cryoFIFA star matched: {paths}")
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                t = line.split()
                if len(t) < 13:
                    continue
                if t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'):
                    continue
                try:
                    rot, tilt, psi = float(t[1]), float(t[2]), float(t[3])
                    ox, oy, apix = float(t[10]), float(t[11]), float(t[12])
                except ValueError:
                    continue
                recs[t[0]] = (rot, tilt, psi, ox, oy, apix)   # last write (=latest epoch) wins
    print(f"[fifa] {len(files)} file(s), {len(recs)} unique particles")
    return recs


def parse_relion(path):
    """RELION star, handles multi-block (RELION-3.1 data_optics + data_particles): returns the
    (colmap, rows) of the block that carries the pose (the one with _rlnAngleRot)."""
    colmap, rows = {}, []
    cur = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('data_'):
                cur = {}                          # new block -> reset column map
                continue
            if s.startswith('loop_'):
                continue
            if s.startswith('_'):
                tok = s.split()
                idx = int(tok[1][1:]) - 1 if len(tok) > 1 and tok[1].startswith('#') else len(cur)
                cur[tok[0]] = idx
                continue
            if '_rlnAngleRot' in cur:             # data row of the particles block
                tok = s.split()
                if len(tok) >= len(cur):
                    rows.append(tok)
                    colmap = cur
    if not colmap:
        sys.exit(f"no particles block (with _rlnAngleRot) found in {path}")
    print(f"[cs]   particle columns: {sorted(colmap, key=colmap.get)}")
    print(f"[cs]   {len(rows)} rows")
    return colmap, rows


# ----------------------------- name matching -----------------------------
def norm_keys(name):
    """Yield candidate match keys for a particle name, most-specific first."""
    keys = [name]
    if '@' in name:
        idx, path = name.split('@', 1)
        keys += [name, os.path.basename(path) + '@' + idx.lstrip('0'), idx.lstrip('0')]
        keys.append(os.path.basename(path))
    keys.append(os.path.basename(name))
    m = re.search(r'(\d+)', name)
    if m:
        keys.append(str(int(m.group(1))))
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


# ----------------------------- rotation math -----------------------------
def kabsch(A, B):
    """Global R_g minimising sum ||A_i - R_g B_i||_F over stacks A,B of shape (N,3,3)."""
    H = np.einsum('nij,nkj->ik', B, A)          # sum_i B_i A_i^T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def geodesic_deg(A, B):
    """Per-pair angular distance (deg) between rotation stacks A,B (N,3,3)."""
    tr = np.einsum('nij,nij->n', A, B)          # trace(A_i^T B_i) via elementwise (A,B orthonormal)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def build_cs_matrices(rot, tilt, psi, euler_type, order, transpose, hand):
    ang = np.stack([rot, tilt, psi], 1) if order == 'rtp' else np.stack([psi, tilt, rot], 1)
    R = SciRot.from_euler(euler_type, ang, degrees=True).as_matrix()
    if transpose:
        R = np.transpose(R, (0, 2, 1))
    if hand:
        R = M_FLIP[None] @ R @ M_FLIP[None]
    return R


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fifa', nargs='+', required=True, help="cryoFIFA save_pose star glob(s)")
    ap.add_argument('--cs', required=True, help="cryoSPARC csparc2star star")
    ap.add_argument('--apix', type=float, required=True, help="A/pixel (for shift comparison)")
    args = ap.parse_args()

    fifa = parse_fifa(args.fifa)
    colmap, rows = parse_relion(args.cs)

    def col(row, key, default=None):
        return row[colmap[key]] if key in colmap else default

    # index cryoSPARC by all candidate keys
    cs_idx = {}
    for r in rows:
        nm = col(r, '_rlnImageName')
        if nm is None:
            continue
        for k in norm_keys(nm):
            cs_idx.setdefault(k, r)

    ox_key = '_rlnOriginXAngst' if '_rlnOriginXAngst' in colmap else '_rlnOriginX'
    oy_key = '_rlnOriginYAngst' if '_rlnOriginYAngst' in colmap else '_rlnOriginY'
    ang_scale = 1.0 / args.apix if ox_key.endswith('Angst') else 1.0   # -> pixels

    Acf, cs_rtp, sh_cf, sh_cs = [], [], [], []
    matched = 0
    for nm, (rot, tilt, psi, ox, oy, apix) in fifa.items():
        row = None
        for k in norm_keys(nm):
            if k in cs_idx:
                row = cs_idx[k]; break
        if row is None:
            continue
        matched += 1
        Acf.append(SciRot.from_euler('zyz', [rot, tilt, psi], degrees=True).as_matrix())  # exact round-trip
        cs_rtp.append([float(col(row, '_rlnAngleRot')), float(col(row, '_rlnAngleTilt')),
                       float(col(row, '_rlnAnglePsi'))])
        sh_cf.append([ox, oy])
        csx = float(col(row, ox_key, 0.0)) * ang_scale
        csy = float(col(row, oy_key, 0.0)) * ang_scale
        sh_cs.append([csx, csy])

    print(f"[match] {matched}/{len(fifa)} cryoFIFA particles matched to cryoSPARC")
    if matched < 10:
        sys.exit("too few matches -- check the ImageName naming in both files")
    Acf = np.stack(Acf)
    cs_rtp = np.array(cs_rtp)
    sh_cf, sh_cs = np.array(sh_cf), np.array(sh_cs)

    # brute-force the convention: euler type x angle order x transpose x hand
    best = None
    for et in ('ZYZ', 'zyz'):
        for order in ('rtp', 'ptr'):
            for tr in (False, True):
                for hand in (False, True):
                    B = build_cs_matrices(cs_rtp[:, 0], cs_rtp[:, 1], cs_rtp[:, 2], et, order, tr, hand)
                    Rg = kabsch(Acf, B)
                    res = geodesic_deg(Acf, np.einsum('ij,njk->nik', Rg, B))
                    med = float(np.median(res))
                    if best is None or med < best['med']:
                        best = dict(med=med, et=et, order=order, tr=tr, hand=hand, Rg=Rg, res=res)

    b = best
    print("\n================ BEST-FIT CONVENTION ================")
    print(f"  cryoSPARC euler : {b['et']}  order={'rot,tilt,psi' if b['order']=='rtp' else 'psi,tilt,rot'}"
          f"  transpose={b['tr']}  hand_flip={b['hand']}")
    print(f"  global frame R_g (apply to cryoSPARC to reach cryoFIFA frame):\n{np.array2string(b['Rg'], precision=4)}")
    r = b['res']
    print("\n---------------- pose agreement (residual) ----------------")
    for q in (10, 25, 50, 75, 90):
        print(f"  {q:>2d}th pct angular error : {np.percentile(r, q):6.2f} deg")
    print(f"  mean {r.mean():.2f} deg | < 5deg: {100*np.mean(r<5):.1f}%  < 10deg: {100*np.mean(r<10):.1f}%"
          f"  > 30deg (outliers/mirror): {100*np.mean(r>30):.1f}%")
    if b['hand']:
        print("  NOTE best fit needed a HAND FLIP -> cryoFIFA map is the enantiomer of cryoSPARC's.")

    # shift sign/scale: fit cryoFIFA_px = a * cryoSPARC_px + b, both axis orders
    print("\n---------------- shift comparison ----------------")
    def fit(u, v):
        A = np.vstack([v, np.ones_like(v)]).T
        (a, c), *_ = np.linalg.lstsq(A, u, rcond=None)
        corr = np.corrcoef(u, v)[0, 1]
        return a, c, corr
    xx = fit(sh_cf[:, 0], sh_cs[:, 0]); yy = fit(sh_cf[:, 1], sh_cs[:, 1])
    xy = fit(sh_cf[:, 0], sh_cs[:, 1]); yx = fit(sh_cf[:, 1], sh_cs[:, 0])
    if abs(xx[2]) + abs(yy[2]) >= abs(xy[2]) + abs(yx[2]):
        print(f"  axes NOT swapped   x: fifa = {xx[0]:+.3f}*cs {xx[1]:+.2f} (r={xx[2]:+.2f})"
              f"   y: fifa = {yy[0]:+.3f}*cs {yy[1]:+.2f} (r={yy[2]:+.2f})")
    else:
        print(f"  axes SWAPPED       x<-y: fifa = {xy[0]:+.3f}*cs {xy[1]:+.2f} (r={xy[2]:+.2f})"
              f"   y<-x: fifa = {yx[0]:+.3f}*cs {yx[1]:+.2f} (r={yx[2]:+.2f})")
    print("  (slope ~ -1 => opposite sign, as expected: cryoFIFA shifts the prediction, RELION re-centers)")


if __name__ == '__main__':
    main()

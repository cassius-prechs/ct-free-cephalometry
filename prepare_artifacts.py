"""
prepare_artifacts.py

Distills the 200-case training set into the small artifacts infer.py
needs. Requires data-200_pca; not meant to be run by users of the
released code -- only its output is published under models/.

Outputs (all under models/):
    calib_predictor.pkl  -- PCA+Ridge model predicting (s, R, t) from
                             a face mesh's own shape and scale
    train_mean_face.npy  -- (400, 3) mean training face, for Procrustes
    ridge_ensemble.pkl   -- 5 (PCA, StandardScaler, Ridge) tuples
    release_ssm.npz      -- SSMCorrector state (n_components=44, sigma=3.0)

Usage:
    python prepare_artifacts.py
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cephalo import (
    ALL_CEPHALO_NAMES, MP_USE_INDICES, SSMCorrector,
    _is_mp3d_csv, _normalize_facemesh_sample, _umeyama,
    canonical_lm_name, load_mp3d_csv, parse_cephalo_csv,
)

ROOT = Path(__file__).parent
TRAIN_ROOT = ROOT / "data-200_pca"
MODELS = ROOT / "models"

MIN_HITS_PER_CASE = 6  # skip cases whose STL depth buffer barely hit the mesh
RIDGE_CONFIGS = [(40, 0.1), (40, 0.3), (40, 1.0), (40, 3.0), (50, 0.1)]
SSM_N_COMPONENTS = 44
SSM_SIGMA_CLIP = 3.0
CALIB_PCA_COMPONENTS = 20
CALIB_RIDGE_ALPHA = 1.0


def main():
    mean_meshes, X_mesh, Y_lm = [], [], []
    calib_X, calib_s, calib_R, calib_t = [], [], [], []
    n_cases = 0

    for case_dir in sorted(d for d in TRAIN_ROOT.iterdir() if d.is_dir()):
        case_id = case_dir.name
        fc = case_dir / f"{case_id}_facemesh_mp3d.csv"
        gc = case_dir / f"{case_id}.csv"
        if not fc.exists() or not _is_mp3d_csv(fc) or not gc.exists():
            continue

        mp3d, ct3d, _, ct_hit = load_mp3d_csv(fc)
        hit_idx = np.where(ct_hit)[0]
        if len(hit_idx) < MIN_HITS_PER_CASE:
            continue

        ct_hits, mp_hits = ct3d[hit_idx].astype(np.float64), mp3d[hit_idx].astype(np.float64)

        fm = mp3d[MP_USE_INDICES].astype(np.float32)
        if not np.isfinite(fm).all():
            continue
        fn, _, center, scale = _normalize_facemesh_sample(fm, None)
        mean_meshes.append(fn)

        # Also the regression target for the calibration predictor below.
        s_, R, t = _umeyama(ct_hits, mp_hits)
        gt_dict = {canonical_lm_name(k): v for k, v in parse_cephalo_csv(gc).items()}
        gt_np = np.zeros((24, 3), dtype=np.float32)
        valid = True
        for i, name in enumerate(ALL_CEPHALO_NAMES):
            cn = canonical_lm_name(name)
            if cn in gt_dict:
                gt_mp = (gt_dict[cn].reshape(1, 3) @ R.T * s_ + t).flatten()
                gt_np[i] = (gt_mp - center) / scale
            else:
                valid = False
                break
        if not valid:
            continue

        X_mesh.append(fn.flatten())
        Y_lm.append(gt_np)
        calib_X.append(np.concatenate([fm.flatten(), [scale]]))
        calib_s.append(s_)
        calib_R.append(R)
        calib_t.append(t)
        n_cases += 1

    if n_cases == 0:
        raise RuntimeError(f"No usable training cases found under {TRAIN_ROOT}")
    print(f"Loaded {n_cases} training cases")

    MODELS.mkdir(parents=True, exist_ok=True)

    # Calibration predictor: face shape -> (s, R, t)
    calib_X = np.array(calib_X)
    calib_s, calib_t = np.array(calib_s), np.stack(calib_t)
    calib_R = np.stack(calib_R)
    calib_pca = PCA(n_components=CALIB_PCA_COMPONENTS)
    calib_sc = StandardScaler()
    Xp = calib_sc.fit_transform(calib_pca.fit_transform(calib_X))
    ridge_s = Ridge(alpha=CALIB_RIDGE_ALPHA).fit(Xp, calib_s)
    ridge_t = Ridge(alpha=CALIB_RIDGE_ALPHA).fit(Xp, calib_t)
    ridge_R = Ridge(alpha=CALIB_RIDGE_ALPHA).fit(Xp, calib_R.reshape(len(calib_R), 9))
    with open(MODELS / "calib_predictor.pkl", "wb") as f:
        pickle.dump((calib_pca, calib_sc, ridge_s, ridge_t, ridge_R), f)
    print(f"  saved {MODELS / 'calib_predictor.pkl'}")

    # Mean training face (for optional Procrustes alignment)
    mean_face = np.mean(mean_meshes, axis=0).astype(np.float32)
    np.save(MODELS / "train_mean_face.npy", mean_face)
    print(f"  saved {MODELS / 'train_mean_face.npy'}  (from {len(mean_meshes)} cases)")

    # SSM shape prior
    X_mesh, Y_lm = np.stack(X_mesh), np.stack(Y_lm)
    ssm = SSMCorrector(n_components=SSM_N_COMPONENTS, sigma_clip=SSM_SIGMA_CLIP)
    ssm.fit(Y_lm)
    ssm.save(MODELS / "release_ssm.npz")
    print(f"  saved {MODELS / 'release_ssm.npz'}")

    # Ridge ensemble
    ridge_ensemble = []
    for n, alpha in RIDGE_CONFIGS:
        pca = PCA(n_components=n)
        Xp = pca.fit_transform(X_mesh)
        sc = StandardScaler()
        Xs = sc.fit_transform(Xp)
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xs, Y_lm.reshape(len(Y_lm), -1))
        ridge_ensemble.append((pca, sc, ridge))
    with open(MODELS / "ridge_ensemble.pkl", "wb") as f:
        pickle.dump(ridge_ensemble, f)
    print(f"  saved {MODELS / 'ridge_ensemble.pkl'}  ({len(ridge_ensemble)} models)")

    print("\nDone. infer.py can now run without data-200_pca.")


if __name__ == "__main__":
    main()

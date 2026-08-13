#!/usr/bin/env python3
"""
infer.py

3D cephalometric landmark estimation from a single RGB photograph.

Pipeline: photo -> MediaPipe Face Landmarker -> MP400 subset -> normalize
-> Stage0+1 ensemble (x6) + Ridge ensemble (x5), blended 50/50 -> SSM
correction -> metric-scale decoding -> angles + classification.

Loads precomputed artifacts from models/ (see prepare_artifacts.py).

Usage:
    python infer.py --image photo.jpg [--procrustes]
"""
import argparse
import math
import pickle
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).parent))
from cephalo import (
    ALL_CEPHALO_NAMES, MP_USE_INDICES, SSMCorrector,
    _load_checkpoint, _normalize_facemesh_sample, paper_lm_name,
    compute_normals,
)
from ensemble import load_stage1, _flip_unflip

DEVICE = "cpu"
ROOT = Path(__file__).parent
M = ROOT / "models"

MP_MODEL_PATH = ROOT / "face_landmarker.task"
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_mp_model():
    if not MP_MODEL_PATH.exists():
        print(f"Downloading MediaPipe Face Landmarker model to {MP_MODEL_PATH}...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)

NN_CFGS = [
    (M / "pca_full200_cross_attn/best.pth", "pca_full200_snap_s0", False),
    (M / "pca_full200_cross_attn_s1/best.pth", "pca_full200_snap_s0_s1", False),
    (M / "pca_full200_cross_attn_s2/best.pth", "pca_full200_snap_s0_s2", False),
    (M / "pca_full200_cross_attn_normals/best.pth", None, True),
    (M / "pca_full200_cross_attn_normals_s1/best.pth", "pca_full200_normals_s1_stage1", True),
    (M / "pca_full200_cross_attn_normals_s3/best.pth", "pca_full200_normals_s3_stage1", True),
]


def _snapshots(d):
    return [M / f"{d}/stage1_snapshot_0{i}.pth" for i in range(1, 6)] + [M / f"{d}/stage1_best.pth"]


def _safe_norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def _ang3(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1 + 1e-6, 1 - 1e-6))))


def compute_angles(lm):
    """lm indexed per ALL_CEPHALO_NAMES order; returns dict of clinical angles."""
    S, Na = lm[3], lm[4]
    A, B_ = lm[14], lm[17]
    PoL, PoR = lm[5], lm[6]
    OrL, OrR = lm[7], lm[8]
    GoL, GoR = lm[20], lm[21]
    Me = lm[19]
    SNA, SNB = _ang3(S, Na, A), _ang3(S, Na, B_)
    ANB = SNA - SNB
    SN, GoMe = Na - S, Me - (GoL + GoR) * 0.5
    SN_MP = math.degrees(math.acos(abs(float(np.clip(
        np.dot(_safe_norm(SN), _safe_norm(GoMe)), -1 + 1e-6, 1 - 1e-6)))))
    Or = (OrL + OrR) * 0.5
    FH_n = _safe_norm(np.cross(PoR - PoL, Or - PoL))
    FMA = math.degrees(math.asin(min(abs(float(np.dot(FH_n, _safe_norm(GoMe)))), 1 - 1e-6)))
    return {"SNA": SNA, "SNB": SNB, "ANB": ANB, "SN-MP": SN_MP, "FMA": FMA}


def classify_skeletal(anb):
    if anb <= 0:
        return "III"
    return "I" if anb <= 4 else "II"


def procrustes_align(q, target, max_iter=50, tol=1e-6):
    q = q.copy()
    for _ in range(max_iter):
        H = q.T @ target
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        q_new = q @ R.T
        if np.linalg.norm(R - np.eye(3), "fro") < tol:
            return q_new
        q = q_new
    return q


class Pipeline:
    def __init__(self, device=DEVICE):
        self.device = device

        _ensure_mp_model()
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.5))

        self.nn_pipes = []
        for s0_path, s1_dir, use_normals in NN_CFGS:
            if not s0_path.exists():
                continue
            s0, _, _ = _load_checkpoint(s0_path, device=device)
            s0.eval()
            if s1_dir is not None:
                s1_paths = _snapshots(s1_dir)
            else:
                s1_paths = [M / f"pca_full200_cross_attn_normals_stage1_v3_s{i}/stage1_best.pth" for i in range(5)]
            s1s = [load_stage1(p) for p in s1_paths if p.exists()]
            self.nn_pipes.append((s0, s1s, use_normals))
        if not self.nn_pipes:
            raise RuntimeError(f"No NN checkpoints found under {M}")

        with open(M / "ridge_ensemble.pkl", "rb") as f:
            self.ridge_ensemble = pickle.load(f)

        self.ssm = SSMCorrector.load(M / "release_ssm.npz")
        self.mean_face = np.load(M / "train_mean_face.npy")

        with open(M / "calib_predictor.pkl", "rb") as f:
            self.calib_pca, self.calib_sc, self.ridge_s, self.ridge_t, self.ridge_R = pickle.load(f)

    def _run_mediapipe(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]], dtype=np.float32)

    def _nn_predict(self, fn):
        nm = compute_normals(fn)
        fn6 = np.concatenate([fn, nm], -1).astype(np.float32)
        t3 = torch.from_numpy(fn)[None].to(self.device).float()
        f3 = t3.clone(); f3[..., 0] *= -1
        t6 = torch.from_numpy(fn6)[None].to(self.device).float()
        f6 = t6.clone(); f6[..., 0] *= -1; f6[..., 3] *= -1
        s0_preds, s1_preds = [], []
        with torch.no_grad():
            for s0, s1s, use_normals in self.nn_pipes:
                fi, ff = (t6, f6) if use_normals else (t3, f3)
                c, cf = s0(fi), s0(ff)
                s0_preds.append(c[0].cpu().numpy())
                s0_preds.append(_flip_unflip(cf[0].cpu().numpy()))
                for s1 in s1s:
                    s1_preds.append(s1(fi, c)[0].cpu().numpy())
                    s1_preds.append(_flip_unflip(s1(ff, cf)[0].cpu().numpy()))
        return np.mean(s1_preds, 0) if s1_preds else np.mean(s0_preds, 0)

    def _ridge_predict(self, fn):
        xf = fn.flatten()[None]
        preds = [ridge.predict(sc.transform(pca.transform(xf))).reshape(24, 3).astype(np.float32)
                 for pca, sc, ridge in self.ridge_ensemble]
        return np.mean(preds, 0)

    def _predict_calibration(self, fm, scale):
        """Predicts (s, R, t) from the input face mesh's own shape/scale."""
        xf = np.concatenate([fm.flatten(), [scale]]).reshape(1, -1)
        xp = self.calib_sc.transform(self.calib_pca.transform(xf))
        s = float(self.ridge_s.predict(xp)[0])
        t = self.ridge_t.predict(xp)[0]
        R_flat = self.ridge_R.predict(xp)[0].reshape(3, 3)
        U, _, Vt = np.linalg.svd(R_flat)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return s, R, t

    def predict(self, image_path, use_procrustes=False):
        """Returns dict with 'landmarks_mm' (24 named 3D points), 'angles', 'skeletal_class'."""
        raw478 = self._run_mediapipe(image_path)
        if raw478 is None:
            return None

        fm = raw478[MP_USE_INDICES].astype(np.float32)
        fn, _, center, scale = _normalize_facemesh_sample(fm, None)
        if use_procrustes:
            fn = procrustes_align(fn, self.mean_face).astype(np.float32)

        pred_nn = self._nn_predict(fn)
        pred_ridge = self._ridge_predict(fn)
        pred_norm = self.ssm.correct(0.5 * pred_nn + 0.5 * pred_ridge)

        s, R, t = self._predict_calibration(fm, scale)
        mp_space = pred_norm.astype(np.float64) * scale + center
        pred_mm = ((mp_space - t) @ R) / s

        lm = {paper_lm_name(name): pred_mm[i] for i, name in enumerate(ALL_CEPHALO_NAMES)}
        angles = compute_angles(pred_mm)
        return {
            "landmarks_mm": lm,
            "angles": angles,
            "skeletal_class": classify_skeletal(angles["ANB"]),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="Path to a facial photograph")
    ap.add_argument("--procrustes", action="store_true",
                     help="Align input landmarks to the training mean face before prediction")
    args = ap.parse_args()

    pipeline = Pipeline()
    result = pipeline.predict(args.image, use_procrustes=args.procrustes)
    if result is None:
        print("No face detected in the image.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Cephalometric landmarks (mm, CT-space calibration) ===")
    for name, xyz in result["landmarks_mm"].items():
        print(f"  {name:<10} {xyz[0]:8.2f} {xyz[1]:8.2f} {xyz[2]:8.2f}")

    print(f"\n=== Cephalometric angles ===")
    for name, val in result["angles"].items():
        print(f"  {name:<6} {val:6.2f} deg")

    print(f"\nSkeletal classification: Class {result['skeletal_class']}")


if __name__ == "__main__":
    main()

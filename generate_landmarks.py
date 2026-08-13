"""
generate_landmarks.py

Renders each case's aligned STL mesh, runs MediaPipe Face Landmarker on
it, and saves the 478 3D landmark coordinates (world mm) to
{case_id}_facemesh_mesh3d.csv.

Output CSV format:
    index, mesh_x, mesh_y, mesh_z, inside_foreground, hit
    0, 12.3, -45.6, 78.9, True, True
    ...
    477, nan, nan, nan, False, False   <- undetected point

Usage:
    python generate_landmarks.py --aligned-root /path/to/dataset_aligned [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

N_MP_LANDMARKS = 478

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────

def _download_model():
    if not Path(MODEL_PATH).exists():
        print("Downloading Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def create_face_landmarker() -> vision.FaceLandmarker:
    _download_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def _look_at(camera_position, target, up=(0, 1, 0)) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target          = np.asarray(target,          dtype=np.float32)
    up              = np.asarray(up,               dtype=np.float32)
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def _camera_intrinsics(width: int, height: int, yfov: float):
    fy = 0.5 * height / np.tan(yfov * 0.5)
    fx = fy   # square pixels
    cx = width  * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


def _cleanup_mesh(mesh):
    mesh = mesh.copy()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _render_from(
    mesh,
    cam_offset: np.ndarray,
    up: Tuple,
    width: int,
    height: int,
    yfov: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renders from an arbitrary view direction."""
    center  = mesh.centroid
    radius  = np.max(mesh.extents) * 2.0
    cam_pos = center + cam_offset / np.linalg.norm(cam_offset) * radius
    pose    = _look_at(cam_pos, center, up=up)

    scene = pyrender.Scene(
        bg_color=[255, 255, 255],
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    camera = pyrender.PerspectiveCamera(yfov=yfov, znear=1e-2, zfar=1e4)
    light  = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(camera, pose=pose)
    scene.add(light,  pose=pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()
    return color, depth, pose


# Six candidate view directions (offset, up), tried in priority order.
# +Z is tried first since PCA-aligned data faces +Z.
_CANDIDATE_VIEWS: List[Tuple[np.ndarray, Tuple]] = [
    (np.array([0,  0, 1], dtype=np.float32), (0, 1, 0)),   # +Z (PCA-aligned front)
    (np.array([0,  0,-1], dtype=np.float32), (0, 1, 0)),   # -Z
    (np.array([0, -1, 0], dtype=np.float32), (0, 0, 1)),   # -Y (legacy-aligned front)
    (np.array([0,  1, 0], dtype=np.float32), (0, 0, 1)),   # +Y
    (np.array([1,  0, 0], dtype=np.float32), (0, 1, 0)),   # +X
    (np.array([-1, 0, 0], dtype=np.float32), (0, 1, 0)),   # -X
]


def _render_best_view(
    mesh,
    landmarker,
    width: int,
    height: int,
    yfov: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Tries the six candidate views and returns the one where MediaPipe
    detects the most landmarks. mp3d is cached and returned to avoid
    detecting twice.
    Returns: (color, depth, pose, mp3d); mp3d may be None.
    """
    best = (None, None, None, None)
    best_count = -1

    for cam_offset, up in _CANDIDATE_VIEWS:
        color, depth, pose = _render_from(mesh, cam_offset, up, width, height, yfov)
        mp3d = _detect_3d(landmarker, color)
        count = 0 if mp3d is None else int(np.isfinite(mp3d[:, 0]).sum())
        if count > best_count:
            best_count = count
            best = (color, depth, pose, mp3d)
        if count == N_MP_LANDMARKS:
            break  # all points detected, stop early

    return best


def _detect_3d(
    landmarker,
    color,
) -> Optional[np.ndarray]:
    """
    Extracts (x, y, z) directly from MediaPipe's face_landmarks.

    x, y: normalized image coordinates [0, 1].
    z: MediaPipe's own monocular depth estimate (relative to face size).

    Doesn't use the CT depth buffer, so the same extraction applies to
    real photographs. Returns (478, 3) float32, or None.
    """
    rgb = np.ascontiguousarray(color[:, :, :3], dtype=np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
        dtype=np.float32,
    )


def _unproject(
    landmarks_norm: np.ndarray,   # (478, 2) normalized [0,1]
    depth: np.ndarray,            # (H, W)
    pose: np.ndarray,             # (4, 4) camera-to-world
    width: int,
    height: int,
    yfov: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        coords3d : (478, 3) float32 world coordinates (mm), NaN where invalid
        detected : (478,) bool
    """
    fx, fy, cx, cy = _camera_intrinsics(width, height, yfov)
    coords3d = np.full((len(landmarks_norm), 3), np.nan, dtype=np.float32)
    detected = np.zeros(len(landmarks_norm), dtype=bool)

    for i, (nx, ny) in enumerate(landmarks_norm):
        u = int(np.clip(nx * width,  0, width  - 1))
        v = int(np.clip(ny * height, 0, height - 1))
        z = float(depth[v, u])
        if z <= 0:
            continue
        x = (u - cx) / fx * z
        y = -((v - cy) / fy * z)
        world = pose @ np.array([x, y, -z, 1.0], dtype=np.float32)
        coords3d[i] = world[:3]
        detected[i] = True

    return coords3d, detected


# ─────────────────────────────────────────────
# Per-case processing
# ─────────────────────────────────────────────

def find_aligned_stl(case_dir: Path) -> Optional[Path]:
    case_id = case_dir.name
    for pattern in [
        f"{case_id}_softtissue.stl",
        f"{case_id}_face_surface.stl",
        f"{case_id}_face_surface_aligned.stl",
        "*softtissue*.stl",
        "*face*.stl",
    ]:
        cands = list(case_dir.glob(pattern)) if "*" in pattern \
            else ([case_dir / pattern] if (case_dir / pattern).exists() else [])
        if cands:
            return cands[0]
    return None


def process_case_mp3d(
    case_dir: Path,
    landmarker: vision.FaceLandmarker,
    width: int,
    height: int,
    yfov: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Renders the aligned STL and returns both the CT-independent MediaPipe
    3D coordinates and the CT-space coordinates used to bridge to ground
    truth during dataset preparation.

    mp3d   : (478, 3) MediaPipe face_landmarks (x, y, z). Valid for all
             478 points once a face is detected (independent of the depth
             buffer) -- the same extraction applies to real photographs,
             consistent with CT-free inference.
    ct3d   : (478, 3) CT-space mm coordinates (depth-buffer unproject).
             Valid only where the depth buffer hit the STL surface; used
             solely as the ground-truth coordinate bridge.
    mp_hit : (478,) bool, whether MediaPipe detected this point (all 478
             True whenever a face is detected).
    ct_hit : (478,) bool, whether the depth buffer hit the STL surface.
    """
    stl = find_aligned_stl(case_dir)
    if stl is None:
        raise FileNotFoundError(f"No face surface STL found in {case_dir}")

    mesh = trimesh.load(str(stl), process=False)
    mesh = _cleanup_mesh(mesh)

    color, depth, pose, mp3d = _render_best_view(mesh, landmarker, width, height, yfov)

    if mp3d is None:
        empty = np.full((N_MP_LANDMARKS, 3), np.nan, dtype=np.float32)
        return (empty, empty,
                np.zeros(N_MP_LANDMARKS, dtype=bool),
                np.zeros(N_MP_LANDMARKS, dtype=bool))

    mp_hit = np.ones(N_MP_LANDMARKS, dtype=bool)
    ct3d, ct_hit = _unproject(mp3d[:, :2], depth, pose, width, height, yfov)

    return mp3d, ct3d, mp_hit, ct_hit


# ─────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────

def save_mp_csv(path: Path, coords3d: np.ndarray, detected: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mesh_x", "mesh_y", "mesh_z",
                          "inside_foreground", "hit"])
        for i in range(len(coords3d)):
            det = bool(detected[i])
            if det:
                x, y, z = coords3d[i]
                writer.writerow([i, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                                  True, True])
            else:
                writer.writerow([i, "nan", "nan", "nan", False, False])


def save_mp3d_csv(
    path: Path,
    mp3d: np.ndarray,
    ct3d: np.ndarray,
    mp_hit: np.ndarray,
    ct_hit: np.ndarray,
):
    """
    Saves the _facemesh_mp3d.csv format.

    Columns:
      mp_x, mp_y, mp_z : MediaPipe face_landmarks (x, y, z) -- matches
                         real-photograph inference; valid for all 478
                         points once a face is detected.
      ct_x, ct_y, ct_z : CT-space mm coordinates, for the ground-truth
                         bridge; valid only where the depth buffer hit
                         the STL surface (NaN at the edges).
      mp_hit : whether mp_x,y,z are valid (MediaPipe detection flag).
      ct_hit : whether ct_x,y,z are valid (depth-buffer hit flag).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index",
                         "mp_x",  "mp_y",  "mp_z",
                         "ct_x",  "ct_y",  "ct_z",
                         "mp_hit", "ct_hit"])
        for i in range(len(mp3d)):
            mh = bool(mp_hit[i])
            ch = bool(ct_hit[i])
            mx_s = f"{mp3d[i,0]:.8f}" if mh else "nan"
            my_s = f"{mp3d[i,1]:.8f}" if mh else "nan"
            mz_s = f"{mp3d[i,2]:.8f}" if mh else "nan"
            cx_s = f"{ct3d[i,0]:.6f}" if ch else "nan"
            cy_s = f"{ct3d[i,1]:.6f}" if ch else "nan"
            cz_s = f"{ct3d[i,2]:.6f}" if ch else "nan"
            writer.writerow([i, mx_s, my_s, mz_s, cx_s, cy_s, cz_s, mh, ch])


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def generate_for_root(
    aligned_root: Path,
    overwrite: bool,
    width: int,
    height: int,
    yfov: float,
    landmarker: vision.FaceLandmarker,
    mp3d_only: bool = False,
):
    """
    mp3d_only=False (default): generates both the legacy
        _facemesh_mesh3d.csv and the current _facemesh_mp3d.csv.
    mp3d_only=True: generates only _facemesh_mp3d.csv.
    """
    case_dirs = sorted([p for p in aligned_root.iterdir() if p.is_dir()])
    print(f"\n=== {aligned_root} : {len(case_dirs)} cases ===")

    n_done = n_skip = n_fail = 0

    for i, case_dir in enumerate(case_dirs):
        case_id      = case_dir.name
        out_old_csv  = case_dir / f"{case_id}_facemesh_mesh3d.csv"
        out_mp3d_csv = case_dir / f"{case_id}_facemesh_mp3d.csv"

        old_exists   = out_old_csv.exists()
        mp3d_exists  = out_mp3d_csv.exists()

        need_old  = not mp3d_only and (not old_exists or overwrite)
        need_mp3d = not mp3d_exists or overwrite

        if not need_old and not need_mp3d:
            print(f"[{i+1:3d}/{len(case_dirs)}] {case_id}  SKIP (exists)")
            n_skip += 1
            continue

        print(f"[{i+1:3d}/{len(case_dirs)}] {case_id}", end="  ", flush=True)
        try:
            mp3d, ct3d, mp_hit, ct_hit = process_case_mp3d(
                case_dir, landmarker, width, height, yfov
            )
            n_mp = int(mp_hit.sum())
            n_ct = int(ct_hit.sum())
            print(f"mp={n_mp}/{N_MP_LANDMARKS}  ct={n_ct}/{N_MP_LANDMARKS}")

            if need_mp3d:
                save_mp3d_csv(out_mp3d_csv, mp3d, ct3d, mp_hit, ct_hit)

            if need_old:
                save_mp_csv(out_old_csv, ct3d, ct_hit)

            n_done += 1
        except Exception as e:
            print(f"FAIL — {e}")
            n_fail += 1

    print(f"\n  done={n_done}  skip={n_skip}  fail={n_fail}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate MediaPipe 478-point 3D landmarks for aligned STL datasets"
    )
    ap.add_argument("--aligned-root", required=True,
                    help="Root of aligned dataset (data-200_aligned or eval-40_aligned)")
    ap.add_argument("--width",     type=int,   default=1024)
    ap.add_argument("--height",    type=int,   default=1024)
    ap.add_argument("--yfov",      type=float, default=0.7854)  # pi/4
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--mp3d-only", action="store_true",
        help="Generate only _facemesh_mp3d.csv, skipping the legacy _facemesh_mesh3d.csv",
    )
    return ap


def main():
    args = build_parser().parse_args()
    aligned_root = Path(args.aligned_root)

    landmarker = create_face_landmarker()
    try:
        generate_for_root(
            aligned_root=aligned_root,
            overwrite=args.overwrite,
            width=args.width,
            height=args.height,
            yfov=args.yfov,
            landmarker=landmarker,
            mp3d_only=args.mp3d_only,
        )
    finally:
        landmarker.close()

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()
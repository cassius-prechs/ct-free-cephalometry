"""
cephalo.py

Direct regression pipeline: 400 MediaPipe FaceMesh 3D points -> 24
cephalometric landmarks.

  Stage 0: CrossAttentionCephaloNet   cross-attention coarse estimate
  Stage 1: Stage1AttnV3RefinementNet  attention-based refinement
  Stage 2: SSMCorrector               statistical shape model outlier correction

Input: {case_id}_facemesh_mp3d.csv (mp3d columns, 400 points).
Ground-truth landmarks are brought into the same normalized frame via
a per-case Umeyama transform (CT space -> MediaPipe space).
"""

import argparse
import csv
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# MP_USE_INDICES: the 400 MediaPipe indices detected consistently across
# every training/eval case (intersection of always-hit indices).
MP_USE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 55, 56, 57, 59, 60, 61, 62, 64, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 106, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 133, 134, 140, 141, 142, 143, 144, 145, 146, 147, 153, 154, 155, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 170, 171, 173, 174, 175, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 285, 286, 287, 289, 290, 291, 292, 294, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331, 335, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 357, 358, 359, 360, 362, 363, 364, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 394, 395, 396, 398, 399, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477]

N_MP = 478
N_USE = len(MP_USE_INDICES)  # 400

def _load_mp_use_indices_from_file(txt_path: Path) -> List[int]:
    indices = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        try:
            indices.append(int(line))
        except ValueError:
            pass
    return sorted(indices)

def _ensure_mp_use_indices(coverage_txt: Optional[Path] = None):
    global MP_USE_INDICES
    if MP_USE_INDICES:
        return
    search_paths = [
        Path("mp_coverage_report/always_hit_indices.txt"),
        Path("always_hit_indices.txt"),
    ]
    if coverage_txt:
        search_paths.insert(0, coverage_txt)
    for p in search_paths:
        if p.exists():
            MP_USE_INDICES = _load_mp_use_indices_from_file(p)
            print(f"  [MP_USE_INDICES] Loaded {len(MP_USE_INDICES)} indices from {p}")
            return
    raise RuntimeError(
        "MP_USE_INDICES is not set.\n"
        "  1. Paste the contents of always_hit_indices.txt into MP_USE_INDICES above, or\n"
        "  2. pass its path via --coverage-txt"
    )

ALL_CEPHALO_NAMES: List[str] = [
    "鼻尖", "SN", "軟組織Pog",
    "S", "Na", "Po L", "Po R", "Or L", "Or R", "Ba",
    "FO L", "FO R", "AZ", "ZA",
    "Pt.A", "ANS", "PNS", "pt.B", "Pog", "Me",
    "Go L", "Go R", "CoS L", "CoS R",
]
assert len(ALL_CEPHALO_NAMES) == 24

FLIP_PAIRS = [
    (ALL_CEPHALO_NAMES.index("Po L"),  ALL_CEPHALO_NAMES.index("Po R")),
    (ALL_CEPHALO_NAMES.index("Or L"),  ALL_CEPHALO_NAMES.index("Or R")),
    (ALL_CEPHALO_NAMES.index("FO L"),  ALL_CEPHALO_NAMES.index("FO R")),
    (ALL_CEPHALO_NAMES.index("Go L"),  ALL_CEPHALO_NAMES.index("Go R")),
    (ALL_CEPHALO_NAMES.index("CoS L"), ALL_CEPHALO_NAMES.index("CoS R")),
]

# Landmarks not directly visible on the facial surface, weighted higher in loss.
_HARD_LM_IDX = [
    ALL_CEPHALO_NAMES.index(n)
    for n in ["Po L", "Po R", "Ba", "FO L", "FO R", "Go L", "Go R", "CoS L", "CoS R"]
]

MP_MODEL_PATH = "face_landmarker.task"
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

_ALIAS_CANON = {
    "pt.b": "pt.B", "pt b": "pt.B", "pt_b": "pt.B",
    "n": "Na", "s": "S", "ｓ": "S",
}

# English display names for paper output (internal name → paper label)
PAPER_LM_NAMES: Dict[str, str] = {
    "鼻尖":      "Prn",    # Pronasale
    "SN":        "Sn",     # Subnasale
    "軟組織Pog": "Pog'",   # Soft-tissue Pogonion
    "pt.B":      "B",      # B point (Supramentale)
    "Pt.A":      "A",      # A point (Subspinale)
}

def paper_lm_name(name: str) -> str:
    """Return English display name for paper tables/figures."""
    return PAPER_LM_NAMES.get(name, name)

def _normalize_token(s: str) -> str:
    s = s.replace("\ufeff", "")
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def canonical_lm_name(name: str) -> str:
    t = _normalize_token(name)
    key = t.casefold().replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return _ALIAS_CANON.get(key, t)

def parse_cephalo_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Reads a ground-truth cephalo CSV (name, x, y, z rows)."""
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    delim = "," if txt.count(",") >= max(txt.count("\t"), txt.count(";")) else ("\t" if "\t" in txt else ";")
    lm: Dict[str, np.ndarray] = {}
    for r in txt.splitlines():
        r = r.strip()
        if not r:
            continue
        parts = [p.strip() for p in r.split(delim)]
        if len(parts) < 4:
            continue
        name = parts[0]
        nums: List[float] = []
        for p in parts[1:]:
            m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p)
            if m:
                nums.append(float(m[0]))
            if len(nums) == 3:
                break
        if len(nums) == 3:
            lm[name] = np.array(nums, dtype=np.float64)
    return lm

def load_facemesh_csv(csv_path: Path) -> Optional[np.ndarray]:
    """Legacy _facemesh_mesh3d.csv format -> (478, 3); hit=False rows are NaN."""
    coords = np.full((N_MP, 3), np.nan, dtype=np.float32)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                hit = row["hit"].strip().lower() in ("true", "1", "yes")
                if not hit or not (0 <= idx < N_MP):
                    continue
                coords[idx, 0] = float(row["mesh_x"])
                coords[idx, 1] = float(row["mesh_y"])
                coords[idx, 2] = float(row["mesh_z"])
            except (KeyError, ValueError):
                continue
    if not np.isfinite(coords).any():
        return None
    return coords


def load_mp3d_csv(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the _facemesh_mp3d.csv format.
    Returns (mp3d, ct3d, mp_hit, ct_hit):
      mp3d   : (478, 3) MediaPipe 3D coordinates (NaN where mp_hit=False)
      ct3d   : (478, 3) CT mm coordinates (NaN where ct_hit=False)
      mp_hit : (478,) bool
      ct_hit : (478,) bool
    """
    mp3d   = np.full((N_MP, 3), np.nan, dtype=np.float32)
    ct3d   = np.full((N_MP, 3), np.nan, dtype=np.float32)
    mp_hit = np.zeros(N_MP, dtype=bool)
    ct_hit = np.zeros(N_MP, dtype=bool)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                if not (0 <= idx < N_MP):
                    continue
                mh = row["mp_hit"].strip().lower() in ("true", "1")
                ch = row["ct_hit"].strip().lower() in ("true", "1")
                mp_hit[idx] = mh
                ct_hit[idx] = ch
                if mh:
                    mp3d[idx] = [float(row["mp_x"]), float(row["mp_y"]), float(row["mp_z"])]
                if ch:
                    ct3d[idx] = [float(row["ct_x"]), float(row["ct_y"]), float(row["ct_z"])]
            except (KeyError, ValueError):
                continue
    return mp3d, ct3d, mp_hit, ct_hit


def find_facemesh_csv(case_dir: Path, case_id: str) -> Optional[Path]:
    """Prefers the _mp3d format, falling back to the legacy _mesh3d format."""
    for pat in (f"{case_id}_facemesh_mp3d.csv", "*_facemesh_mp3d.csv",
                f"{case_id}_facemesh_mesh3d.csv", "*_facemesh_mesh3d.csv"):
        if "*" in pat:
            cands = sorted(case_dir.glob(pat))
            if cands:
                return cands[0]
        else:
            p = case_dir / pat
            if p.exists():
                return p
    return None


def _is_mp3d_csv(csv_path: Path) -> bool:
    return "_facemesh_mp3d" in csv_path.name


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama similarity transform: dst ~= s * (R @ src.T).T + t
    src, dst: (N, 3). Returns (s, R, t).
    """
    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst
    var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(d, dtype=np.float64)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[d - 1, d - 1] = -1.0
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / max(var_src, 1e-12))
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def load_face_mesh_use(csv_path: Path) -> np.ndarray:
    """Returns the MP_USE_INDICES subset (N_USE, 3), preferring mp3d columns."""
    if _is_mp3d_csv(csv_path):
        mp3d, _, _, _ = load_mp3d_csv(csv_path)
        coords = mp3d[MP_USE_INDICES]
    else:
        raw = load_facemesh_csv(csv_path)
        if raw is None:
            raise RuntimeError(f"Failed to load facemesh CSV: {csv_path}")
        coords = raw[MP_USE_INDICES]
    if not np.isfinite(coords).all():
        raise RuntimeError(f"NaN in FaceMesh (USE indices): {csv_path}")
    return coords.astype(np.float32)


def _normalize_facemesh_sample(
    face_mesh: np.ndarray,
    cephalo: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """Centroid+RMS-scale normalization; applies the same transform to cephalo."""
    center = face_mesh.mean(axis=0)
    face_c = face_mesh - center
    scale  = float(np.sqrt(np.mean(np.sum(face_c ** 2, axis=1))))
    scale  = max(scale, 1e-6)
    face_n = face_c / scale
    cephalo_n = None if cephalo is None else (cephalo - center) / scale
    return (
        face_n.astype(np.float32),
        None if cephalo_n is None else cephalo_n.astype(np.float32),
        center.astype(np.float32),
        scale,
    )


def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Local-PCA surface normal estimation.
    points: (N, 3) normalized coords -> (N, 3) outward-facing unit normals.
    """
    n = len(points)
    d2 = np.sum((points[:, None] - points[None]) ** 2, axis=-1)  # (N, N)
    nn_idx = np.argsort(d2, axis=1)[:, :k + 1]                   # (N, k+1)
    neighbors = points[nn_idx]                                     # (N, k+1, 3)
    centroid_local = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroid_local                          # (N, k+1, 3)
    face_center = points.mean(axis=0)
    normals = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        _, _, vh = np.linalg.svd(centered[i], full_matrices=False)
        normal = vh[-1].astype(np.float32)
        if np.dot(normal, points[i] - face_center) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def _augment_facemesh(
    face_mesh: np.ndarray,
    cephalo: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies augmentation in the normalized frame.
    face_mesh: (N, 3) or (N, 6) [xyz | nxyz], cephalo: (24, 3), both normalized.
    When face_mesh is (N, 6), the normal columns (3-5) are transformed too.
    """
    use_normals = face_mesh.shape[1] == 6
    xyz  = face_mesh[:, :3].copy()
    nxyz = face_mesh[:, 3:].copy() if use_normals else None

    # Left-right flip (50%)
    if np.random.rand() < 0.5:
        xyz[:, 0] *= -1
        if use_normals:
            nxyz[:, 0] *= -1
        cephalo = cephalo.copy(); cephalo[:, 0] *= -1
        for i, j in FLIP_PAIRS:
            cephalo[[i, j]] = cephalo[[j, i]]

    # Random rotation about Z, +/-15 deg
    angle = np.random.uniform(-15.0, 15.0) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    Rz = np.array([[cos_a, -sin_a, 0.0],
                   [sin_a,  cos_a, 0.0],
                   [0.0,    0.0,   1.0]], dtype=np.float32)
    xyz     = xyz @ Rz.T
    if use_normals:
        nxyz = nxyz @ Rz.T
    cephalo = cephalo @ Rz.T

    # Z-axis position noise, simulating MediaPipe monocular depth uncertainty
    z_noise_std = np.random.uniform(0.0, 0.02)
    xyz[:, 2] += np.random.randn(xyz.shape[0]).astype(np.float32) * z_noise_std

    # +/-5% scale noise (position only; normals are unit vectors, not scaled)
    scale_factor = np.random.uniform(0.95, 1.05)
    xyz     = xyz * scale_factor
    cephalo = cephalo * scale_factor

    if use_normals:
        return np.concatenate([xyz, nxyz], axis=-1), cephalo
    return xyz, cephalo

# ---------------------------------------------------------------------------
# Baseline 2: FaceMeshMLPCephaloNet
# ---------------------------------------------------------------------------

class FaceMeshMLPCephaloNet(nn.Module):
    """
    Baseline 2 (MLP): flattened FaceMesh points -> MLP -> 24 landmarks.
    No encoder or attention.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_landmarks: int = 24,
    ):
        super().__init__()
        input_dim = N_USE * 3  # 400 * 3
        self.model_config = {
            "model_type": "facemesh_mlp",
            "hidden_dim": int(hidden_dim),
            "num_landmarks": int(num_landmarks),
        }
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_landmarks * 3),
        )
        self.num_landmarks = int(num_landmarks)

    def forward(self, face_mesh: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) -> (B, 24, 3)"""
        b = face_mesh.shape[0]
        return self.net(face_mesh.view(b, -1)).view(b, self.num_landmarks, 3)


class PointNetCephaloNet(nn.Module):
    """
    Baseline 1 (PointNet): treats the FaceMesh points as a point set.
    per-point shared MLP -> global max pooling -> MLP head -> 24 landmarks.
    Permutation-invariant, unlike the flattened-MLP baseline.
    """
    def __init__(
        self,
        point_feat_dim: int = 128,
        global_feat_dim: int = 256,
        num_landmarks: int = 24,
    ):
        super().__init__()
        self.model_config = {
            "model_type": "pointnet",
            "point_feat_dim": int(point_feat_dim),
            "global_feat_dim": int(global_feat_dim),
            "num_landmarks": int(num_landmarks),
        }
        # Applied independently to each point.
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, point_feat_dim),
            nn.BatchNorm1d(point_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(point_feat_dim, global_feat_dim),
            nn.BatchNorm1d(global_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(global_feat_dim, global_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(global_feat_dim, global_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(global_feat_dim // 2, num_landmarks * 3),
        )
        self.num_landmarks = int(num_landmarks)

    def forward(self, face_mesh: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) -> (B, 24, 3)"""
        B, N, _ = face_mesh.shape
        # BatchNorm1d is applied over the flattened (B*N, C) batch.
        x = face_mesh.view(B * N, 3)
        x = self.point_mlp(x)                  # (B*N, global_feat_dim)
        x = x.view(B, N, -1)                   # (B, N, global_feat_dim)
        x = x.max(dim=1).values                # global max pool -> (B, global_feat_dim)
        return self.head(x).view(B, self.num_landmarks, 3)


class CrossAttentionCephaloNet(nn.Module):
    """
    Stage 0: 24 learnable landmark queries cross-attend to the FaceMesh
    points. Each query corresponds to one cephalometric landmark and
    learns which FaceMesh points are relevant to it.

      point encoder  : (B, N, 3)          -> (B, N, d_model)
      cross-attention: (B, 24, d_model) x (B, N, d_model) -> (B, 24, d_model)
      output MLP     : (B, 24, d_model)   -> (B, 24, 3)
    """
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        num_landmarks: int = 24,
        in_dim: int = 3,
    ):
        super().__init__()
        self.model_config = {
            "model_type": "cross_attn",
            "d_model": d_model,
            "num_heads": num_heads,
            "num_attn_layers": num_attn_layers,
            "num_landmarks": num_landmarks,
            "in_dim": in_dim,
        }
        self.num_landmarks = num_landmarks

        self.point_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.landmark_queries = nn.Parameter(torch.randn(num_landmarks, d_model))

        # Cross-attention layers (query=landmarks, key/value=FaceMesh points)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_attn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_attn_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, face_mesh: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) -> (B, 24, 3)"""
        B = face_mesh.shape[0]
        kv = self.point_encoder(face_mesh)              # (B, N, d_model)
        q = self.landmark_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 24, d_model)

        for attn, norm in zip(self.attn_layers, self.norms):
            out, _ = attn(q, kv, kv)
            q = norm(q + out)   # residual cross-attention

        return self.output_mlp(q)   # (B, 24, 3)


_STAGE0_MODELS = {
    "pointnet":   PointNetCephaloNet,
    "mlp":        FaceMeshMLPCephaloNet,
    "cross_attn": CrossAttentionCephaloNet,
}

def build_stage0(model_type: str = "mlp", in_dim: int = 3) -> nn.Module:
    cls = _STAGE0_MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose from {list(_STAGE0_MODELS)}")
    if model_type == "cross_attn":
        return cls(in_dim=in_dim)
    return cls()


class Stage1AttnV3RefinementNet(nn.Module):
    """
    Stage 1: refines the Stage 0 coarse estimate. Each FaceMesh point gets
    a learnable positional embedding (by index) so the network can learn
    which anatomical region it corresponds to, then self-attends among
    FaceMesh points before cross-attending from the 24 landmark queries.
    """
    def __init__(
        self,
        num_landmarks: int = 24,
        d_model: int = 128,
        num_heads: int = 4,
        num_fm_self_attn: int = 1,
        num_self_attn: int = 3,
        num_fm_points: int = 400,
        in_dim: int = 3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.fm_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable positional embedding per face mesh point (semantic region)
        self.fm_pos_embed = nn.Embedding(num_fm_points, d_model)
        self.coarse_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fm_self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_fm_self_attn)
        ])
        self.fm_self_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_fm_self_attn)])
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_self_attn)
        ])
        self.self_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_self_attn)])
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 3),
        )
        self.model_config = {
            "model_type": "stage1_attn_v3",
            "num_landmarks": num_landmarks,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_fm_self_attn": num_fm_self_attn,
            "num_self_attn": num_self_attn,
            "num_fm_points": num_fm_points,
            "in_dim": in_dim,
        }

    def forward(
        self,
        face_mesh: torch.Tensor,    # (B, N_fm, 3)
        coarse_lm: torch.Tensor,    # (B, 24, 3)
    ) -> torch.Tensor:
        N_fm = face_mesh.shape[1]
        pos = torch.arange(N_fm, device=face_mesh.device)
        kv = self.fm_encoder(face_mesh) + self.fm_pos_embed(pos)[None]  # (B, N_fm, d_model)
        for fm_sa, fm_sn in zip(self.fm_self_attn_layers, self.fm_self_norms):
            sa_out, _ = fm_sa(kv, kv, kv)
            kv = fm_sn(kv + sa_out)
        q = self.coarse_encoder(coarse_lm)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.cross_norm(q + out)
        for sa, sn in zip(self.self_attn_layers, self.self_norms):
            sa_out, _ = sa(out, out, out)
            out = sn(out + sa_out)
        feat  = torch.cat([out, q], dim=-1)
        delta = self.output_mlp(feat)
        return coarse_lm + delta


class SSMCorrector:
    """
    Stage 2: PCA-based statistical shape model.
    Learns the mean shape and principal components from training-set
    cephalo landmarks, then projects a prediction into PCA space and
    clips outlier coefficients. Sigma clipping is kept loose so
    asymmetric (jaw-deformity) cases aren't over-corrected toward the mean.
    """
    def __init__(self, n_components: int = 20, sigma_clip: float = 3.0):
        self.n_components = n_components
        self.sigma_clip   = sigma_clip
        self.mean_: Optional[np.ndarray]   = None  # (72,)
        self.U_:    Optional[np.ndarray]   = None  # (72, n_components)
        self.lam_:  Optional[np.ndarray]   = None  # (n_components,) variance

    def fit(self, landmarks: np.ndarray):
        """landmarks: (N_cases, 24, 3), normalized."""
        N = landmarks.shape[0]
        X = landmarks.reshape(N, -1).astype(np.float64)   # (N, 72)
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        n_comp = min(self.n_components, len(s))
        self.U_   = Vt[:n_comp].T                          # (72, n_comp)
        self.lam_ = (s[:n_comp] ** 2) / max(N - 1, 1)     # variance
        print(f"[SSM] fit: N={N}  n_components={n_comp}  "
              f"explained_var={((s[:n_comp]**2).sum()/(s**2).sum())*100:.1f}%")

    def correct(self, pred: np.ndarray) -> np.ndarray:
        """pred: (24, 3) normalized -> (24, 3) SSM-corrected."""
        assert self.mean_ is not None, "call SSMCorrector.fit() first"
        x    = pred.reshape(-1).astype(np.float64)
        x_c  = x - self.mean_
        alpha = self.U_.T @ x_c                             # project into PCA space
        sigma = np.sqrt(np.maximum(self.lam_, 1e-12))
        alpha_clipped = np.clip(alpha, -self.sigma_clip * sigma,
                                        self.sigma_clip * sigma)
        x_refined = self.mean_ + self.U_ @ alpha_clipped    # reconstruct
        return x_refined.reshape(24, 3).astype(np.float32)

    def save(self, path: Path):
        np.savez(str(path),
                 mean=self.mean_, U=self.U_, lam=self.lam_,
                 n_components=self.n_components, sigma_clip=self.sigma_clip)

    @classmethod
    def load(cls, path: Path) -> "SSMCorrector":
        d = np.load(str(path))
        obj = cls(int(d["n_components"]), float(d["sigma_clip"]))
        obj.mean_ = d["mean"]
        obj.U_    = d["U"]
        obj.lam_  = d["lam"]
        return obj


class FaceMeshCephaloDataset(Dataset):
    """
    FaceMesh points + 24 ground-truth cephalo landmarks. Never touches
    the STL mesh directly. When use_normals=True, face_mesh is (N, 6)
    [xyz | nxyz].
    """
    def __init__(
        self,
        root_dir: Path,
        csv_suffix: str = ".csv",
        augment: bool = False,
        use_normals: bool = False,
    ):
        self.augment = augment
        self.use_normals = use_normals
        self.samples: List[Tuple[Path, Path, str]] = []
        for case_dir in sorted([d for d in Path(root_dir).iterdir() if d.is_dir()]):
            sid = case_dir.name
            facemesh_csv = find_facemesh_csv(case_dir, sid)
            gt_csv = case_dir / f"{sid}{csv_suffix}"
            if facemesh_csv is not None and facemesh_csv.exists() and gt_csv.exists():
                self.samples.append((facemesh_csv, gt_csv, sid))
        print(f"[FaceMeshCephaloDataset] {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        facemesh_csv, gt_csv, _sid = self.samples[idx]

        gt_raw = parse_cephalo_csv(gt_csv)
        gt = {canonical_lm_name(k): v for k, v in gt_raw.items()}
        cephalo_ct = []
        for name in ALL_CEPHALO_NAMES:
            key = canonical_lm_name(name)
            if key not in gt:
                raise RuntimeError(f"{name} missing in {gt_csv}")
            cephalo_ct.append(gt[key])
        cephalo_ct = np.stack(cephalo_ct, axis=0).astype(np.float32)  # (24, 3) CT mm

        if _is_mp3d_csv(facemesh_csv):
            # mp3d format: bring GT into MediaPipe space via this case's
            # own Umeyama transform, then normalize both in that frame.
            mp3d, ct3d, _, ct_hit = load_mp3d_csv(facemesh_csv)
            face_mesh = mp3d[MP_USE_INDICES].astype(np.float32)

            hit_idx = np.where(ct_hit)[0]
            if len(hit_idx) < 6:
                raise RuntimeError(f"too few ct_hit points ({len(hit_idx)}): {facemesh_csv}")
            s, R, t = _umeyama(ct3d[hit_idx].astype(np.float64),
                                mp3d[hit_idx].astype(np.float64))
            cephalo = (s * (R @ cephalo_ct.T).T + t).astype(np.float32)

            face_n, cephalo_n, _, _ = _normalize_facemesh_sample(face_mesh, cephalo)
        else:
            face_mesh = load_face_mesh_use(facemesh_csv)
            face_n, cephalo_n, _, _ = _normalize_facemesh_sample(face_mesh, cephalo_ct)

        if self.use_normals:
            normals = compute_normals(face_n)                   # (N, 3)
            face_n  = np.concatenate([face_n, normals], axis=-1)  # (N, 6)

        if self.augment:
            face_n, cephalo_n = _augment_facemesh(face_n, cephalo_n)

        return {
            "face_mesh": torch.from_numpy(face_n),
            "cephalo":   torch.from_numpy(cephalo_n),
        }

def _safe_norm3(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,3) → (B,1) L2 norm, clamped to eps."""
    return torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)


def _angle_3pts(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Angle at vertex b formed by a-b-c, in degrees. Inputs: (B,3) → (B,)"""
    v1 = a - b; v2 = c - b
    cos_v = (v1 * v2).sum(-1) / (_safe_norm3(v1).squeeze(-1) * _safe_norm3(v2).squeeze(-1))
    return torch.acos(cos_v.clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / math.pi)


def compute_angle_loss_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Differentiable cephalometric angle loss between predicted and GT landmarks.
    pred, gt: (B, 24, 3) in normalized coordinate space.
    Angles preserved under translation + uniform scaling → valid in normalized space.
    Returns mean absolute angle error (degrees) averaged over SNA, SNB, ANB, SN-MP, FMA.
    """
    # --- landmark extraction ---
    S,  Na  = pred[:, 3],  pred[:, 4]
    A,  B_  = pred[:, 14], pred[:, 17]
    PoL, PoR = pred[:, 5], pred[:, 6]
    OrL, OrR = pred[:, 7], pred[:, 8]
    GoL, GoR = pred[:, 20], pred[:, 21]
    Me       = pred[:, 19]

    S_g,  Na_g  = gt[:, 3],  gt[:, 4]
    A_g,  B_g   = gt[:, 14], gt[:, 17]
    PoL_g, PoR_g = gt[:, 5], gt[:, 6]
    OrL_g, OrR_g = gt[:, 7], gt[:, 8]
    GoL_g, GoR_g = gt[:, 20], gt[:, 21]
    Me_g         = gt[:, 19]

    losses = []

    # SNA
    losses.append(torch.abs(_angle_3pts(S, Na, A) - _angle_3pts(S_g, Na_g, A_g)))
    # SNB
    losses.append(torch.abs(_angle_3pts(S, Na, B_) - _angle_3pts(S_g, Na_g, B_g)))
    # ANB = SNA - SNB (computed from above, keeps gradient flow)
    anb_p = _angle_3pts(S, Na, A) - _angle_3pts(S, Na, B_)
    anb_g = _angle_3pts(S_g, Na_g, A_g) - _angle_3pts(S_g, Na_g, B_g)
    losses.append(torch.abs(anb_p - anb_g))

    # SN-MP: acute angle between SN direction and GoMe direction
    SN  = Na - S;              GoMe  = Me - (GoL + GoR) * 0.5
    SN_g = Na_g - S_g;         GoMe_g = Me_g - (GoL_g + GoR_g) * 0.5
    cos_snmp   = (SN  * GoMe).sum(-1)  / (_safe_norm3(SN).squeeze(-1)  * _safe_norm3(GoMe).squeeze(-1))
    cos_snmp_g = (SN_g * GoMe_g).sum(-1) / (_safe_norm3(SN_g).squeeze(-1) * _safe_norm3(GoMe_g).squeeze(-1))
    snmp   = torch.acos(cos_snmp.abs().clamp(0, 1 - 1e-6))   * (180.0 / math.pi)
    snmp_g = torch.acos(cos_snmp_g.abs().clamp(0, 1 - 1e-6)) * (180.0 / math.pi)
    losses.append(torch.abs(snmp - snmp_g))

    # FMA: angle between Frankfort plane normal and mandibular plane direction
    Or  = (OrL + OrR) * 0.5;   Or_g = (OrL_g + OrR_g) * 0.5
    FH_n   = torch.linalg.cross(PoR - PoL,   Or - PoL,   dim=-1)
    FH_n_g = torch.linalg.cross(PoR_g - PoL_g, Or_g - PoL_g, dim=-1)
    FH_n   = FH_n   / _safe_norm3(FH_n)
    FH_n_g = FH_n_g / _safe_norm3(FH_n_g)
    GoMe_n   = GoMe   / _safe_norm3(GoMe)
    GoMe_n_g = GoMe_g / _safe_norm3(GoMe_g)
    sin_fma   = (FH_n   * GoMe_n).sum(-1).abs().clamp(0, 1 - 1e-6)
    sin_fma_g = (FH_n_g * GoMe_n_g).sum(-1).abs().clamp(0, 1 - 1e-6)
    fma   = torch.asin(sin_fma)   * (180.0 / math.pi)
    fma_g = torch.asin(sin_fma_g) * (180.0 / math.pi)
    losses.append(torch.abs(fma - fma_g))

    return torch.stack(losses, dim=0).mean()


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def train_model(
    train_root: Path,
    model_out_dir: Path,
    model_type: str = "mlp",
    csv_suffix: str = ".csv",
    augment: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    save_interval: int = 50,
    seed: int = 0,
    val_ratio: float = 0.15,
    patience: int = 50,
    device: Optional[str] = None,
    use_normals: bool = False,
    angle_loss_weight: float = 0.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = 6 if use_normals else 3

    # Loaded once with augment=False just to get the sample count for splitting.
    base_ds = FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=False,
                                     use_normals=use_normals)
    if len(base_ds) < 2:
        raise RuntimeError(f"Not enough samples: {len(base_ds)}")

    indices = list(range(len(base_ds)))
    np.random.RandomState(seed).shuffle(indices)
    n_val   = max(1, int(len(base_ds) * val_ratio))
    n_train = len(base_ds) - n_val
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = Subset(FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=augment,
                                             use_normals=use_normals), train_idx)
    val_ds   = Subset(FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=False,
                                             use_normals=use_normals), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"  train={n_train}  val={n_val}  model_type={model_type}  in_dim={in_dim}")

    model     = build_stage0(model_type, in_dim=in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.SmoothL1Loss()
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Train [{model_type}] ===")
    print(f"  root   : {train_root}")
    print(f"  device : {device}")
    print(f"  angle_loss_weight : {angle_loss_weight}")

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    train_log     = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running = 0.0
        for batch in train_loader:
            face_mesh = batch["face_mesh"].to(device).float()
            target    = batch["cephalo"].to(device).float()
            pred      = model(face_mesh)
            loss      = loss_fn(pred, target)
            if angle_loss_weight > 0.0:
                loss = loss + angle_loss_weight * compute_angle_loss_torch(pred, target) / 180.0
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item())
        train_loss = running / max(1, len(train_loader))

        # --- Val ---
        model.eval()
        val_running = 0.0
        val_mae_sum = 0.0
        val_n       = 0
        with torch.no_grad():
            for batch in val_loader:
                face_mesh = batch["face_mesh"].to(device).float()
                target    = batch["cephalo"].to(device).float()
                pred      = model(face_mesh)
                val_running += float(loss_fn(pred, target).item())
                val_mae_sum += float(torch.mean(torch.norm(pred - target, dim=-1)).item())
                val_n += 1
        val_loss = val_running / max(1, val_n)
        val_mae  = val_mae_sum  / max(1, val_n)
        train_log.append((epoch, train_loss, val_mae))

        print(f"[{epoch:04d}/{epochs}] train={train_loss:.5f}  val_loss={val_loss:.5f}  val_mae={val_mae:.4f}")

        # 定期保存
        if epoch % save_interval == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "model_type": model_type,
                 "model_config": model.model_config},
                model_out_dir / f"epoch_{epoch}.pth",
            )

        # Early stopping + best 保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            torch.save(
                {"epoch": epoch, "model": best_state, "model_type": model_type,
                 "model_config": model.model_config},
                model_out_dir / "best.pth",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}  best_val_loss={best_val_loss:.5f}")
                break

    import csv as _csv
    with open(model_out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_mae"])
        w.writerows(train_log)

    final_state = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"epoch": epoch, "model": final_state, "model_type": model_type,
         "model_config": model.model_config},
        model_out_dir / "final.pth",
    )
    print(f"[✓] Final model saved  best_val_loss={best_val_loss:.5f}")


def _load_checkpoint(checkpoint_path: Path, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "mlp")
    in_dim = ckpt.get("model_config", {}).get("in_dim", 3)
    model  = build_stage0(model_type, in_dim=in_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, device

# ---------------------------------------------------------------------------
# Stage 1 学習
# ---------------------------------------------------------------------------

def train_stage1(
    train_root: Path,
    stage0_checkpoint: Path,
    model_out_dir: Path,
    csv_suffix: str = ".csv",
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 8,
    num_workers: int = 4,
    patience: int = 30,
    seed: int = 0,
    val_ratio: float = 0.15,
    device: Optional[str] = None,
    d_model: int = 64,
    num_heads: int = 4,
    num_self_attn: int = 1,
    num_fm_self_attn: int = 1,
    use_normals: bool = False,
    snapshot_cycles: int = 0,
    hard_lm_weight: float = 1.0,
    angle_loss_weight: float = 0.0,
):
    """
    Trains Stage 1 to refine the (frozen) Stage 0 coarse estimate.

    snapshot_cycles > 0: trains for snapshot_cycles CosineAnnealingWarmRestarts
        cycles, saving stage1_snapshot_{i:02d}.pth at the end of each cycle;
        early stopping is disabled in this mode.
    hard_lm_weight: loss weight for landmarks not directly visible on the
        facial surface (Po/Ba/FO/Go/CoS), default 1.0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    stage0_model, _, _ = _load_checkpoint(stage0_checkpoint, device=device)
    stage0_model.eval()
    for p in stage0_model.parameters():
        p.requires_grad_(False)
    print(f"[Stage1] Stage0 loaded & frozen: {stage0_checkpoint}")

    in_dim = 6 if use_normals else 3

    # Split is decided on the un-augmented dataset; training uses augment=True.
    ds_noaug = FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=False,
                                      use_normals=use_normals)
    n = len(ds_noaug)
    indices = list(np.random.RandomState(seed).permutation(n))
    n_val   = max(1, int(n * val_ratio))
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    ds_aug = FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=True,
                                    use_normals=use_normals)
    train_ds = Subset(ds_aug,   train_idx)
    val_ds   = Subset(ds_noaug, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    stage1 = Stage1AttnV3RefinementNet(
        num_landmarks=len(ALL_CEPHALO_NAMES),
        d_model=d_model,
        num_heads=num_heads,
        num_fm_self_attn=num_fm_self_attn,
        num_self_attn=num_self_attn,
        num_fm_points=N_USE,
        in_dim=in_dim,
    ).to(device)
    print(f"  Stage1 model: attn_v3 (d={d_model}, h={num_heads}, fm_sa={num_fm_self_attn}, lm_sa={num_self_attn}, in_dim={in_dim})")
    optimizer = torch.optim.AdamW(stage1.parameters(), lr=lr, weight_decay=1e-4)

    # landmark-weighted L1 loss
    lm_w = torch.ones(len(ALL_CEPHALO_NAMES), 1, device=device)
    if hard_lm_weight != 1.0:
        for idx in _HARD_LM_IDX:
            lm_w[idx] = hard_lm_weight
    def _loss(pred, tgt):
        return (torch.abs(pred - tgt) * lm_w[None]).mean()

    # snapshot / scheduler
    use_snapshot = snapshot_cycles > 0
    if use_snapshot:
        cycle_len = max(1, epochs // snapshot_cycles)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cycle_len, T_mult=1)
        effective_patience = epochs  # early stopping disabled in snapshot mode
    else:
        scheduler = None
        effective_patience = patience

    print(f"=== Train Stage1 ===  train={len(train_idx)}  val={len(val_idx)}  "
          f"augment=True  snapshot={use_snapshot}  hlw={hard_lm_weight}  angle_loss_weight={angle_loss_weight}")

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    snap_count    = 0

    for epoch in range(1, epochs + 1):
        stage1.train()
        running = 0.0
        for batch in train_loader:
            fm     = batch["face_mesh"].to(device).float()   # (B, N, 3or6)
            target = batch["cephalo"].to(device).float()     # (B, 24, 3)
            with torch.no_grad():
                coarse = stage0_model(fm)                     # (B, 24, 3)
            refined = stage1(fm, coarse)                      # (B, 24, 3)
            loss = _loss(refined, target)
            if angle_loss_weight > 0.0:
                loss = loss + angle_loss_weight * compute_angle_loss_torch(refined, target) / 180.0
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
        if scheduler is not None:
            scheduler.step()
        train_loss = running / max(1, len(train_loader))

        stage1.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                fm     = batch["face_mesh"].to(device).float()
                target = batch["cephalo"].to(device).float()
                coarse = stage0_model(fm)
                refined = stage1(fm, coarse)
                val_running += float(_loss(refined, target).item())
                val_n += 1
        val_loss = val_running / max(1, val_n)
        print(f"  [{epoch:03d}/{epochs}] train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in stage1.state_dict().items()}
            no_improve    = 0
            torch.save({"model": best_state, "model_config": stage1.model_config},
                       model_out_dir / "stage1_best.pth")
        else:
            no_improve += 1
            if no_improve >= effective_patience:
                print(f"  Early stop at epoch {epoch}")
                break

        # Saved at the end of each cosine-annealing cycle.
        if use_snapshot and epoch % cycle_len == 0:
            snap_count += 1
            snap_path = model_out_dir / f"stage1_snapshot_{snap_count:02d}.pth"
            snap_state = {k: v.cpu().clone() for k, v in stage1.state_dict().items()}
            torch.save({"model": snap_state, "model_config": stage1.model_config}, snap_path)
            print(f"  [Snapshot {snap_count}] saved -> {snap_path.name}  val={val_loss:.5f}")

    print(f"[✓] Stage1 best_val_loss={best_val_loss:.5f}")


def fit_ssm(
    train_root: Path,
    stage0_checkpoint: Path,
    out_path: Path,
    csv_suffix: str = ".csv",
    n_components: int = 20,
    sigma_clip: float = 3.0,
    device: Optional[str] = None,
):
    """Fits the SSM on normalized ground-truth cephalo coordinates
    (uses GT directly, so it doesn't depend on prediction accuracy)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = FaceMeshCephaloDataset(train_root, csv_suffix=csv_suffix, augment=False)
    all_lm = [ds[i]["cephalo"].numpy() for i in range(len(ds))]  # each (24, 3), normalized
    landmarks = np.stack(all_lm, axis=0)            # (N, 24, 3)

    ssm = SSMCorrector(n_components=n_components, sigma_clip=sigma_clip)
    ssm.fit(landmarks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ssm.save(out_path)
    print(f"[✓] SSM saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="FaceMesh (400 pts) -> cross-attention -> 24 cephalo landmarks"
    )
    ap.add_argument("--coverage-txt", default=None,
                    help="Path to always_hit_indices.txt (auto-discovered if omitted)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_train_args(p):
        p.add_argument("--train-root",    required=True)
        p.add_argument("--model-out",     required=True)
        p.add_argument("--model-type", default="mlp",
                       choices=list(_STAGE0_MODELS),
                       help="mlp=Baseline 2 / pointnet=Baseline 1 / cross_attn=Stage 0")
        p.add_argument("--no-augment",    action="store_true",
                       help="Disable augmentation (enabled by default)")
        p.add_argument("--csv-suffix",    default=".csv")
        p.add_argument("--batch-size",    type=int,   default=8)
        p.add_argument("--num-workers",   type=int,   default=4)
        p.add_argument("--lr",            type=float, default=1e-3)
        p.add_argument("--weight-decay",  type=float, default=1e-4)
        p.add_argument("--epochs",        type=int,   default=200)
        p.add_argument("--save-interval", type=int,   default=50)
        p.add_argument("--patience",      type=int,   default=50)
        p.add_argument("--seed",          type=int,   default=0)

    p = sub.add_parser("train", help="Train a Stage 0 model on FaceMesh -> 24 landmarks")
    _add_train_args(p)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--normals", action="store_true",
                   help="Add surface normals to the input features ((N,3) -> (N,6))")
    p.add_argument("--angle-loss-weight", type=float, default=0.0,
                   help="Angle-loss weight (0=disabled, recommended: 0.5-2.0)")

    p = sub.add_parser("train-stage1", help="Train the Stage 1 refinement network")
    p.add_argument("--train-root",        required=True)
    p.add_argument("--stage0-checkpoint", required=True)
    p.add_argument("--model-out",         required=True)
    p.add_argument("--csv-suffix",        default=".csv")
    p.add_argument("--d-model",           type=int,   default=64)
    p.add_argument("--num-heads",         type=int,   default=4)
    p.add_argument("--num-self-attn",     type=int,   default=1)
    p.add_argument("--num-fm-self-attn",  type=int,   default=1)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--epochs",            type=int,   default=200)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--patience",          type=int,   default=40)
    p.add_argument("--val-ratio",         type=float, default=0.15)
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--normals",           action="store_true",
                   help="Add surface normals to the input features (must match Stage 0)")
    p.add_argument("--snapshot-cycles",   type=int,   default=0,
                   help="Number of CosineAnnealingWarmRestarts cycles (0=disabled, disables early stopping)")
    p.add_argument("--hard-lm-weight",    type=float, default=1.0,
                   help="Loss weight for hidden landmarks (Po/Ba/FO/Go/CoS), default 1.0")
    p.add_argument("--angle-loss-weight", type=float, default=0.0,
                   help="Angle-loss weight (0=disabled, recommended: 0.005)")

    p = sub.add_parser("fit-ssm", help="Fit the statistical shape model")
    p.add_argument("--train-root",        required=True)
    p.add_argument("--stage0-checkpoint", required=True)
    p.add_argument("--out",               required=True, help="Output path for ssm.npz")
    p.add_argument("--csv-suffix",    default=".csv")
    p.add_argument("--n-components",  type=int,   default=20)
    p.add_argument("--sigma-clip",    type=float, default=3.0)

    return ap


def main():
    args    = build_parser().parse_args()
    cov_txt = Path(args.coverage_txt) if args.coverage_txt else None
    _ensure_mp_use_indices(cov_txt)

    if args.cmd == "train":
        train_model(
            train_root=Path(args.train_root),
            model_out_dir=Path(args.model_out),
            model_type=args.model_type,
            csv_suffix=args.csv_suffix,
            augment=not args.no_augment,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            save_interval=args.save_interval,
            val_ratio=args.val_ratio,
            patience=args.patience,
            seed=args.seed,
            use_normals=getattr(args, "normals", False),
            angle_loss_weight=getattr(args, "angle_loss_weight", 0.0),
        )

    elif args.cmd == "train-stage1":
        train_stage1(
            train_root=Path(args.train_root),
            stage0_checkpoint=Path(args.stage0_checkpoint),
            model_out_dir=Path(args.model_out),
            csv_suffix=args.csv_suffix,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patience=args.patience,
            val_ratio=args.val_ratio,
            seed=args.seed,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_self_attn=args.num_self_attn,
            num_fm_self_attn=args.num_fm_self_attn,
            use_normals=getattr(args, "normals", False),
            snapshot_cycles=getattr(args, "snapshot_cycles", 0),
            hard_lm_weight=getattr(args, "hard_lm_weight", 1.0),
            angle_loss_weight=getattr(args, "angle_loss_weight", 0.0),
        )

    elif args.cmd == "fit-ssm":
        fit_ssm(
            train_root=Path(args.train_root),
            stage0_checkpoint=Path(args.stage0_checkpoint),
            out_path=Path(args.out),
            csv_suffix=args.csv_suffix,
            n_components=args.n_components,
            sigma_clip=args.sigma_clip,
        )


if __name__ == "__main__":
    main()
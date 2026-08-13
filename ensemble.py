"""
Ensemble evaluation: average predictions from multiple Stage1 models,
with optional left-right-flip test-time augmentation.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cephalo import (
    Stage1AttnV3RefinementNet, FLIP_PAIRS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Stage1LegacyNet(torch.nn.Module):
    """Backward compatibility for older Stage1 checkpoints without fm_pos_embed/fm_self_attn."""
    def __init__(self, num_landmarks=24, d_model=128, num_heads=4, num_self_attn=3, **_):
        super().__init__()
        self.fm_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, d_model), torch.nn.LayerNorm(d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        self.coarse_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, d_model), torch.nn.LayerNorm(d_model), torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model))
        self.cross_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.norm = torch.nn.LayerNorm(d_model)
        self.self_attn_layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_self_attn)])
        self.self_norms = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(num_self_attn)])
        self.output_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model), torch.nn.GELU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(d_model, 3))

    def forward(self, face_mesh, coarse_lm):
        kv = self.fm_encoder(face_mesh)
        q  = self.coarse_encoder(coarse_lm)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.norm(q + out)
        for sa, sn in zip(self.self_attn_layers, self.self_norms):
            sa_out, _ = sa(out, out, out)
            out = sn(out + sa_out)
        feat  = torch.cat([out, q], dim=-1)
        delta = self.output_mlp(feat)
        return coarse_lm + delta


def _flip_unflip(pred: np.ndarray) -> np.ndarray:
    """Un-flips a prediction from a flipped face mesh: swaps L/R pairs and negates x."""
    out = pred.copy()
    for i, j in FLIP_PAIRS:
        out[i] = pred[j]
        out[j] = pred[i]
    out[:, 0] *= -1
    return out


def load_stage1(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["model_config"]
    sd   = ckpt["model"]
    # Older checkpoints don't have fm_pos_embed.
    if "fm_pos_embed.weight" not in sd:
        m = _Stage1LegacyNet(**{k: v for k, v in cfg.items() if k != "model_type"}).to(DEVICE)
    else:
        m = Stage1AttnV3RefinementNet(**{k: v for k, v in cfg.items() if k != "model_type"}).to(DEVICE)
    m.load_state_dict(sd)
    m.eval()
    return m

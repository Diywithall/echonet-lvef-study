"""Core library: segmentation, volume estimation, cardiac-cycle detection, EF models.

Everything downstream (calibration, test evaluation, statistics, figures) imports from here.
"""
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from scipy import ndimage, signal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "dataset", "Echocardiogram Dataset", "EchoNet-Dynamic")
VIDEOS = os.path.join(DATA, "Videos")
WEIGHTS_SEG = os.path.join(ROOT, "backend", "models", "deeplabv3_resnet50_random.pt")
WEIGHTS_EF = os.path.join(ROOT, "backend", "models", "r2plus1d_18_32_2_pretrained.pt")

MEAN = np.array([33.94153, 33.881313, 33.85676], dtype=np.float32)
STD = np.array([50.211212, 50.17746, 50.145348], dtype=np.float32)
N_DISKS = 20
CLIP_FRAMES, CLIP_PERIOD = 32, 2
CLIP_SPAN = CLIP_FRAMES * CLIP_PERIOD  # 64 raw frames

torch.set_num_threads(os.cpu_count() or 1)


# ----------------------------------------------------------------------------- IO
def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    return np.stack(frames).astype(np.uint8)  # T,H,W,3


def _load_state(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model"):
        if isinstance(st, dict) and k in st:
            st = st[k]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in st.items()}


def load_seg_model():
    m = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)
    m.classifier[-1] = nn.Conv2d(m.classifier[-1].in_channels, 1, kernel_size=1)
    m.load_state_dict(_load_state(WEIGHTS_SEG))
    return m.eval()


def load_ef_model():
    m = torchvision.models.video.r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m.load_state_dict(_load_state(WEIGHTS_EF))
    return m.eval()


# ----------------------------------------------------------------------- segmentation
def _norm(frames):
    return ((frames.astype(np.float32) - MEAN) / STD)


@torch.inference_mode()
def segment(model, frames, batch=64):
    """frames: T,H,W,3 uint8 -> boolean masks T,H,W (logit > 0, as in the official code)."""
    x = torch.from_numpy(_norm(frames)).permute(0, 3, 1, 2).contiguous()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch])["out"][:, 0] > 0)
    return torch.cat(out).numpy()


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def mask_geometry(mask):
    """Return (area_px, long_axis_len_px, disks_volume_px3, arealength_volume_px3).

    Long axis = extent of the mask along its first principal axis. The disk volume divides
    that axis into N_DISKS slices; each disk diameter is the mean mask width in the slice
    (pixel count / slice thickness): V = sum(pi/4 * d_i^2 * h).
    """
    ys, xs = np.nonzero(mask)
    area = float(len(ys))
    if area < 10:
        return 0.0, 0.0, 0.0, 0.0
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    t = (pts - c) @ ax
    lo, hi = t.min(), t.max() + 1.0
    L = hi - lo
    h = L / N_DISKS
    idx = np.minimum(((t - lo) / h).astype(int), N_DISKS - 1)
    counts = np.bincount(idx, minlength=N_DISKS).astype(np.float64)
    d = counts / h
    v_disks = float(np.sum(np.pi / 4.0 * d ** 2 * h))
    v_al = 8.0 * area ** 2 / (3.0 * np.pi * L)
    return area, float(L), v_disks, float(v_al)


def curves_from_masks(masks):
    """Per-frame geometry arrays: area, length, disks volume, area-length volume (px units)."""
    g = np.array([mask_geometry(largest_component(m)) for m in masks])
    return g[:, 0], g[:, 1], g[:, 2], g[:, 3]


# ------------------------------------------------------------------ cycle detection
def detect_cycles(area, fps, smooth=True):
    """Detect end-systolic minima and end-diastolic maxima from the LV-area curve.

    - Savitzky-Golay smoothing (window ~ 0.1 s, odd, poly 2)
    - ES = prominent minima of area (min distance 0.30 s, prominence 0.25 * trimmed range)
    - one beat = consecutive ES pair; ED = frame of maximal area between them.
      Cycles truncated at the video boundaries (no ES on both sides) are discarded.
    Returns dict(es=[...], beats=[(ed, es_prev, es_next)...], smoothed=array).
    """
    a = np.asarray(area, dtype=np.float64)
    n = len(a)
    if smooth and n >= 9:
        w = int(max(5, round(0.10 * fps) // 2 * 2 + 1))
        w = min(w, n - (1 - n % 2))
        if w % 2 == 0:
            w -= 1
        a_s = signal.savgol_filter(a, w, 2) if w >= 5 else a
    else:
        a_s = a
    srt = np.sort(a_s)
    lo, hi = srt[int(0.05 * (n - 1))], srt[int(0.95 * (n - 1))]
    rng = max(hi - lo, 1e-6)
    dist = max(3, int(round(0.30 * fps)))
    es, _ = signal.find_peaks(-a_s, distance=dist, prominence=0.25 * rng)
    beats = []
    for i in range(len(es) - 1):
        seg = a_s[es[i]:es[i + 1] + 1]
        ed = es[i] + int(np.argmax(seg))
        beats.append((int(ed), int(es[i]), int(es[i + 1])))
    return {"es": es.astype(int), "beats": beats, "smoothed": a_s}


# --------------------------------------------------------------------- EF regression
def clip_starts(n_frames, n_clips):
    if n_frames <= CLIP_SPAN:
        return [0]
    mx = n_frames - CLIP_SPAN
    return sorted({int(round(mx * i / max(n_clips - 1, 1))) for i in range(n_clips)})


def make_clip(frames, start):
    n = len(frames)
    idx = np.minimum(start + CLIP_PERIOD * np.arange(CLIP_FRAMES), n - 1)
    clip = _norm(frames[idx])  # T,H,W,3
    return torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()  # 3,T,H,W


@torch.inference_mode()
def predict_ef_clips(model, frames, starts):
    x = torch.stack([make_clip(frames, s) for s in starts])
    return model(x).squeeze(1).numpy()


# ------------------------------------------------------------------------ metrics
def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def trace_mask(trace_rows, shape=(112, 112)):
    """Polygon mask from the VolumeTracings line segments (same construction as official code)."""
    import skimage.draw
    t = np.asarray(trace_rows, dtype=float)
    x = np.concatenate((t[1:, 0], np.flip(t[1:, 2])))
    y = np.concatenate((t[1:, 1], np.flip(t[1:, 3])))
    r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), shape)
    m = np.zeros(shape, bool)
    m[r, c] = True
    return m

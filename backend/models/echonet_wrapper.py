"""
echonet_wrapper.py
------------------
EchoNet-Dynamic inference wrapper.

Only the R(2+1)D-18 EF-regression model is used.
DeepLabV3 / segmentation / LV-area beat-detection / volumetrics are removed.

Beat-variability is replaced by multi-clip sliding-window EF:
  n_clips evenly-spaced 64-frame windows → per-clip EF → stats + MAD outlier flags.
"""

import os
import sys
import time
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

# ── Thread count: use all logical cores reported by the OS ─────────────────
_CPU_COUNT = os.cpu_count() or 1
torch.set_num_threads(_CPU_COUNT)

# ── Official EchoNet-Dynamic dataset normalisation (0-255 float range) ─────
_MEAN = torch.tensor([33.94153, 33.881313, 33.85676]).view(3, 1, 1, 1)
_STD  = torch.tensor([50.211212, 50.17746, 50.145348]).view(3, 1, 1, 1)

# Window constants matching the official EchoNet repo
TARGET_FRAMES = 32
PERIOD        = 2
WINDOW_SIZE   = TARGET_FRAMES * PERIOD   # 64 raw frames → 32 sampled frames


class EchoNetAnalyzer:
    """
    Wraps the pretrained R(2+1)D-18 EF-regression model.
    All inference runs under torch.inference_mode().
    """

    def __init__(self, ef_weights_path: str, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(
            f"[EchoNetAnalyzer] device={self.device} | "
            f"torch_threads={torch.get_num_threads()} | "
            f"logical_cores={_CPU_COUNT}"
        )

        # R(2+1)D-18 — single linear head for EF regression
        self.ef_model = r2plus1d_18(pretrained=False)
        self.ef_model.fc = nn.Linear(self.ef_model.fc.in_features, 1)
        self._load_weights(self.ef_model, ef_weights_path)
        self.ef_model.to(self.device).eval()

    # ── Weight loading ─────────────────────────────────────────────────────

    def _load_weights(self, model: nn.Module, path: str) -> None:
        if not os.path.exists(path):
            print(f"[EchoNetAnalyzer] WARNING: weights not found at {path}")
            return

        # Register numpy scalar types so PyTorch unpickler accepts them
        safe_globals = []
        for mod_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
            try:
                m = importlib.import_module(mod_name)
                safe_globals.append(m.scalar)
            except (AttributeError, ImportError):
                pass
        if safe_globals:
            try:
                torch.serialization.add_safe_globals(safe_globals)
            except AttributeError:
                pass  # older torch — weights_only=False handles it

        state = torch.load(path, map_location=self.device, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

        # Strip DataParallel 'module.' prefix if present
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[EchoNetAnalyzer] Loaded weights from {path}")

    # ── Video I/O ──────────────────────────────────────────────────────────

    @staticmethod
    def _read_frames(video_path: str) -> list:
        """Read every frame from *video_path*, resize to 112×112 RGB. Returns list of arrays."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()
        return frames

    # ── Preprocessing ──────────────────────────────────────────────────────

    def preprocess_video(self, frames: list, start_frame: int = 0) -> torch.Tensor:
        """
        Build a [1, 3, 32, 112, 112] float tensor from *frames*.

        Official EchoNet-Dynamic sampling:
          • 64-frame contiguous window starting at *start_frame*
          • Every 2nd frame (stride-2) → 32 frames
          • Short videos (or windows extending past the end) are padded by
            repeating the last frame.

        Args:
            frames      : list of HxWx3 uint8 ndarrays (full video)
            start_frame : index of the first frame in the 64-frame window

        Returns:
            Tensor [1, 3, 32, 112, 112], normalised to EchoNet-Dynamic stats.
        """
        if not frames:
            return None

        # Pad video in-place until the window fits
        padded = list(frames)
        required = start_frame + WINDOW_SIZE
        while len(padded) < required:
            padded.append(padded[-1])

        indices = start_frame + PERIOD * np.arange(TARGET_FRAMES)  # shape (32,)
        sampled = [padded[i] for i in indices]

        # [T, H, W, C] → [C, T, H, W]
        arr = np.array(sampled, dtype=np.float32).transpose(3, 0, 1, 2)
        tensor = torch.from_numpy(arr)
        tensor = (tensor - _MEAN) / _STD
        return tensor.unsqueeze(0)   # [1, C, T, H, W]

    # ── Inference ──────────────────────────────────────────────────────────

    def _infer_ef(self, tensor: torch.Tensor) -> float:
        """Run R(2+1)D-18 on tensor [1,C,T,H,W]. Returns scalar EF (float)."""
        with torch.inference_mode():
            out = self.ef_model(tensor.to(self.device))
        return float(out.item())

    # ── Multi-clip sliding-window EF ───────────────────────────────────────

    def analyze_multiclip(self, video_path: str, n_clips: int = 5):
        """
        Read all frames once, sample *n_clips* evenly-spaced 64-frame windows,
        run EF regression on each.

        If the video is shorter than 64 frames, a single padded clip from
        frame 0 is used regardless of *n_clips*.

        Returns:
            (frames, clip_ef_values)
            frames          : list of raw frame arrays (reused by analyze())
            clip_ef_values  : list of float EF values, one per clip
        """
        frames = self._read_frames(video_path)
        if not frames:
            return None, []

        total = len(frames)

        if total < WINDOW_SIZE:
            starts = [0]
        else:
            max_start = total - WINDOW_SIZE
            # n evenly-spaced integers in [0, max_start]
            starts = sorted({
                int(round(max_start * i / max(n_clips - 1, 1)))
                for i in range(n_clips)
            })

        clip_efs = []
        for s in starts:
            tensor = self.preprocess_video(frames, start_frame=s)
            clip_efs.append(round(self._infer_ef(tensor), 1))

        return frames, clip_efs

    # ── Clinical interpretation ────────────────────────────────────────────

    def get_clinical_interpretation(self, ef: float):
        """
        Returns (diagnosis_label, explanation_string).
        Thresholds: ≥55 Normal, 40-54 Mildly Reduced, <40 Severely Reduced.
        """
        disclaimer = (
            "LVEF estimation only — not diagnostic of underlying etiology. "
            "Research prototype, not for clinical use."
        )
        if ef >= 55:
            return (
                "Normal Left Ventricular Function",
                f"The ejection fraction is within the normal range (≥55%), "
                f"indicating normal left ventricular systolic function. {disclaimer}",
            )
        elif ef >= 40:
            return (
                "Mildly Reduced Left Ventricular Function",
                f"The ejection fraction is mildly reduced (40–54%), "
                f"indicating mild left ventricular systolic impairment. {disclaimer}",
            )
        else:
            return (
                "Severely Reduced Left Ventricular Function",
                f"The ejection fraction is severely reduced (<40%), "
                f"indicating severe left ventricular systolic impairment. {disclaimer}",
            )

    # ── Main entry point ───────────────────────────────────────────────────

    def analyze(self, video_path: str, n_clips: int = 5) -> dict:
        """
        Full pipeline for one video.

        Returns a dict with:
          ef               – headline LVEF (mean of clip EFs)
          center_clip_ef   – single centered-clip prediction (ablation baseline)
          diagnosis        – functional category label
          explanation      – one-paragraph clinical note + disclaimer
          clip_ef_values   – list of per-clip EF floats
          mean_ef          – same as ef (explicit alias)
          median_ef
          ef_std
          ef_range         – {min, max, range}
          num_clips
          outlier_clips    – indices where |EF - median| > 2.5 × MAD
                             (empty when MAD == 0; data never dropped)
          consistency      – HIGH / MODERATE / LOW  (engineering thresholds)
          inference_time_sec
        """
        t0 = time.time()

        frames, clip_efs = self.analyze_multiclip(video_path, n_clips=n_clips)

        if frames is None or len(clip_efs) == 0:
            return {
                "error": "Could not read video — file may be empty or corrupted."
            }

        arr = np.array(clip_efs, dtype=np.float64)

        # ── Headline EF ───────────────────────────────────────────────────
        ef = round(float(arr.mean()), 1)

        # ── Center-clip EF (ablation) ──────────────────────────────────────
        total = len(frames)
        center_start = 0 if total < WINDOW_SIZE else (total - WINDOW_SIZE) // 2
        center_tensor = self.preprocess_video(frames, start_frame=center_start)
        center_clip_ef = round(self._infer_ef(center_tensor), 1)

        # ── Summary statistics ─────────────────────────────────────────────
        median_ef = round(float(np.median(arr)), 1)
        ef_std    = round(float(np.std(arr)), 1)
        min_ef    = round(float(arr.min()), 1)
        max_ef    = round(float(arr.max()), 1)
        ef_range  = {
            "min":   min_ef,
            "max":   max_ef,
            "range": round(max_ef - min_ef, 1),
        }

        # ── MAD outlier detection ──────────────────────────────────────────
        mad = float(np.median(np.abs(arr - np.median(arr))))
        if mad == 0.0:
            outlier_clips: list[int] = []
        else:
            threshold = 2.5 * mad
            outlier_clips = [
                int(i) for i, v in enumerate(clip_efs)
                if abs(v - median_ef) > threshold
            ]

        # ── Consistency label (documented engineering thresholds) ──────────
        # HIGH: SD < 3 pp,  MODERATE: 3–6 pp,  LOW: > 6 pp
        if ef_std < 3.0:
            consistency = "HIGH"
        elif ef_std <= 6.0:
            consistency = "MODERATE"
        else:
            consistency = "LOW"

        diagnosis, explanation = self.get_clinical_interpretation(ef)

        return {
            "ef":                 ef,
            "center_clip_ef":     center_clip_ef,
            "diagnosis":          diagnosis,
            "explanation":        explanation,
            "clip_ef_values":     clip_efs,
            "mean_ef":            ef,          # explicit alias
            "median_ef":          median_ef,
            "ef_std":             ef_std,
            "ef_range":           ef_range,
            "num_clips":          len(clip_efs),
            "outlier_clips":      outlier_clips,
            "consistency":        consistency,
            "inference_time_sec": round(time.time() - t0, 2),
        }

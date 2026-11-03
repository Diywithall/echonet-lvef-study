"""
backend/tests/smoke_test.py
----------------------------
Quick sanity check: run analyze() on one EchoNet-Dynamic video and
print the full payload + timing.

Run from the backend/ directory:
    python tests/smoke_test.py
"""

import sys
import os
import json
import time

# Make sure 'backend/' is on the path so 'models' resolves correctly
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_DIR)

from models.echonet_wrapper import EchoNetAnalyzer

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

EF_WEIGHTS = os.path.join(BACKEND_DIR, "models", "r2plus1d_18_32_2_pretrained.pt")

VIDEOS_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "Echocardiogram Dataset", "EchoNet-Dynamic", "Videos"
)

# ── Pick first available AVI ───────────────────────────────────────────────
def find_first_video(directory: str) -> str:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Videos directory not found: {directory!r}")
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".avi"):
            return os.path.join(directory, name)
    raise FileNotFoundError(f"No .avi files found in {directory!r}")


def main():
    print("=" * 60)
    print("EchoScan AI Pro — Smoke Test")
    print("=" * 60)

    video_path = find_first_video(VIDEOS_DIR)
    print(f"\nTest video : {os.path.basename(video_path)}")
    print(f"Full path  : {video_path}\n")

    print("Loading model …")
    t_load = time.time()
    analyzer = EchoNetAnalyzer(ef_weights_path=EF_WEIGHTS)
    print(f"Model ready in {time.time() - t_load:.2f}s\n")

    print("Running analyze() …")
    result = analyzer.analyze(video_path, n_clips=5)

    print("\n" + "=" * 60)
    print("RESULT PAYLOAD")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"  Headline EF        : {result['ef']} %")
    print(f"  Center-clip EF     : {result['center_clip_ef']} %")
    print(f"  Clip EF values     : {result['clip_ef_values']}")
    print(f"  Median EF          : {result['median_ef']} %")
    print(f"  EF Std Dev         : {result['ef_std']} pp")
    print(f"  EF Range           : {result['ef_range']}")
    print(f"  Num clips          : {result['num_clips']}")
    print(f"  Outlier clips      : {result['outlier_clips']}")
    print(f"  Consistency        : {result['consistency']}")
    print(f"  Diagnosis          : {result['diagnosis']}")
    print(f"  Inference time     : {result['inference_time_sec']} s")
    print()

    # Basic assertions
    assert isinstance(result["ef"], float), "ef must be float"
    assert 0 < result["ef"] < 100, f"ef out of range: {result['ef']}"
    assert result["num_clips"] > 0, "no clips produced"
    assert len(result["clip_ef_values"]) == result["num_clips"]
    assert result["consistency"] in ("HIGH", "MODERATE", "LOW")

    print("All assertions passed. Smoke test OK.")


if __name__ == "__main__":
    main()

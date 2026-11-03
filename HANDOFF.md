# Handoff: EchoNet beat-to-beat LV function study (manuscript revision)

## Goal
Address the reviewer report (15 major + 6 minor comments, "Major Revision") on the manuscript about
beat-to-beat LVEF / EDV / ESV / SV from apical-4-chamber echo using pretrained EchoNet-Dynamic
(DeepLabV3-ResNet50 segmentation + R(2+1)D-18 EF). No manuscript file existed on disk: a revised
manuscript, figures, tables and a point-by-point response letter must be written from real results.
Never fabricate numbers; moderate the "lightweight"/POCUS claims (reviewer comments 6, 9, 12).

## Setup on a new machine
- Python 3.12 with torch, torchvision, opencv-python, numpy, pandas, scipy, scikit-image, scikit-learn,
  matplotlib, python-docx, statsmodels (no GPU needed; everything ran on a CPU laptop).
- Weights in `backend/models/*.pt` (tracked with Git LFS: run `git lfs install` before cloning).
- Dataset is NOT in git: place EchoNet-Dynamic at
  `dataset/Echocardiogram Dataset/EchoNet-Dynamic/{Videos/,FileList.csv,VolumeTracings.csv}`.

## Code (study/)
- `core.py`: segmentation, mask geometry (PCA long axis, 20-disk volume, area-length), cycle detection
  (Savitzky-Golay + find_peaks on LV area), multi-clip R(2+1)D EF, Dice.
- `run_calibration.py`: pixel->mL scale fitted on the VAL split only (400 videos, expert ED/ES frames).
  Output `outputs/calibration.json` (implied pixel 0.129 cm). Raw px->mL correlation is weak
  (r~0.31 ED / 0.51 ES vs clinical biplane reference), so report area-length vs disks honestly.
- `run_test.py [N]`: TEST split, seeded random order, resumable. Writes `outputs/test_videos.csv`
  and per-video curves in `outputs/curves/`. Target 300 videos (~20 s/video on CPU; was still running
  when this was pushed, so outputs may be partial; re-running continues where it stopped).

## Findings so far
- VAL Dice: ED 0.924, ES 0.898 (matches published ~0.92).
- Reference EDV/ESV/EF in FileList.csv are clinical (biplane) values; VolumeTracings are A4C ED/ES traces.
- Existing app wrapper (`backend/models/echonet_wrapper.py`) only does R(2+1)D EF, and its "beats" are
  sliding windows, not real cycles; `backend/evaluation/evaluate.py` is out of sync with it.

## Remaining plan
1. Finish/extend `run_test.py`; analysis script: MAE/RMSE/r, Bland-Altman (bias, LoA + CIs, proportional
   bias), ICC(2,1)/CCC with cluster bootstrap, SV = EDV - ESV validation, ED/ES frame error vs expert,
   beat-level SV CV (variance components; noise-floor test via input perturbation), subgroup/robustness
   (HR, RR irregularity, image-quality proxies, synthetic degradations), ablations (center vs multi-clip,
   area-length vs disks, fixed-FOV vs calibrated, fusion).
2. Efficiency table: params, size, FLOPs, latency, peak memory (DeepLabV3-R50, R(2+1)D-18 vs
   LRASPP-MobileNetV3, r3d_18 etc.; profiling only for untrained comparators).
3. Figures (architecture diagram, overlays, scatter, Bland-Altman, ED/ES detection, efficiency).
4. Revised manuscript (.docx) with complete verified references + response-to-reviewers letter.
   Novelty must be reframed as a reliability-aware analysis layer around pretrained EchoNet.

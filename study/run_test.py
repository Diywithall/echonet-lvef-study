"""Main TEST-split run. Random (seeded) order, resumable, one row + one curve file per video.

Usage: py -3.12 study/run_test.py [max_videos]
Stored per video (raw, before any calibration): area / length / disk volume / area-length volume
curves for every frame, five-clip R(2+1)D EF predictions, Dice at the expert ED/ES frames.
"""
import sys, os, json, time
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(__file__))
import core

MAX_V = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
OUT = os.path.join(os.path.dirname(__file__), "outputs")
CURVES = os.path.join(OUT, "curves")
os.makedirs(CURVES, exist_ok=True)
CSV = os.path.join(OUT, "test_videos.csv")
N_CLIPS = 5
N_FIG = 12

fl = pd.read_csv(os.path.join(core.DATA, "FileList.csv"))
vt = pd.read_csv(os.path.join(core.DATA, "VolumeTracings.csv"))
vt_g = {k: v for k, v in vt.groupby("FileName")}
test = fl[fl.Split == "TEST"].sample(frac=1.0, random_state=0).reset_index(drop=True)
done = set(pd.read_csv(CSV).FileName) if os.path.exists(CSV) else set()

seg, ef = core.load_seg_model(), core.load_ef_model()
t_start, n_done = time.time(), 0
for i, r in test.iterrows():
    if n_done >= MAX_V:
        break
    if r.FileName in done:
        continue
    g = vt_g.get(r.FileName + ".avi")
    if g is None or g.Frame.nunique() < 2:
        continue
    fr = core.read_video(os.path.join(core.VIDEOS, r.FileName + ".avi"))
    if fr is None:
        continue
    T = len(fr)
    t0 = time.time()
    masks = core.segment(seg, fr)
    t_seg = time.time() - t0
    area, L, vd, va = core.curves_from_masks(masks)
    t0 = time.time()
    starts = core.clip_starts(T, N_CLIPS)
    efs = core.predict_ef_clips(ef, fr, starts)
    t_ef = time.time() - t0
    center_start = 0 if T <= core.CLIP_SPAN else (T - core.CLIP_SPAN) // 2
    ef_center = float(core.predict_ef_clips(ef, fr, [center_start])[0])

    tm = {int(f): core.trace_mask(g[g.Frame == f][["X1", "Y1", "X2", "Y2"]].values) for f in g.Frame.unique()}
    fs = sorted(tm, key=lambda f: tm[f].sum())
    es_f, ed_f = fs[0], fs[-1]
    d_ed = core.dice(core.largest_component(masks[ed_f]), tm[ed_f])
    d_es = core.dice(core.largest_component(masks[es_f]), tm[es_f])
    gray = cv2.cvtColor(fr[T // 2], cv2.COLOR_RGB2GRAY)
    row = dict(FileName=r.FileName, FPS=r.FPS, NumberOfFrames=T, EF_ref=r.EF, EDV_ref=r.EDV, ESV_ref=r.ESV,
               ef_clips=json.dumps([round(float(x), 3) for x in efs]), ef_center=ef_center, ef_mean=float(np.mean(efs)),
               ed_frame=ed_f, es_frame=es_f, dice_ED=d_ed, dice_ES=d_es, t_seg=t_seg, t_ef=t_ef,
               sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()), brightness=float(gray.mean()), contrast=float(gray.std()))
    np.savez_compressed(os.path.join(CURVES, r.FileName + ".npz"), area=area.astype(np.float32), length=L.astype(np.float32),
                        vdisk=vd.astype(np.float32), val=va.astype(np.float32))
    if n_done < N_FIG:
        np.savez_compressed(os.path.join(CURVES, "fig_" + r.FileName + ".npz"),
                            frames=fr[[ed_f, es_f]], pred=masks[[ed_f, es_f]], ref=np.stack([tm[ed_f], tm[es_f]]))
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
    n_done += 1
    el = time.time() - t_start
    print(f"[{n_done}] {r.FileName} EFref={r.EF:.1f} EFmean={np.mean(efs):.1f} dice={d_ed:.2f}/{d_es:.2f} "
          f"seg={t_seg:.1f}s ef={t_ef:.1f}s | elapsed {el/60:.1f} min", flush=True)

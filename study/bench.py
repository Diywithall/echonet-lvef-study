import sys, time, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import core

fl = pd.read_csv(os.path.join(core.DATA, "FileList.csv"))
vt = pd.read_csv(os.path.join(core.DATA, "VolumeTracings.csv"))
test = fl[fl.Split == "TEST"].sample(4, random_state=1)

t0 = time.time(); seg = core.load_seg_model(); ef = core.load_ef_model(); print("load", round(time.time() - t0, 1))
for _, r in test.iterrows():
    fr = core.read_video(os.path.join(core.VIDEOS, r.FileName + ".avi"))
    t0 = time.time(); m = core.segment(seg, fr); ts = time.time() - t0
    A, L, Vd, Val = core.curves_from_masks(m)
    cyc = core.detect_cycles(A, r.FPS)
    t0 = time.time(); e = core.predict_ef_clips(ef, fr, core.clip_starts(len(fr), 5)); te = time.time() - t0
    tr = vt[vt.FileName == r.FileName + ".avi"]
    frames_ = tr.Frame.unique()
    ds = []
    for f in frames_:
        mk = core.trace_mask(tr[tr.Frame == f][["X1", "Y1", "X2", "Y2"]].values)
        ds.append(round(core.dice(m[f], mk), 3))
    print(r.FileName, "T", len(fr), "seg", round(ts, 1), "s | ef5", round(te, 1), "s | refEF", round(r.EF, 1), "R2+1D", np.round(e, 1),
          "| beats", len(cyc["beats"]), "ES", cyc["es"][:6], "| expert frames", list(frames_), "dice", ds,
          "| Vdisk ED/ES px", int(Vd.max()), int(Vd.min()), "refEDV/ESV", round(r.EDV, 1), round(r.ESV, 1))

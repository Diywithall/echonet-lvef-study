"""Pixel->mL calibration fitted on the VAL split only (expert ED/ES frames), never on TEST.

For each VAL video: segment the two expert-labelled frames, compute geometry (area, length,
single-plane disk volume, area-length volume) and compare with the reference EDV/ESV
distributed with EchoNet-Dynamic. Fit a global scale a: V_ml = a * V_px (through the origin;
physical meaning a = pixel_size_cm^3). An intercept variant is also stored for ablation.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import core

N_CAL = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

fl = pd.read_csv(os.path.join(core.DATA, "FileList.csv"))
vt = pd.read_csv(os.path.join(core.DATA, "VolumeTracings.csv"))
vt_g = {k: v for k, v in vt.groupby("FileName")}
val = fl[fl.Split == "VAL"].sample(frac=1.0, random_state=0)
seg = core.load_seg_model()
rows, t0 = [], time.time()
for _, r in val.iterrows():
    if len(rows) >= N_CAL:
        break
    g = vt_g.get(r.FileName + ".avi")
    if g is None or g.Frame.nunique() < 2:
        continue
    fr = core.read_video(os.path.join(core.VIDEOS, r.FileName + ".avi"))
    if fr is None:
        continue
    tm = {int(f): core.trace_mask(g[g.Frame == f][["X1", "Y1", "X2", "Y2"]].values) for f in g.Frame.unique()}
    fs = sorted(tm, key=lambda f: tm[f].sum())
    es_f, ed_f = fs[0], fs[-1]  # smaller traced area = ES, larger = ED
    m = core.segment(seg, fr[[ed_f, es_f]])
    row = dict(FileName=r.FileName, EDV_ref=r.EDV, ESV_ref=r.ESV, EF_ref=r.EF, ed_frame=ed_f, es_frame=es_f)
    for tag, mi, f in (("ED", m[0], ed_f), ("ES", m[1], es_f)):
        mi = core.largest_component(mi)
        a, L, vd, va = core.mask_geometry(mi)
        row.update({f"area_{tag}": a, f"len_{tag}": L, f"vdisk_{tag}": vd, f"val_{tag}": va, f"dice_{tag}": core.dice(mi, tm[f])})
    rows.append(row)
    if len(rows) % 50 == 0:
        print(len(rows), round(time.time() - t0), "s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "calibration_val.csv"), index=False)


def fit(vpx, vref):
    vpx, vref = np.asarray(vpx), np.asarray(vref)
    a0 = float(np.sum(vpx * vref) / np.sum(vpx ** 2))
    A = np.stack([vpx, np.ones_like(vpx)], 1)
    a1, b1 = np.linalg.lstsq(A, vref, rcond=None)[0]
    return dict(scale=a0, slope=float(a1), intercept=float(b1))


cal = {"n_videos": len(df), "split": "VAL", "fixed_fov_scale": (16.0 / 112) ** 3}
for name, key in (("disks", "vdisk"), ("arealength", "val")):
    vpx = np.concatenate([df[f"{key}_ED"], df[f"{key}_ES"]])
    vref = np.concatenate([df.EDV_ref, df.ESV_ref])
    cal[name] = fit(vpx, vref)
    cal[name]["implied_pixel_cm"] = float(cal[name]["scale"] ** (1 / 3))
cal["val_dice_ED_mean"] = float(df.dice_ED.mean())
cal["val_dice_ES_mean"] = float(df.dice_ES.mean())
json.dump(cal, open(os.path.join(OUT, "calibration.json"), "w"), indent=2)
print(json.dumps(cal, indent=2))

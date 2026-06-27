#!/usr/bin/env python3
"""
Scan the full CASH SAC archive once and write a per-hour metrics CSV.

One row per hourly SAC file. Metrics are designed for an ambient-noise /
data-quality study (lockdown signal, diurnal/seasonal cycles, completeness),
not for picking individual earthquakes.

Robust noise level = median of per-minute RMS within the hour, which rejects
brief transients (quakes, spikes) so the value reflects the background level.

Output: analysis/cash_hourly_metrics.csv
"""
import os, struct, sys, time
import numpy as np
from scipy.signal import butter, sosfilt, detrend

from config import DATA_ROOT as ROOT   # CASH SAC archive (see config.py)
OUT  = os.path.join(os.path.dirname(__file__), "cash_hourly_metrics.csv")

# ---- SAC reader (big-endian) -------------------------------------------------
def read_sac(path):
    d = open(path, "rb").read()
    if len(d) < 632:
        return None, None
    delta = struct.unpack(">f", d[0:4])[0]          # word 0
    npts  = struct.unpack(">i", d[316:320])[0]       # word 79
    if npts <= 0 or 632 + npts*4 > len(d):
        return None, None
    x = np.frombuffer(d[632:632+npts*4], dtype=">f4").astype("f8")
    return x, delta

# ---- filter cache (sample rate is ~constant, build once) ---------------------
_filters = {}
def get_sos(fs, lo, hi):
    key = (round(fs, 3), lo, hi)
    if key not in _filters:
        ny = fs / 2.0
        hi = min(hi, ny * 0.95)
        _filters[key] = butter(4, [lo/ny, hi/ny], btype="band", output="sos")
    return _filters[key]

def band_rms(x, fs, lo, hi):
    try:
        sos = get_sos(fs, lo, hi)
        y = sosfilt(sos, x)
        return float(np.sqrt(np.mean(y*y)))
    except Exception:
        return np.nan

# ---- per-file metrics --------------------------------------------------------
def metrics(x, fs):
    x = detrend(x, type="linear")
    full_rms = float(np.sqrt(np.mean(x*x)))
    # robust hourly noise: median of per-minute RMS
    win = max(1, int(round(60*fs)))
    n = (len(x)//win)*win
    if n >= win:
        w = x[:n].reshape(-1, win)
        minute_rms = np.sqrt(np.mean(w*w, axis=1))
        robust = float(np.median(minute_rms))
    else:
        robust = full_rms
    max_abs = float(np.max(np.abs(x)))
    # crude clipping flag: many samples pinned at the extreme value
    raw_max = np.max(np.abs(x + 0))  # x already detrended; use distribution tail
    p9999 = np.percentile(np.abs(x), 99.99)
    clip_frac = float(np.mean(np.abs(x) >= 0.999*max_abs)) if max_abs > 0 else 0.0
    dead = robust < 1e-3
    low_rms  = band_rms(x, fs, 0.2, 1.0)   # long-period / microseism leakage
    cult_rms = band_rms(x, fs, 2.0, 8.0)   # cultural / anthropogenic band
    return dict(full_rms=full_rms, robust_rms=robust, low_rms=low_rms,
                cult_rms=cult_rms, max_abs=max_abs, clip_frac=clip_frac,
                dead=int(dead))

# ---- walk the archive --------------------------------------------------------
def main():
    rows = []
    t0 = time.time()
    nfiles = 0
    years = sorted(d for d in os.listdir(ROOT) if d.isdigit())
    for y in years:
        ypath = os.path.join(ROOT, y)
        for m in sorted((d for d in os.listdir(ypath) if d.isdigit()), key=int):
            mpath = os.path.join(ypath, m)
            for dd in sorted((d for d in os.listdir(mpath) if d.isdigit()), key=int):
                dpath = os.path.join(mpath, dd)
                for h in range(24):
                    f = os.path.join(dpath, f"{h}.sac")
                    if not os.path.exists(f):
                        continue
                    x, delta = read_sac(f)
                    if x is None or delta is None or delta <= 0:
                        rows.append((int(y),int(m),int(dd),h,0,np.nan,*([np.nan]*5),0,0))
                        continue
                    fs = 1.0/delta
                    mm = metrics(x, fs)
                    rows.append((int(y),int(m),int(dd),h,len(x),round(fs,4),
                                 mm["full_rms"],mm["robust_rms"],mm["low_rms"],
                                 mm["cult_rms"],mm["max_abs"],mm["clip_frac"],
                                 mm["dead"]))
                    nfiles += 1
                    if nfiles % 2000 == 0:
                        dt = time.time()-t0
                        print(f"  {nfiles} files  {dt:5.0f}s  "
                              f"({nfiles/dt:.0f}/s)  at {y}-{m}-{dd}", flush=True)
    # write CSV
    import csv
    hdr = ["year","month","day","hour","npts","fs","full_rms","robust_rms",
           "low_rms","cult_rms","max_abs","clip_frac","dead"]
    with open(OUT, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(hdr); w.writerows(rows)
    print(f"DONE  {nfiles} files, {len(rows)} rows in {time.time()-t0:.0f}s -> {OUT}",
          flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STA/LTA earthquake detection over the CASH archive.

Two-stage for speed:
  1. Use cash_hourly_metrics.csv to flag CANDIDATE hours that contain a transient
     (full_rms >> robust background, or a large peak amplitude). Quiet hours are
     skipped without re-reading the file.
  2. Re-read only candidate hours, run recursive STA/LTA on a 1.5-8 Hz bandpass,
     extract precise trigger onset(s), peak amplitude, SNR and duration.

Output: analysis/detections.csv  (one row per trigger)
"""
import os, csv, struct, datetime as dt
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

HERE = os.path.dirname(__file__)
from config import DATA_ROOT as ROOT   # CASH SAC archive (see config.py)
CSV  = os.path.join(HERE, "cash_hourly_metrics.csv")
OUT  = os.path.join(HERE, "detections.csv")

# candidate-hour thresholds (tunable)
T_FULL_RATIO = 1.6     # full_rms / robust_rms
T_AMP_RATIO  = 12.0    # max_abs / robust_rms
# STA/LTA
F_LO, F_HI = 1.5, 8.0
STA_S, LTA_S = 1.0, 30.0
TRIG_ON, TRIG_OFF = 5.0, 1.5
MIN_SNR = 8.0          # keep triggers with peak/robust above this

def read_sac(path):
    d = open(path, "rb").read()
    if len(d) < 632: return None, None
    delta = struct.unpack(">f", d[0:4])[0]
    npts  = struct.unpack(">i", d[316:320])[0]
    if npts <= 0 or 632+npts*4 > len(d) or delta <= 0: return None, None
    return np.frombuffer(d[632:632+npts*4], dtype=">f4").astype("f8"), delta

def recursive_sta_lta(x, nsta, nlta):
    sq = x*x
    csta, clta = 1.0/nsta, 1.0/nlta
    sta = lfilter([csta], [1, -(1-csta)], sq)
    lta = lfilter([clta], [1, -(1-clta)], sq)
    lta = np.maximum(lta, 1e-12)
    r = sta/lta
    r[:int(nlta)] = 0.0     # suppress filter warm-up
    return r

def trigger_onset(ratio, on, off):
    trigs=[]; inq=False; start=0
    for i,v in enumerate(ratio):
        if not inq and v>=on:
            inq=True; start=i
        elif inq and v<off:
            trigs.append((start,i)); inq=False
    if inq: trigs.append((start,len(ratio)-1))
    return trigs

def load_candidates():
    cand=[]
    with open(CSV) as f:
        for d in csv.DictReader(f):
            try:
                npts=float(d["npts"]); rob=float(d["robust_rms"])
                full=float(d["full_rms"]); mx=float(d["max_abs"])
            except Exception:
                continue
            if not (npts>0 and rob>0): continue
            if full/rob>=T_FULL_RATIO or mx/rob>=T_AMP_RATIO:
                cand.append((int(d["year"]),int(d["month"]),int(d["day"]),
                             int(d["hour"]),rob))
    return cand

_sos=None
def get_sos(fs):
    global _sos
    if _sos is None:
        ny=fs/2.0
        _sos=butter(4,[F_LO/ny, min(F_HI,ny*0.95)/ny],btype="band",output="sos")
    return _sos

def main():
    cand=load_candidates()
    print(f"candidate hours: {len(cand)}")
    dets=[]
    for k,(y,m,day,h,rob) in enumerate(cand):
        f=os.path.join(ROOT,str(y),str(m),str(day),f"{h}.sac")
        x,delta=read_sac(f)
        if x is None: continue
        fs=1.0/delta
        x=x-np.mean(x)
        xf=sosfilt(get_sos(fs),x)
        ratio=recursive_sta_lta(xf,int(STA_S*fs),int(LTA_S*fs))
        hour_start=dt.datetime(y,m,day,h)
        for (a,b) in trigger_onset(ratio,TRIG_ON,TRIG_OFF):
            seg=x[a:b+1]
            if len(seg)==0: continue
            peak=float(np.max(np.abs(seg)))
            snr=peak/rob
            if snr<MIN_SNR: continue
            onset=hour_start+dt.timedelta(seconds=a/fs)
            dur=(b-a)/fs
            dets.append((onset.strftime("%Y-%m-%dT%H:%M:%S"),round(peak,1),
                         round(snr,1),round(dur,1),round(float(ratio[a:b+1].max()),1)))
        if (k+1)%500==0:
            print(f"  {k+1}/{len(cand)} candidate hours, {len(dets)} triggers")
    # collapse triggers that are very close (<90 s apart) into one event (keep max SNR)
    dets.sort()
    merged=[]
    for d in dets:
        t=dt.datetime.strptime(d[0],"%Y-%m-%dT%H:%M:%S")
        if merged:
            pt=dt.datetime.strptime(merged[-1][0],"%Y-%m-%dT%H:%M:%S")
            if (t-pt).total_seconds()<90:
                if d[2]>merged[-1][2]: merged[-1]=d   # keep stronger
                continue
        merged.append(list(d))
    with open(OUT,"w",newline="") as fo:
        w=csv.writer(fo)
        w.writerow(["time_utc","peak_amp","snr","duration_s","sta_lta_peak"])
        w.writerows(merged)
    print(f"DONE: {len(dets)} raw triggers -> {len(merged)} merged detections -> {OUT}")

if __name__=="__main__":
    main()

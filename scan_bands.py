#!/usr/bin/env python3
"""
Second pass: robust band-limited noise levels for the microseism study.

For each hourly SAC file, in each frequency band, compute the MEDIAN of
per-minute RMS values (so brief earthquakes / spikes don't inflate the level —
the broadband low_rms in the first pass was full-hour RMS and was earthquake
-contaminated).

Bands:
  sec   0.10-0.35 Hz  secondary ocean microseism (~3-10 s)
  short 0.35-1.0  Hz  short-period microseism / surf / wind
  cult  2.0-8.0   Hz  cultural / anthropogenic control

Output: analysis/cash_bands.csv
"""
import os, struct, csv, time
import numpy as np
from scipy.signal import butter, sosfilt, detrend

HERE=os.path.dirname(__file__)
from config import DATA_ROOT as ROOT   # CASH SAC archive (see config.py)
OUT=os.path.join(HERE,"cash_bands.csv")
BANDS=[("sec",0.10,0.35),("short",0.35,1.0),("cult",2.0,8.0)]

def read_sac(path):
    d=open(path,"rb").read()
    if len(d)<632: return None,None
    delta=struct.unpack(">f",d[0:4])[0]; npts=struct.unpack(">i",d[316:320])[0]
    if npts<=0 or 632+npts*4>len(d) or delta<=0: return None,None
    return np.frombuffer(d[632:632+npts*4],dtype=">f4").astype("f8"),delta

_sos={}
def sos_for(fs):
    key=round(fs,3)
    if key not in _sos:
        ny=fs/2.0
        out=[]
        for _,lo,hi in BANDS:
            loN=max(lo,0.001)/ny; hiN=min(hi,ny*0.95)/ny
            if loN>=hiN:               # band above Nyquist (anomalous fs) -> skip
                out.append(None)
            else:
                out.append(butter(4,[loN,hiN],btype="band",output="sos"))
        _sos[key]=out
    return _sos[key]

def robust_band(y,win):
    n=(len(y)//win)*win
    if n<win: return float(np.sqrt(np.mean(y*y)))
    w=y[:n].reshape(-1,win)
    return float(np.median(np.sqrt(np.mean(w*w,axis=1))))

def main():
    t0=time.time(); rows=[]; n=0
    years=sorted(d for d in os.listdir(ROOT) if d.isdigit())
    for y in years:
        yp=os.path.join(ROOT,y)
        for m in sorted((d for d in os.listdir(yp) if d.isdigit()),key=int):
            mp=os.path.join(yp,m)
            for dd in sorted((d for d in os.listdir(mp) if d.isdigit()),key=int):
                dp=os.path.join(mp,dd)
                for h in range(24):
                    f=os.path.join(dp,f"{h}.sac")
                    if not os.path.exists(f): continue
                    x,delta=read_sac(f)
                    if x is None: continue
                    fs=1.0/delta; win=max(1,int(round(60*fs)))
                    x=detrend(x,type="linear")
                    sl=sos_for(fs)
                    vals=[(robust_band(sosfilt(sl[i],x),win) if sl[i] is not None
                           else float("nan")) for i in range(len(BANDS))]
                    rows.append((int(y),int(m),int(dd),h,round(vals[0],4),
                                 round(vals[1],4),round(vals[2],4)))
                    n+=1
                    if n%4000==0:
                        dt=time.time()-t0
                        print(f"  {n} files {dt:.0f}s ({n/dt:.0f}/s) at {y}-{m}-{dd}",flush=True)
    with open(OUT,"w",newline="") as fo:
        w=csv.writer(fo); w.writerow(["year","month","day","hour","sec","short","cult"])
        w.writerows(rows)
    print(f"DONE {n} files in {time.time()-t0:.0f}s -> {OUT}",flush=True)

if __name__=="__main__":
    main()

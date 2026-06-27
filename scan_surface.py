#!/usr/bin/env python3
"""
Third scan pass: per-hour SURFACE-WAVE-band energy, for a teleseismic detector.

The body-wave STA/LTA detector (1.5-8 Hz) systematically misses great distant
earthquakes whose recorded energy is long-period surface waves. Here we measure
sustained low-frequency energy per hour so those events can be flagged.

Band 0.04-0.10 Hz (10-25 s). For each hour we filter, take per-minute RMS, and
record several PERCENTILES: p50 ~ background, p80/p95 ~ sustained packet. A
surface-wave train fills ~15-30 min of the hour, so it lifts p80 well above p50;
brief cultural spikes (sub-minute) do not.

Output: analysis/cash_surface.csv
"""
import os, struct, csv, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

HERE=os.path.dirname(__file__)
from config import DATA_ROOT as ROOT   # CASH SAC archive (see config.py)
OUT=os.path.join(HERE,"cash_surface.csv")
LO,HI=0.04,0.10

def read_sac(path):
    d=open(path,"rb").read()
    if len(d)<632: return None,None
    delta=struct.unpack(">f",d[0:4])[0]; npts=struct.unpack(">i",d[316:320])[0]
    if npts<=0 or 632+npts*4>len(d) or delta<=0: return None,None
    return np.frombuffer(d[632:632+npts*4],dtype=">f4").astype("f8"),delta

_sos={}
def sos_for(fs):
    k=round(fs,3)
    if k not in _sos:
        ny=fs/2.0
        _sos[k]=butter(4,[LO/ny,HI/ny],btype="band",output="sos") if HI<ny*0.95 else None
    return _sos[k]

def main():
    t0=time.time(); rows=[]; n=0
    for y in sorted(d for d in os.listdir(ROOT) if d.isdigit()):
        yp=os.path.join(ROOT,y)
        for m in sorted((d for d in os.listdir(yp) if d.isdigit()),key=int):
            mp=os.path.join(yp,m)
            for dd in sorted((d for d in os.listdir(mp) if d.isdigit()),key=int):
                dp=os.path.join(mp,dd)
                for h in range(24):
                    f=os.path.join(dp,f"{h}.sac")
                    if not os.path.exists(f): continue
                    x,delta=read_sac(f)
                    if x is None or len(x)<2000: continue
                    fs=1.0/delta; sos=sos_for(fs)
                    if sos is None: continue
                    x=detrend(x,type="linear")
                    try:
                        yb=sosfiltfilt(sos,x)
                    except Exception:
                        continue
                    win=max(1,int(round(60*fs))); nn=(len(yb)//win)*win
                    if nn<win: continue
                    mr=np.sqrt(np.mean(yb[:nn].reshape(-1,win)**2,axis=1))
                    p50,p80,p95=np.percentile(mr,[50,80,95])
                    rows.append((int(y),int(m),int(dd),h,round(p50,4),round(p80,4),round(p95,4)))
                    n+=1
                    if n%4000==0:
                        print(f"  {n} files {time.time()-t0:.0f}s at {y}-{m}-{dd}",flush=True)
    with open(OUT,"w",newline="") as fo:
        w=csv.writer(fo); w.writerow(["year","month","day","hour","p50","p80","p95"]); w.writerows(rows)
    print(f"DONE {n} files in {time.time()-t0:.0f}s -> {OUT}",flush=True)

if __name__=="__main__":
    main()

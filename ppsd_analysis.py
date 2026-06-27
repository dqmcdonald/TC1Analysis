#!/usr/bin/env python3
"""
Probabilistic Power Spectral Density (McNamara-Buland) for station CASH, via obspy.

We have no instrument response, so we attach a SYNTHETIC 4.5 Hz geophone response
(2 zeros at origin, 2 poles; nominal sensitivity). This gives the correct spectral
SHAPE and temporal variation; the ABSOLUTE dB level is nominal/uncalibrated (the
true gain is unknown), chosen to place the curve in a plausible range vs the
Peterson NLNM/NHNM.

Subsamples the 9-year archive (6 hours/day every 7th day) for tractable runtime
while keeping good temporal coverage.

Outputs (analysis dir):
  fig_ppsd.png            probabilistic PSD histogram (the McNamara plot)
  fig_ppsd_temporal.png   PSD at 1 s and 5 s over 2016-2025 (seasonal + 2025 fault)
  cash_ppsd.npz           the PPSD object (reusable)
Run with the obspy venv.
"""
import os, math, datetime as dt, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import read
from obspy.signal import PPSD
import config

HERE=os.path.dirname(os.path.abspath(__file__))

def temporal_plot(ppsd, filename, periods=(1.0,5.0)):
    """PSD at given periods over time (obspy's plot_temporal renders empty here)."""
    pc=np.array(ppsd.period_bin_centers); vals=np.array(ppsd.psd_values)
    times=np.array([t.datetime for t in ppsd.times_processed])
    o=np.argsort(times); times=times[o]; vals=vals[o]
    def roll(y,k=41):
        out=np.full(len(y),np.nan)
        for i in range(len(y)):
            a,b=max(0,i-k//2),min(len(y),i+k//2+1); out[i]=np.median(y[a:b])
        return out
    fig,ax=plt.subplots(figsize=(13,5))
    for P,c in zip(periods,["#2c7fb8","#d62728"]):
        j=int(np.argmin(np.abs(pc-P))); y=vals[:,j]
        ax.plot(times,y,lw=0.3,alpha=0.3,color=c)
        ax.plot(times,roll(y),lw=1.8,color=c,label=f"{pc[j]:.1f} s ({1/pc[j]:.2f} Hz)")
    ax.axvspan(dt.datetime(2025,7,1),dt.datetime(2025,8,1),color="orange",alpha=0.35,label="mid-2025 fault")
    ax.set_ylabel("PSD [dB] (nominal abs. level)"); ax.set_xlabel("year")
    ax.set_title("CASH noise PSD over time (subsampled; 41-pt rolling median)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3); ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout(); fig.savefig(filename,dpi=120); plt.close(fig)

# synthetic 4.5 Hz geophone (velocity) response
f0,h=4.5,0.7; w0=2*math.pi*f0
PAZ={"gain":1.0,
     "poles":[complex(-h*w0, w0*math.sqrt(1-h*h)),
              complex(-h*w0,-w0*math.sqrt(1-h*h))],
     "zeros":[0j,0j],
     "sensitivity":1e10}          # nominal; sets only the absolute dB offset

def main():
    start=dt.date(2016,12,3); end=dt.date(2025,7,31)
    hours=(0,4,8,12,16,20); step=7
    ppsd=None; n=0; d=start
    warnings.simplefilter("ignore")
    while d<=end:
        for hh in hours:
            p=os.path.join(config.DATA_ROOT,f"{d.year}/{d.month}/{d.day}/{hh}.sac")
            if not os.path.exists(p): continue
            try:
                tr=read(p)[0]
            except Exception:
                continue
            # SAC files carry an inconsistent/garbage location code per file;
            # force one channel id so PPSD accepts every hour.
            tr.stats.network="S"; tr.stats.station="CASH"
            tr.stats.location=""; tr.stats.channel="BHZ"
            tr.stats.delta=0.0533     # per-file delta varies slightly; force a uniform,
                                      # slightly-coarse delta so PPSD accepts every hour
                                      # and the 3600 s window always fits
            if ppsd is None:
                ppsd=PPSD(tr.stats, metadata=PAZ)
                print("init PPSD:",tr.id,"fs",round(tr.stats.sampling_rate,2))
            if ppsd.add(tr): n+=1
        if n and n%600<len(hours): print(f"  {d}  {n} segments",flush=True)
        d+=dt.timedelta(days=step)
    print(f"total PSD segments: {len(ppsd.times_processed)}")

    ppsd.save_npz(os.path.join(HERE,"cash_ppsd.npz"))
    ppsd.plot(filename=os.path.join(HERE,"fig_ppsd.png"),
              show_coverage=True, show_noise_models=True, period_lim=(0.2,20))
    temporal_plot(ppsd, os.path.join(HERE,"fig_ppsd_temporal.png"))
    print("-> fig_ppsd.png, fig_ppsd_temporal.png, cash_ppsd.npz")

if __name__=="__main__":
    main()

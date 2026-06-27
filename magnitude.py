#!/usr/bin/env python3
"""
Estimate earthquake magnitude from the CASH trace, vs the catalogue.

Two station magnitude scales, fitted to the catalogue magnitudes:
  ML-style (amplitude):  M = c0 + c1*log10(A) + c2*log10(dist) + c3*dist
  Md (coda duration):    M = d0 + d1*log10(dur) + d2*log10(dist)
A is the peak envelope amplitude in COUNTS (no instrument response, so the fitted
constant absorbs the gain); dur is the coda duration; dist is catalogue distance.
A 70/30 train/test split gives an honest out-of-sample accuracy.

Restricted to regional NZ events (30-1500 km) so the catalogue magnitudes are a
consistent local scale (GeoNet ML/MLv), not mixed with teleseismic mww.

Outputs: magnitude_estimates.csv, fig_magnitude.png, console summary.
Run with the obspy/cartopy venv.
"""
import os, csv, datetime as dt
import numpy as np
from scipy.signal import detrend, hilbert
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import generate_report as G, phases as ph

HERE=os.path.dirname(os.path.abspath(__file__))
COMB=os.path.join(HERE,"combined_catalog.csv")
DMIN,DMAX=30.0,1500.0

def measure(o, cat_dist, depth):
    S=ph.tt_s(cat_dist,depth)
    t,y,fs=G.get_trace(o,90,S+250)
    if t is None or len(t)<400: return None
    yb=G.bp(detrend(y,type="linear"),fs,0.8,9.0)
    w=max(1,int(round(1.0*fs)))
    env=np.convolve(np.abs(hilbert(yb)),np.ones(w)/w,mode="same")
    pre=env[t<-10]; sig=env[t>=0]
    if len(pre)<10 or len(sig)<10: return None
    noise=float(np.median(pre))+1e-9
    Apk=float(sig.max()); snr=Apk/noise
    if snr<5: return None
    above=np.where((t>=0)&(env>3*noise))[0]
    if len(above)==0: return None
    t0=t[above[0]]
    below=np.where((t>t0)&(env<2*noise))[0]
    dur=max((t[below[0]] if len(below) else t[-1])-t0,1.0)
    return dict(A=Apk,snr=snr,dur=dur)

def fit_report(name,X,y,tr,te):
    c,*_=np.linalg.lstsq(X[tr],y[tr],rcond=None)
    pred=X@c; r=pred[te]-y[te]
    ss=1-np.sum(r**2)/np.sum((y[te]-y[te].mean())**2)
    print(f"{name}: test std {r.std():.2f} mag, within +/-0.5 {100*np.mean(np.abs(r)<=0.5):.0f}%, "
          f"within +/-0.3 {100*np.mean(np.abs(r)<=0.3):.0f}%, R2 {ss:.2f}")
    return c,pred

def main():
    rows=[r for r in csv.DictReader(open(COMB))
          if r["detected_any"]=="1" and r["method"] in ("body","body+surface")
          and DMIN<=float(r["dist_km"])<DMAX and r["mag"]]
    print(f"candidate events ({DMIN:.0f}-{DMAX:.0f} km): {len(rows)}")
    out=[]; n=0
    for r in rows:
        n+=1
        try: o=dt.datetime.strptime(r["time"],"%Y-%m-%dT%H:%M:%S")
        except Exception: continue
        cat=float(r["dist_km"]); dep=float(r["depth"] or 15); m=float(r["mag"])
        try: e=measure(o,cat,dep)
        except Exception: e=None
        if e:
            out.append(dict(time=r["time"],place=r["place"][:38],mag=m,dist_km=round(cat),
                A=round(e["A"],1),dur_s=round(e["dur"],1),snr=round(e["snr"],1)))
        if n%200==0: print(f"  {n}/{len(rows)} scanned, {len(out)} measured",flush=True)

    with open(os.path.join(HERE,"magnitude_estimates.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    mag=np.array([r["mag"] for r in out]); dist=np.array([r["dist_km"] for r in out],float)
    A=np.array([r["A"] for r in out]); dur=np.array([r["dur_s"] for r in out])
    logA=np.log10(A); logD=np.log10(dist); logdur=np.log10(dur)
    rng=np.random.default_rng(0); idx=rng.permutation(len(mag)); cut=int(0.7*len(mag))
    tr,te=idx[:cut],idx[cut:]
    print(f"\nmeasured {len(out)} events; train {len(tr)} / test {len(te)}")
    Xml=np.column_stack([np.ones_like(logA),logA,logD,dist])
    Xmd=np.column_stack([np.ones_like(logdur),logdur,logD])
    cml,pml=fit_report("ML (amplitude)",Xml,mag,tr,te)
    cmd,pmd=fit_report("Md (duration) ",Xmd,mag,tr,te)
    print("ML coefs [c0,c1*logA,c2*logD,c3*dist]:",np.round(cml,3))

    # ---- figure: estimated vs catalogue magnitude (test set), both scales ----
    fig,axs=plt.subplots(1,2,figsize=(13,5.8))
    for ax,pred,name,c in [(axs[0],pml,"ML (amplitude)","#1f4e79"),
                           (axs[1],pmd,"Md (coda duration)","#7b3294")]:
        sc=ax.scatter(mag[te],pred[te],s=16,c=dist[te],cmap="viridis",alpha=0.7,edgecolors="none")
        ax.plot([2.5,7],[2.5,7],"k--",lw=1,label="1:1")
        for d in (0.5,-0.5): ax.plot([2.5,7],[2.5+d,7+d],":",c="#aaa",lw=1)
        ax.set_xlim(2.7,6.8); ax.set_ylim(2.7,6.8); ax.set_aspect("equal")
        r=pred[te]-mag[te]
        ax.set_xlabel("catalogue magnitude"); ax.set_ylabel(f"{name} estimate")
        ax.set_title(f"{name}\ntest std {r.std():.2f} mag, {100*np.mean(np.abs(r)<=0.5):.0f}% within +/-0.5")
        ax.grid(alpha=0.3); ax.legend(loc="upper left")
        fig.colorbar(sc,ax=ax,label="distance (km)")
    fig.suptitle("CASH single-station magnitude estimation (out-of-sample test set)",fontsize=13,weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_magnitude.png"),dpi=120)
    print("\n-> magnitude_estimates.csv, fig_magnitude.png")

if __name__=="__main__":
    main()

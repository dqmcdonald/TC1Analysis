#!/usr/bin/env python3
"""
Single-station distance from the trace alone (S-P time) vs the catalogue.

For each body-detected event we auto-pick the P and S onsets (recursive STA/LTA),
measure S-P, and invert it to an epicentral distance through the TauP (iasp91)
S-P-vs-distance curve at a NOMINAL 15 km depth -- i.e. using only the recording,
no catalogue distance/depth. We then compare to the catalogue distance.

Outputs: sp_estimates.csv, fig_sp_distance.png, console summary.
Run with the obspy/cartopy venv.
"""
import os, csv, datetime as dt
import numpy as np
from scipy.signal import detrend
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import generate_report as G, phases as ph, detect as D

HERE=os.path.dirname(os.path.abspath(__file__))
COMB=os.path.join(HERE,"combined_catalog.csv")
DEPTH_NOMINAL=15.0                 # assumed depth (we don't know it from the trace)
DMIN,DMAX=50.0,3000.0             # km range where S-P picking is meaningful

# ---- TauP S-P(distance) curve, for inverting S-P -> distance --------------
_deg=np.linspace(0.3,90,400)
_sp=np.array([ph.tt_s(d*111.19,DEPTH_NOMINAL)-ph.tt_p(d*111.19,DEPTH_NOMINAL) for d in _deg])
_ok=np.isfinite(_sp); _deg,_sp=_deg[_ok],_sp[_ok]
def sp_to_km(sp):
    return float(np.interp(sp,_sp,_deg))*111.19    # monotonic; clamps at ends

# ---- pick P and S onsets from one trace ----------------------------------
def estimate(o, cat_dist, depth):
    P=ph.tt_p(cat_dist,depth); S=ph.tt_s(cat_dist,depth)   # only to size the read window
    t,y,fs=G.get_trace(o,60,S+250)
    if t is None or len(t)<400: return None
    band=(1.5,9.0) if cat_dist<300 else (0.8,8.0)
    yb=G.bp(detrend(y,type="linear"),fs,*band)
    ratio=D.recursive_sta_lta(yb,int(round(1.0*fs)),int(round(30*fs)))
    trigs=D.trigger_onset(ratio,5.0,1.5)
    if len(trigs)<2: return None
    tP=t[trigs[0][0]]; snrP=float(ratio[trigs[0][0]:trigs[0][1]+1].max())
    tS=snrS=None
    for (a,b) in trigs[1:]:
        if t[a]-tP>2.0:
            tS=t[a]; snrS=float(ratio[a:b+1].max()); break
    if tS is None: return None
    sp=tS-tP
    if not (2.0<=sp<=700.0): return None
    return dict(sp=sp,est_km=sp_to_km(sp),tP=tP,tS=tS,snrP=snrP,snrS=snrS)

def main():
    rows=[r for r in csv.DictReader(open(COMB))
          if r["detected_any"]=="1" and r["method"] in ("body","body+surface")
          and DMIN<=float(r["dist_km"])<DMAX]
    print(f"candidate body-detected events ({DMIN:.0f}-{DMAX:.0f} km): {len(rows)}")
    out=[]; n=0
    for r in rows:
        n+=1
        try: o=dt.datetime.strptime(r["time"],"%Y-%m-%dT%H:%M:%S")
        except Exception: continue
        cat=float(r["dist_km"]); dep=float(r["depth"] or 15)
        try: e=estimate(o,cat,dep)
        except Exception: e=None
        if e:
            out.append(dict(time=r["time"],place=r["place"],cat_km=round(cat),
                est_km=round(e["est_km"]),sp_s=round(e["sp"],1),
                snrP=round(e["snrP"],1),snrS=round(e["snrS"],1)))
        if n%200==0: print(f"  {n}/{len(rows)} scanned, {len(out)} estimable",flush=True)

    with open(os.path.join(HERE,"sp_estimates.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    cat=np.array([r["cat_km"] for r in out],float); est=np.array([r["est_km"] for r in out],float)
    ratio=est/cat; logres=np.log10(ratio)
    print(f"\nEstimable: {len(out)} of {len(rows)} candidates ({100*len(out)/len(rows):.0f}%)")
    print(f"median est/catalogue ratio: {np.median(ratio):.2f}")
    print(f"median |log10 ratio|: {np.median(np.abs(logres)):.2f}  "
          f"(factor {10**np.median(np.abs(logres)):.2f})")
    for tol in (0.25,0.5):
        frac=np.mean(np.abs(ratio-1)<=tol); print(f"within +/-{tol*100:.0f}%: {100*frac:.0f}%")
    print("by distance band (n / median ratio / within +/-50% / typical factor):")
    for lo,hi in [(50,300),(300,1000),(1000,3000)]:
        m=(cat>=lo)&(cat<hi)
        if m.sum():
            r=ratio[m]
            print(f"  {lo:4d}-{hi:<4d} km: {m.sum():4d}  {np.median(r):.2f}  "
                  f"{100*np.mean(np.abs(r-1)<=.5):.0f}%  x{10**np.median(np.abs(np.log10(r))):.2f}")

    # ---- figure: estimated vs catalogue + residuals ----
    fig,(ax,axr)=plt.subplots(1,2,figsize=(13,5.6))
    ax.scatter(cat,est,s=14,c="#1f4e79",alpha=0.5,edgecolors="none")
    lim=[40,6000]; ax.plot(lim,lim,"k--",lw=1,label="1:1")
    ax.plot(lim,[1.25*x for x in lim],":",c="#888",lw=1,label="+/-25%")
    ax.plot(lim,[0.75*x for x in lim],":",c="#888",lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("catalogue distance (km)"); ax.set_ylabel("S-P estimated distance (km)")
    ax.set_title(f"Trace-only S-P distance vs catalogue  (n={len(out)})"); ax.legend(); ax.grid(alpha=0.3,which="both")
    axr.hist(logres,bins=40,color="#2c7fb8")
    axr.axvline(0,c="k",lw=1); axr.axvline(np.median(logres),c="#d62728",lw=1.5,label="median")
    axr.set_xlabel("log10(estimated / catalogue)"); axr.set_ylabel("count")
    axr.set_title("Residuals"); axr.legend()
    fig.suptitle("CASH single-station S-P distance estimation",fontsize=13,weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_sp_distance.png"),dpi=120)
    print("\n-> sp_estimates.csv, fig_sp_distance.png")

if __name__=="__main__":
    main()

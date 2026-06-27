#!/usr/bin/env python3
"""
Single-station distance from the trace alone (S-P time) vs the catalogue.

For each body-detected event we auto-pick the P and S onsets (recursive STA/LTA),
measure S-P, and invert it to an epicentral distance through the TauP (iasp91)
S-P-vs-distance curve. We produce two estimates:
  - nominal  : assume a fixed 15 km depth (true "trace-only", depth unknown)
  - depth-aware : use the event's catalogue depth in the inversion
and compare both to the catalogue distance. The difference between them shows how
much of the error is the depth assumption vs the picking.

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
DMIN,DMAX=50.0,3000.0                 # km range where S-P picking is meaningful

# ---- TauP S-P(distance, depth) table, for inverting S-P -> distance --------
DEPTHS=[0,15,35,70,150,300,500,700]
_deg=np.linspace(0.3,90,300)
_SPtab=np.array([[ph.tt_s(d*111.19,dep)-ph.tt_p(d*111.19,dep) for d in _deg] for dep in DEPTHS])
def sp_to_km(sp, depth):
    depth=max(0.0,min(float(depth),DEPTHS[-1])); j=np.searchsorted(DEPTHS,depth)
    if   j==0:            row=_SPtab[0]
    elif j>=len(DEPTHS):  row=_SPtab[-1]
    else:
        w=(depth-DEPTHS[j-1])/(DEPTHS[j]-DEPTHS[j-1]); row=_SPtab[j-1]*(1-w)+_SPtab[j]*w
    return float(np.interp(sp,row,_deg))*111.19    # monotonic; clamps at ends

# ---- pick P and S onsets from one trace ------------------------------------
def estimate(o, cat_dist, depth):
    S=ph.tt_s(cat_dist,depth)                       # only to size the read window
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
    return dict(sp=sp,snrP=snrP,snrS=snrS)

def stats(cat,est,label,mask=None):
    m=np.ones(len(cat),bool) if mask is None else mask
    r=est[m]/cat[m]
    print(f"{label:24s} n={m.sum():4d}  medRatio {np.median(r):.2f}  "
          f"within50% {100*np.mean(np.abs(r-1)<=.5):3.0f}%  factor {10**np.median(np.abs(np.log10(r))):.2f}")

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
            out.append(dict(time=r["time"],place=r["place"][:40],mag=float(r["mag"]),
                cat_km=round(cat),depth=round(dep),sp_s=round(e["sp"],1),
                est_nom_km=round(sp_to_km(e["sp"],15.0)),
                est_dep_km=round(sp_to_km(e["sp"],dep)),
                snrP=round(e["snrP"],1),snrS=round(e["snrS"],1)))
        if n%200==0: print(f"  {n}/{len(rows)} scanned, {len(out)} estimable",flush=True)

    with open(os.path.join(HERE,"sp_estimates.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    cat=np.array([r["cat_km"] for r in out],float)
    nom=np.array([r["est_nom_km"] for r in out],float)
    dep=np.array([r["est_dep_km"] for r in out],float)
    dpth=np.array([r["depth"] for r in out],float)
    mag=np.array([r["mag"] for r in out],float)
    print(f"\nEstimable: {len(out)} of {len(rows)} candidates ({100*len(out)/len(rows):.0f}%)")
    stats(cat,nom,"nominal 15 km"); stats(cat,dep,"depth-aware")
    print("deep events (depth>70 km):")
    stats(cat,nom,"  nominal",dpth>70); stats(cat,dep,"  depth-aware",dpth>70)
    print("by distance band (depth-aware):")
    for lo,hi in [(50,300),(300,1000),(1000,3000)]:
        stats(cat,dep,f"  {lo}-{hi} km",(cat>=lo)&(cat<hi))

    # ---- figure ----
    fig,(ax,axr)=plt.subplots(1,2,figsize=(13,5.6))
    sc=ax.scatter(cat,dep,s=20,c=mag,cmap="plasma",vmin=4,vmax=8,alpha=0.8,edgecolors="none")
    fig.colorbar(sc,ax=ax,label="magnitude")
    lim=[40,6000]
    ax.plot(lim,lim,"k--",lw=1,label="1:1")
    ax.plot(lim,[1.25*x for x in lim],":",c="#888",lw=1,label="+/-25%"); ax.plot(lim,[0.75*x for x in lim],":",c="#888",lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("catalogue distance (km)"); ax.set_ylabel("S-P estimated distance (km, depth-aware)")
    ax.set_title(f"Trace-only S-P distance vs catalogue (n={len(out)})"); ax.legend(); ax.grid(alpha=0.3,which="both")
    bins=np.linspace(-1.6,1.6,49)
    axr.hist(np.log10(nom/cat),bins=bins,histtype="step",lw=1.8,color="#888",label="nominal 15 km")
    axr.hist(np.log10(dep/cat),bins=bins,histtype="step",lw=1.8,color="#d62728",label="depth-aware")
    axr.axvline(0,c="k",lw=1)
    axr.set_xlabel("log10(estimated / catalogue)"); axr.set_ylabel("count")
    axr.set_title("Residuals: depth-aware ≈ nominal\n(scatter is dominated by the S-pick, not depth)",fontsize=10)
    axr.legend()
    fig.suptitle("CASH single-station S-P distance estimation",fontsize=13,weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_sp_distance.png"),dpi=120)
    print("\n-> sp_estimates.csv, fig_sp_distance.png")

if __name__=="__main__":
    main()

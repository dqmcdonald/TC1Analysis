#!/usr/bin/env python3
"""
Combined CASH earthquake catalog: merge the body-wave and surface-wave detectors
into one table tagging how (if at all) CASH recorded each catalogued earthquake.

The two methods are complementary:
  - body-wave STA/LTA (detect.py / matched_events.csv): local & regional quakes
  - surface-wave detector (surface_detections.csv): great distant earthquakes

Inputs:  matched_events.csv (catalogue + body-wave flag), surface_detections.csv
Outputs: combined_catalog.csv      every catalogued event, tagged by method
         cash_detected_catalog.csv  just the events CASH recorded (the catalogue)
         fig_combined_detection.png magnitude vs distance coloured by method
"""
import os, csv, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__)
MATCH=os.path.join(HERE,"matched_events.csv")
SURF=os.path.join(HERE,"surface_detections.csv")

def load_surface_events():
    evs=[]
    for d in csv.DictReader(open(SURF)):
        try:
            s=dt.datetime.fromisoformat(d["start"]); e=dt.datetime.fromisoformat(d["end"])
            evs.append((s,e,float(d["score"])))
        except Exception: continue
    return evs

def regime(dist):
    return "local" if dist<150 else "regional" if dist<1500 else "teleseismic"

def main():
    surf=load_surface_events()
    rows=[]
    for d in csv.DictReader(open(MATCH)):
        try:
            o=dt.datetime.strptime(d["time"],"%Y-%m-%dT%H:%M:%S")
            m=float(d["mag"]); dist=float(d["dist_km"])
        except Exception: continue
        body=d["detected"]=="1"
        bsnr=d["det_snr"]; blag=d["lag_s"]
        # surface match: Rayleigh arrival window overlaps a flagged surface event
        a=o+dt.timedelta(seconds=dist/4.2); b=o+dt.timedelta(seconds=dist/2.3)
        sscore=None
        for (s,e,sc) in surf:
            if s<=b and e>=a:
                sscore=sc if sscore is None else max(sscore,sc);
        surface=sscore is not None
        method=("body+surface" if body and surface else "body" if body else
                "surface" if surface else "none")
        rows.append(dict(time=d["time"],mag=round(m,1),mtype=d["mtype"],
            dist_km=round(dist),depth=d["depth"],src=d["src"],place=d["place"],
            regime=regime(dist),body=int(body),body_snr=bsnr,body_lag=blag,
            surface=int(surface),surface_score=("" if sscore is None else round(sscore,1)),
            detected_any=int(body or surface),method=method))

    cols=["time","mag","mtype","dist_km","depth","src","place","regime",
          "body","body_snr","body_lag","surface","surface_score","detected_any","method"]
    with open(os.path.join(HERE,"combined_catalog.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=cols); w.writeheader(); w.writerows(rows)
    det=[r for r in rows if r["detected_any"]]
    with open(os.path.join(HERE,"cash_detected_catalog.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=cols); w.writeheader(); w.writerows(det)

    # ---- summary ----
    n=len(rows); nd=len(det)
    by=lambda k:sum(1 for r in rows if r["method"]==k)
    print(f"Catalogued events: {n}")
    print(f"Detected by CASH (any method): {nd} ({100*nd/n:.0f}%)")
    print(f"  body only     : {by('body')}")
    print(f"  surface only  : {by('surface')}")
    print(f"  body+surface  : {by('body+surface')}")
    print("\nComplementarity by regime (detected / catalogued):")
    for rg in ["local","regional","teleseismic"]:
        sel=[r for r in rows if r["regime"]==rg]
        b=sum(r["body"] for r in sel); s=sum(r["surface"] for r in sel)
        a=sum(r["detected_any"] for r in sel)
        print(f"  {rg:11s} n={len(sel):4d}  body {100*b/len(sel):3.0f}%  "
              f"surface {100*s/len(sel):3.0f}%  any {100*a/len(sel):3.0f}%")
    print("\nGreat teleseisms (>1500 km, M>=7.5) -- where surface waves dominate:")
    for r in sorted([r for r in rows if r["dist_km"]>1500 and r["mag"]>=7.5],
                    key=lambda r:-r["mag"])[:12]:
        print(f"  {r['time'][:10]}  M{r['mag']:.1f}  {r['dist_km']:5d} km  {r['method']:13s} {r['place'][:34]}")

    # ---- figure: method map ----
    style={"body+surface":("#7b3294","both methods",36),
           "body":("#2c7fb8","body-wave only",26),
           "surface":("#1a9850","surface-wave only",30),
           "none":("#cccccc","not detected",10)}
    fig,ax=plt.subplots(figsize=(11,6.5))
    for meth,(c,lbl,sz) in style.items():
        xs=[r["dist_km"] for r in rows if r["method"]==meth]
        ys=[r["mag"] for r in rows if r["method"]==meth]
        ax.scatter(xs,ys,s=sz,c=c,label=f"{lbl} ({len(xs)})",alpha=0.75,
                   edgecolors="none",zorder=3 if meth!="none" else 1)
    ax.axvline(1500,ls="--",c="#888",lw=1); ax.text(1600,3.1,"regional | teleseismic",color="#666",fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("epicentral distance from CASH (km)")
    ax.set_ylabel("magnitude"); ax.grid(alpha=0.3)
    ax.set_title("Combined CASH earthquake catalog — detection method by magnitude & distance")
    ax.legend(loc="lower right",fontsize=9,framealpha=0.95)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_combined_detection.png"),dpi=120); plt.close(fig)
    print("\n-> combined_catalog.csv, cash_detected_catalog.csv, fig_combined_detection.png")

if __name__=="__main__":
    main()

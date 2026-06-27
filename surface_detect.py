#!/usr/bin/env python3
"""
Surface-wave detector for great distant earthquakes the body-wave detector misses.

Input: cash_surface.csv (per-hour percentiles of 0.04-0.10 Hz per-minute RMS).
A teleseismic surface-wave train is a BRIEF (1-2 h) burst of sustained low-frequency
energy. We score each hour as

    score = p80 / baseline,   baseline = centered 7-day median of p50

so brief teleseisms spike (the 7-day baseline barely moves) while multi-day ocean
microseism storms do NOT (baseline rises with them). Hours above THRESH are flagged
and merged into events, then cross-matched to catalogued M>=7 quakes via their
Rayleigh-wave arrival window.

Outputs: analysis/surface_detections.csv, fig_surface_detector.png, console report.
"""
import os, csv, math, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(__file__)
SURF=os.path.join(HERE,"cash_surface.csv")
MATCH=os.path.join(HERE,"matched_events.csv")
THRESH=3.5            # score threshold
BASE_DAYS=2           # half-window (+-2 days) for the quiet-floor baseline
BASE_PCTL=20          # baseline = this percentile of p50 (the local quiet floor,
                      # not the median -- median is biased high during storm weeks)

def load_surface():
    rows=[]
    for d in csv.DictReader(open(SURF)):
        try:
            t=dt.datetime(int(d["year"]),int(d["month"]),int(d["day"]),int(d["hour"]))
            rows.append((t,float(d["p50"]),float(d["p80"])))
        except Exception: continue
    rows.sort()
    return rows

def rolling_baseline(t,p50,days,pctl=50):
    half=days*24*3600           # half window in seconds (+-days)
    ts=np.array([x.timestamp() for x in t]); base=np.empty(len(p50))
    lo=0; hi=0; n=len(p50)
    for i in range(n):
        while ts[i]-ts[lo]>half: lo+=1
        if hi<i: hi=i
        while hi+1<n and ts[hi+1]-ts[i]<=half: hi+=1
        base[i]=np.percentile(p50[lo:hi+1],pctl)
    return base

def main():
    rows=load_surface()
    t=[r[0] for r in rows]; p50=np.array([r[1] for r in rows]); p80=np.array([r[2] for r in rows])
    base=rolling_baseline(t,p50,BASE_DAYS,BASE_PCTL)
    score=p80/np.maximum(base,1e-6)

    flag=score>=THRESH
    # merge consecutive flagged hours (allow 1 h gap) into events; keep peak hour
    events=[]
    i=0; n=len(t)
    while i<n:
        if not flag[i]: i+=1; continue
        j=i
        while j+1<n and (flag[j+1] or (j+2<n and flag[j+2])) and (t[j+1]-t[i]).total_seconds()<6*3600:
            j+=1
        seg=range(i,j+1); k=max(seg,key=lambda x:score[x])
        events.append((t[i],t[j]+dt.timedelta(hours=1),t[k],float(score[k])))
        i=j+1
    print(f"hours scanned {n}, flagged {int(flag.sum())}, merged surface events {len(events)}")

    # catalogue: M>=7 teleseisms from matched_events.csv
    cat=[]
    for d in csv.DictReader(open(MATCH)):
        try:
            m=float(d["mag"]); dist=float(d["dist_km"])
            o=dt.datetime.strptime(d["time"],"%Y-%m-%dT%H:%M:%S")
        except Exception: continue
        if m>=7.0 and dist>1500:
            cat.append(dict(o=o,m=m,dist=dist,place=d["place"],
                            body=(d["detected"]=="1")))
    # match each catalogue event to a surface event via Rayleigh arrival window
    def surf_window(o,dist):
        return o+dt.timedelta(seconds=dist/4.2), o+dt.timedelta(seconds=dist/2.3)
    nsurf=0; newly=0; lines=[]
    used=[False]*len(events)
    for c in cat:
        a,b=surf_window(c["o"],c["dist"])
        hit=None
        for ei,(s,e,pkt,sc) in enumerate(events):
            if s<=b and e>=a:           # overlap
                hit=(ei,sc); break
        c["surf"]=hit is not None; c["sc"]=hit[1] if hit else None
        if hit: nsurf+=1; used[hit[0]]=True
        if hit and not c["body"]: newly+=1
    tot=len(cat); body=sum(c["body"] for c in cat)
    print(f"\nCatalogued M>=7 teleseisms (>1500 km): {tot}")
    print(f"  caught by BODY-wave detector:    {body}  ({100*body/tot:.0f}%)")
    print(f"  caught by SURFACE-wave detector: {nsurf}  ({100*nsurf/tot:.0f}%)")
    print(f"  NEWLY caught (surface, missed by body): {newly}")
    # combined
    comb=sum(c["body"] or c["surf"] for c in cat)
    print(f"  combined body OR surface:        {comb}  ({100*comb/tot:.0f}%)")
    print("  detection rate by magnitude (body / surface / combined):")
    for lo,hi in [(7.0,7.5),(7.5,8.0),(8.0,9.9)]:
        sel=[c for c in cat if lo<=c["m"]<hi]
        if not sel: continue
        b=sum(c["body"] for c in sel); s=sum(c["surf"] for c in sel)
        cb=sum(c["body"] or c["surf"] for c in sel)
        print(f"    M{lo}-{hi}: n={len(sel):3d}  {100*b/len(sel):3.0f}% / "
              f"{100*s/len(sel):3.0f}% / {100*cb/len(sel):3.0f}%")
    # false alarms: flagged surface events not matched to any M>=7 teleseism
    fa=sum(1 for u in used if not u)
    print(f"  surface events NOT matching M>=7 (storms/regional/noise): {fa} of {len(events)}")

    # Myanmar check
    my=[c for c in cat if "Mandalay" in c["place"] or "Myanmar" in c["place"]]
    for c in my:
        print(f"\nMyanmar {c['o']:%Y-%m-%d} M{c['m']}: body={c['body']} surface={c['surf']}"
              + (f" (score {c['sc']:.1f})" if c['surf'] else ""))

    # newly-caught list
    print("\nGreat quakes newly caught by the surface detector (body missed):")
    for c in sorted([c for c in cat if c["surf"] and not c["body"]],key=lambda c:-c["m"])[:15]:
        print(f"  {c['o']:%Y-%m-%d} M{c['m']:.1f}  {c['dist']:.0f} km  score {c['sc']:.1f}  {c['place'][:42]}")

    with open(os.path.join(HERE,"surface_detections.csv"),"w",newline="") as fo:
        w=csv.writer(fo); w.writerow(["start","end","peak","score"])
        for s,e,pk,sc in events: w.writerow([s,e,pk,round(sc,2)])

    # figure: (a) detection rate by magnitude, (b) score vs magnitude
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5))
    bins=[(7.0,7.5),(7.5,8.0),(8.0,9.9)]; labels=["M7.0-7.5","M7.5-8.0","M8.0+"]
    bodyr=[];surfr=[];combr=[]
    for lo,hi in bins:
        sel=[c for c in cat if lo<=c["m"]<hi]; nseln=max(len(sel),1)
        bodyr.append(100*sum(c["body"] for c in sel)/nseln)
        surfr.append(100*sum(c["surf"] for c in sel)/nseln)
        combr.append(100*sum(c["body"] or c["surf"] for c in sel)/nseln)
    x=np.arange(3); w=0.26
    axL.bar(x-w,bodyr,w,label="body-wave detector",color="#bbbbbb")
    axL.bar(x,surfr,w,label="surface-wave detector",color="#1a9850")
    axL.bar(x+w,combr,w,label="combined",color="#1f4e79")
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel("% of catalogued teleseisms detected"); axL.set_ylim(0,100)
    axL.set_title("Detection rate by magnitude (>1500 km)"); axL.legend(fontsize=9); axL.grid(alpha=0.3,axis="y")
    for xi,(lo,hi) in zip(x,bins):
        axL.text(xi,2,f"n={len([c for c in cat if lo<=c['m']<hi])}",ha="center",fontsize=8,color="#444")

    M=np.array([c["m"] for c in cat]); SC=np.array([min(c["sc"] or 0,40) for c in cat])
    det=np.array([c["surf"] for c in cat])
    axR.scatter(M[~det],[THRESH*0.6]*np.sum(~det),s=25,c="#cccccc",label="missed",zorder=2)
    axR.scatter(M[det],SC[det],s=30,c="#1a9850",label="caught",zorder=3)
    axR.axhline(THRESH,color="#d62728",ls="--",lw=1,label=f"threshold {THRESH}")
    my=[c for c in cat if "Mandalay" in c["place"]]
    if my: axR.annotate("Myanmar M7.7",(my[0]["m"],min(my[0]["sc"],40)),
                        textcoords="offset points",xytext=(6,6),fontsize=8,color="#063")
    axR.set_xlabel("magnitude"); axR.set_ylabel("surface-wave score (capped 40)")
    axR.set_title("Detector response vs magnitude"); axR.legend(fontsize=9); axR.grid(alpha=0.3)
    fig.suptitle("CASH surface-wave detector catches great distant earthquakes the body-wave detector misses",
                 fontsize=12,weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_surface_detector.png"),dpi=120); plt.close(fig)
    print("\n-> surface_detections.csv, fig_surface_detector.png")

if __name__=="__main__":
    main()

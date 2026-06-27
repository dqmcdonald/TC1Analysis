#!/usr/bin/env python3
"""
Cross-match CASH STA/LTA detections against earthquake catalogs.

Fetches:
  GeoNet (regional NZ)  M>=3.0 within 8 deg of CASH   [FDSN text]
  USGS  (global)        M>=6.0 worldwide               [FDSN csv]
for 2016-12 .. 2025-07, chunked by year, cached in analysis/catalogs/.

For every catalog event it predicts the wave-arrival window at CASH from the
great-circle distance and checks whether a detection falls in that window.

Outputs:
  analysis/matched_events.csv   one row per catalog event (detected or not)
  analysis/fig_detection_capability.png   magnitude vs distance, detected/not
  analysis/catalog_summary.txt
"""
import os, csv, io, math, datetime as dt, urllib.request, urllib.parse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phases as ph   # TauP-based tt_p / tt_s (obspy iasp91, depth-corrected)

HERE = os.path.dirname(__file__)
CATDIR = os.path.join(HERE, "catalogs"); os.makedirs(CATDIR, exist_ok=True)
DET = os.path.join(HERE, "detections.csv")

CASH_LAT, CASH_LON = -43.567, 172.622      # Cashmere, Christchurch NZ
YEARS = list(range(2016, 2026))

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    p1,p2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# iasp91 first-arriving P travel time (surface source) vs epicentral degrees.
# Used so teleseismic match windows track the real P arrival (a 4.5 Hz geophone
# records short-period body waves, not long-period surface waves).
_pscache={}
def _PS(dist_km, depth_km):
    key=(round(dist_km/27.8), round(depth_km/10.0))   # ~0.25 deg, 10 km bins
    if key not in _pscache:
        _pscache[key]=(ph.tt_p(dist_km, depth_km), ph.tt_s(dist_km, depth_km))
    return _pscache[key]

def arrival_window(dist_km, depth_km, origin_ts):
    """(t0,t1): body-wave match window from TauP P (and S/Lg for regional)."""
    P,S=_PS(dist_km, depth_km)
    if dist_km < 1500:                       # regional: P .. S/Lg + coda
        return origin_ts+P-30, origin_ts+S+150
    return origin_ts+P-45, origin_ts+P+360   # teleseismic: body-wave coda around P

def fetch(url, cache):
    path=os.path.join(CATDIR,cache)
    if os.path.exists(path) and os.path.getsize(path)>0:
        return open(path,encoding="utf-8",errors="replace").read()
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"cash-analysis/1.0"})
        txt=urllib.request.urlopen(req,timeout=60).read().decode("utf-8","replace")
    except Exception as e:
        txt=""
        print(f"  fetch failed {cache}: {e}")
    open(path,"w").write(txt)
    return txt

def get_geonet():
    evs=[]
    for y in YEARS:
        a=f"{y}-01-01T00:00:00"; b=f"{y+1}-01-01T00:00:00"
        q=urllib.parse.urlencode(dict(starttime=a,endtime=b,minmagnitude=3.0,
            latitude=CASH_LAT,longitude=CASH_LON,maxradius=8.0,format="text"))
        txt=fetch("https://service.geonet.org.nz/fdsnws/event/1/query?"+q,
                  f"geonet_{y}.txt")
        for line in txt.splitlines():
            if not line or line.startswith("#"): continue
            p=line.split("|")
            if len(p)<13: continue
            try:
                t=dt.datetime.strptime(p[1][:19],"%Y-%m-%dT%H:%M:%S")
                evs.append(dict(time=t,lat=float(p[2]),lon=float(p[3]),
                    depth=float(p[4] or 0),mag=float(p[10]),mtype=p[9],
                    place=p[12],src="geonet"))
            except Exception: continue
    return evs

def get_usgs():
    evs=[]
    for y in YEARS:
        a=f"{y}-01-01"; b=f"{y+1}-01-01"
        q=urllib.parse.urlencode(dict(format="csv",starttime=a,endtime=b,
            minmagnitude=6.0))
        txt=fetch("https://earthquake.usgs.gov/fdsnws/event/1/query?"+q,
                  f"usgs_{y}.csv")
        if not txt.strip(): continue
        for r in csv.DictReader(io.StringIO(txt)):
            try:
                t=dt.datetime.strptime(r["time"][:19],"%Y-%m-%dT%H:%M:%S")
                evs.append(dict(time=t,lat=float(r["latitude"]),
                    lon=float(r["longitude"]),depth=float(r["depth"] or 0),
                    mag=float(r["mag"]),mtype=r["magType"],place=r["place"],
                    src="usgs"))
            except Exception: continue
    return evs

def load_detections():
    dets=[]
    if not os.path.exists(DET): return dets
    for r in csv.DictReader(open(DET)):
        try:
            t=dt.datetime.strptime(r["time_utc"],"%Y-%m-%dT%H:%M:%S")
            dets.append((t,float(r["snr"]),float(r["peak_amp"])))
        except Exception: continue
    dets.sort()
    return dets

def main():
    span0=dt.datetime(2016,12,3); span1=dt.datetime(2025,8,1)
    evs=[e for e in (get_geonet()+get_usgs()) if span0<=e["time"]<=span1]
    # distance + dedup (same quake in both catalogs: <90s & <120km -> keep geonet)
    for e in evs: e["dist"]=haversine_km(CASH_LAT,CASH_LON,e["lat"],e["lon"])
    evs.sort(key=lambda e:(e["time"], -e["mag"]))
    dedup=[]
    for e in evs:
        dup=False
        for f in dedup[-30:]:
            if abs((e["time"]-f["time"]).total_seconds())<90 and \
               haversine_km(e["lat"],e["lon"],f["lat"],f["lon"])<120:
                dup=True; break
        if not dup: dedup.append(e)
    evs=dedup
    dets=load_detections()
    dts=np.array([d[0].timestamp() for d in dets]) if dets else np.array([])

    # false-alarm floor: random detection rate * window length
    span_s=(span1-span0).total_seconds()
    det_rate=len(dets)/span_s if dets else 0.0   # detections per second

    rows=[]; ndet=0
    for e in evs:
        o=e["time"].timestamp(); dist=e["dist"]
        t0,t1=arrival_window(dist, e["depth"], o)
        p_false=1.0-math.exp(-det_rate*(t1-t0))   # chance of a chance match
        det=False; dsnr=np.nan; lag=np.nan
        if dts.size:
            m=(dts>=t0)&(dts<=t1)
            if m.any():
                idx=np.where(m)[0]
                best=idx[np.argmax([dets[i][1] for i in idx])]
                det=True; dsnr=dets[best][1]; lag=dets[best][0].timestamp()-o
                ndet+=1
        rows.append(dict(time=e["time"].strftime("%Y-%m-%dT%H:%M:%S"),
            mag=round(e["mag"],1),mtype=e["mtype"],dist_km=round(dist,0),
            depth=round(e["depth"],0),src=e["src"],place=e["place"][:48],
            detected=int(det),det_snr=("" if np.isnan(dsnr) else round(dsnr,1)),
            lag_s=("" if np.isnan(lag) else round(lag,0)),
            p_false=round(p_false,3)))

    with open(os.path.join(HERE,"matched_events.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)

    # ---- figure 1: magnitude vs distance, detected/not ----
    D=np.array([r["dist_km"] for r in rows]); M=np.array([r["mag"] for r in rows])
    det=np.array([r["detected"] for r in rows],bool)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.scatter(D[~det],M[~det],s=10,c="#cccccc",label="not detected",alpha=0.5)
    ax.scatter(D[det],M[det],s=22,c="#d62728",label="matched a CASH trigger",alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("epicentral distance from CASH (km)")
    ax.set_ylabel("magnitude"); ax.set_title("CASH earthquake detection capability (catalog cross-match)")
    ax.axvline(1450,ls="--",c="#888",lw=1); ax.text(1500,3.1,"regional | teleseismic",color="#666",fontsize=8)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_detection_capability.png"),dpi=120)
    plt.close(fig)

    # ---- figure 2: chance-corrected detection probability vs magnitude ----
    Pf=np.array([r["p_false"] for r in rows])
    reg=D<1450
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,mask,title in [(axs[0],reg,"Regional (<1450 km)"),
                          (axs[1],~reg,"Teleseismic (>1450 km)")]:
        edges=np.arange(3.0,9.01,0.5); cx=[]; obs=[]; cha=[]; nn=[]
        for i in range(len(edges)-1):
            b=mask&(M>=edges[i])&(M<edges[i+1])
            if b.sum()==0: continue
            cx.append((edges[i]+edges[i+1])/2)
            obs.append(det[b].mean()); cha.append(Pf[b].mean()); nn.append(b.sum())
        cx=np.array(cx);obs=np.array(obs);cha=np.array(cha)
        ax.bar(cx,obs,width=0.4,color="#d62728",alpha=0.85,label="observed match rate")
        ax.plot(cx,cha,"k--o",ms=4,label="chance (false-alarm) floor")
        corr=np.clip((obs-cha)/np.clip(1-cha,1e-6,None),0,1)
        ax.plot(cx,corr,"-s",color="#1f4e79",ms=4,label="chance-corrected P(detect)")
        for x,y,n in zip(cx,obs,nn): ax.text(x,y+0.02,str(n),ha="center",fontsize=7,color="#444")
        ax.set_title(title); ax.set_xlabel("magnitude"); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)
    axs[0].set_ylabel("P(detected)"); axs[0].legend(fontsize=8)
    fig.suptitle("CASH detection probability vs magnitude (n = catalogued events per bin)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_detection_vs_mag.png"),dpi=120)
    plt.close(fig)

    # summary
    exp_false=sum(r["p_false"] for r in rows)        # expected chance matches
    L=[f"Detections fed in: {len(dets)} ({len(dets)/3162:.1f}/day) -> false-match floor",
       f"Catalog events in window: {len(rows)} (after dedup)"]
    L.append(f"Detected by CASH: {ndet} ({100*ndet/len(rows):.1f}%)")
    L.append(f"Expected chance matches (false-alarm floor): {exp_false:.0f}"
             f"  => ~{max(0,ndet-exp_false):.0f} likely real")
    L.append("Detection rate by distance (detected / catalogued):")
    for lo,hi in [(0,100),(100,300),(300,1000),(1000,1e9)]:
        sel=[r for r in rows if lo<=r["dist_km"]<hi]
        if sel:
            dd=sum(r["detected"] for r in sel)
            fa=sum(r["p_false"] for r in sel)
            L.append(f"  {lo}-{hi if hi<1e9 else 'inf'} km: {dd}/{len(sel)} "
                     f"({100*dd/len(sel):.0f}%)  [~{fa:.0f} expected by chance]")
    # detection capability: smallest reliably-detected mag per distance band
    L.append("Detection threshold (smallest mag detected per band):")
    for lo,hi in [(0,150),(150,400),(400,1000)]:
        sel=[r for r in rows if lo<=r["dist_km"]<hi and r["detected"]]
        if sel:
            L.append(f"  {lo}-{hi} km: M{min(r['mag'] for r in sel)} "
                     f"(of {sum(1 for r in rows if lo<=r['dist_km']<hi)} catalogued)")
    big=sorted([r for r in rows if r["detected"]],key=lambda r:-r["mag"])[:10]
    L.append("Largest detected events:")
    for r in big:
        L.append(f"  {r['time']}  M{r['mag']} {r['mtype']}  {r['dist_km']:.0f} km  "
                 f"snr={r['det_snr']} lag={r['lag_s']}s  {r['place']}")
    txt="\n".join(L)
    open(os.path.join(HERE,"catalog_summary.txt"),"w").write(txt+"\n")
    print(txt)
    print("\n-> matched_events.csv, fig_detection_capability.png")

if __name__=="__main__":
    main()

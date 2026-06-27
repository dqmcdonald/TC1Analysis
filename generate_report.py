#!/usr/bin/env python3
"""
Build a custom PDF report of the most significant earthquakes detected at CASH.

Pulls the actual recorded waveforms from the SAC archive for a curated set of
featured events, plus summary tables, into analysis/CASH_earthquake_report.pdf.
Uses matplotlib's PdfPages (no external PDF/seismology libs needed).
"""
import os, csv, struct, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt, detrend

HERE=os.path.dirname(__file__)
from config import DATA_ROOT as ROOT   # CASH SAC archive (see config.py)
MATCH=os.path.join(HERE,"matched_events.csv")
OUT=os.path.join(HERE,"CASH_earthquake_report.pdf")
STATION="CASH (Cashmere, Christchurch NZ) - TC1 vertical geophone, ~18.8 Hz"

# ---- SAC + trace assembly ----------------------------------------------------
def read_sac(path):
    d=open(path,"rb").read()
    if len(d)<632: return None,None
    delta=struct.unpack(">f",d[0:4])[0]; npts=struct.unpack(">i",d[316:320])[0]
    if npts<=0 or 632+npts*4>len(d) or delta<=0: return None,None
    return np.frombuffer(d[632:632+npts*4],dtype=">f4").astype("f8"),delta

def get_trace(center, pre, post):
    """Return (t_rel[s], amp) over [center-pre, center+post], stitching hour files."""
    h0=center.replace(minute=0,second=0,microsecond=0)
    start=center-dt.timedelta(seconds=pre); end=center+dt.timedelta(seconds=post)
    h=h0-dt.timedelta(hours=1); ts=[]; ys=[]; fs=None
    while h<=end:
        p=os.path.join(ROOT,str(h.year),str(h.month),str(h.day),f"{h.hour}.sac")
        if os.path.exists(p):
            x,delta=read_sac(p)
            if x is not None:
                fs=1.0/delta
                t0=(h-center).total_seconds()
                t=t0+np.arange(len(x))*delta
                m=(t>=-pre)&(t<=post)
                ts.append(t[m]); ys.append(x[m])
        h+=dt.timedelta(hours=1)
    if not ts: return None,None,None
    t=np.concatenate(ts); y=np.concatenate(ys)
    o=np.argsort(t); return t[o],y[o],fs

def bp(y,fs,lo,hi):
    ny=fs/2.0; hi=min(hi,ny*0.95)
    sos=butter(4,[lo/ny,hi/ny],btype="band",output="sos")
    return sosfiltfilt(sos,y)

# ---- data --------------------------------------------------------------------
def load_matched():
    rows=[]
    for r in csv.DictReader(open(MATCH)):
        try:
            r["_t"]=dt.datetime.strptime(r["time"],"%Y-%m-%dT%H:%M:%S")
            r["_mag"]=float(r["mag"]); r["_dist"]=float(r["dist_km"])
            r["_snr"]=float(r["det_snr"]) if r["det_snr"] else 0.0
            r["_lag"]=float(r["lag_s"]) if r["lag_s"] else 0.0
            r["_det"]=r["detected"]=="1"
        except Exception: continue
        rows.append(r)
    return rows

# curated featured events (by exact origin time) — global giants + local jolts
FEATURED=[
    "2018-08-19T00:19:40",   # M8.2 Fiji - largest caught
    "2021-03-04T19:28:33",   # M8.1 Kermadec (the tsunami-warning sequence)
    "2020-06-18T12:49:53",   # M7.4 south Kermadec - SNR ~1050
    "2020-03-15T15:28:58",   # M5.1 Culverden - felt across Canterbury
    "2020-12-22T11:36:46",   # M3.8 5 km E of Christchurch - SNR 2742, right under sensor
    "2023-03-18T00:42:06",   # M4.7 near Akaroa - SNR 2249
]

def win_for(dist):
    if dist<300:  return 25,150, (1.5,9.0)
    if dist<1500: return 45,400, (1.0,8.0)
    return 90,1000,(0.7,6.0)

# ---- pages -------------------------------------------------------------------
def page_cover(pdf, rows):
    det=[r for r in rows if r["_det"]]
    big=max(det,key=lambda r:r["_mag"])
    loud=max(det,key=lambda r:r["_snr"])
    fig=plt.figure(figsize=(8.27,11.69))  # A4 portrait
    fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0.86),1,0.14,color="#1f3b57"))
    ax.text(0.5,0.935,"CASH Seismic Station",ha="center",va="center",
            fontsize=30,color="white",weight="bold")
    ax.text(0.5,0.895,"Most Significant Earthquakes Detected, 2016-2025",
            ha="center",va="center",fontsize=15,color="#cfe0f0")
    lines=[
        STATION,
        "",
        f"Archive span:   {min(r['_t'] for r in rows).date()}  to  {max(r['_t'] for r in rows).date()}",
        f"Catalogued events checked (GeoNet + USGS):   {len(rows):,}",
        f"Events matched to a CASH detection:   {len(det):,}",
        "",
        "Headline results",
        f"  - Largest earthquake recorded:   M{big['_mag']:.1f}  {big['place']}",
        f"      ({big['_dist']:.0f} km away, {big['_t']:%Y-%m-%d})",
        f"  - Strongest shaking recorded:   M{loud['_mag']:.1f}  {loud['place']}",
        f"      ({loud['_dist']:.0f} km away, signal-to-noise {loud['_snr']:.0f}x)",
        "",
        "This report features the recorded ground-motion waveforms for a",
        "selection of the largest distant earthquakes and the strongest local",
        "shaking captured by this single backyard-class seismometer.",
    ]
    y=0.81
    for ln in lines:
        wt="bold" if ln in("Headline results",) else "normal"
        sz=12.5 if ln.startswith("  ") or ln in("Headline results",) else 12
        ax.text(0.10,y,ln,ha="left",va="top",fontsize=sz,family="DejaVu Sans",weight=wt)
        y-=0.0235
    # embed capability figure if present
    cap=os.path.join(HERE,"fig_detection_capability.png")
    if os.path.exists(cap):
        ax.text(0.5,0.40,"Detection capability: which catalogued quakes CASH actually caught",
                ha="center",fontsize=10,style="italic",color="#444")
        im=plt.imread(cap)
        iax=fig.add_axes([0.12,0.07,0.76,0.31]); iax.axis("off"); iax.imshow(im)
    ax.text(0.5,0.025,f"Generated {dt.date.today().isoformat()}  -  analysis/generate_report.py",
            ha="center",fontsize=8,color="#888")
    pdf.savefig(fig); plt.close(fig)

def table_page(pdf, rows):
    det=[r for r in rows if r["_det"]]
    fig=plt.figure(figsize=(8.27,11.69)); ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5,0.96,"Top detected earthquakes",ha="center",fontsize=18,weight="bold")

    # fixed column widths shared by header and rows so everything lines up
    COLW=[(18,"left"),(6,"left"),(9,"right"),(8,"right"),(30,"left")]
    def fmt_row(cells):
        out=[]
        for txt,(w,align) in zip(cells,COLW):
            txt=str(txt)
            out.append(txt.ljust(w) if align=="left" else txt.rjust(w))
            out.append("  ")            # 2-space gutter between columns
        return "".join(out)

    def block(title, sel, y0, valcol, valhdr, valfmt):
        ax.text(0.07,y0,title,fontsize=13,weight="bold",color="#1f3b57")
        hdr=fmt_row(["Date (UTC)","Mag","Dist km",valhdr,"Location"])
        ax.text(0.07,y0-0.028,hdr,fontsize=8.5,family="monospace",weight="bold")
        yy=y0-0.050
        for r in sel:
            line=fmt_row([f"{r['_t']:%Y-%m-%d %H:%M}",f"M{r['_mag']:.1f}",
                          f"{r['_dist']:.0f}",valfmt(r),r['place'][:30]])
            ax.text(0.07,yy,line,fontsize=8.3,family="monospace")
            yy-=0.0205
        return yy

    a=sorted(det,key=lambda r:-r["_mag"])[:16]
    y=block("A.  Largest magnitude (global reach)",a,0.90,"_mag","SNR",
            lambda r:f"{r['_snr']:.0f}")
    b=sorted(det,key=lambda r:-r["_snr"])[:16]
    block("B.  Strongest shaking recorded at CASH (mostly local Canterbury)",b,
          y-0.04,"_snr","SNR",lambda r:f"{r['_snr']:.0f}")
    ax.text(0.07,0.05,"SNR = peak recorded amplitude / background noise level that hour. "
            "Local quakes a few km away ring the sensor far harder than distant giants.",
            fontsize=8,style="italic",color="#555",wrap=True)
    pdf.savefig(fig); plt.close(fig)

def event_page(pdf, r):
    pre,post,(lo,hi)=win_for(r["_dist"])
    det_t=r["_t"]+dt.timedelta(seconds=r["_lag"])
    t,y,fs=get_trace(det_t,pre,post)
    fig=plt.figure(figsize=(8.27,11.69))
    ax0=fig.add_axes([0,0.82,1,0.16]); ax0.axis("off")
    ax0.add_patch(plt.Rectangle((0,0),1,1,color="#eef3f8"))
    cat="LOCAL / REGIONAL" if r["_dist"]<300 else ("REGIONAL" if r["_dist"]<1500 else "TELESEISM (distant)")
    ax0.text(0.06,0.74,f"M{r['_mag']:.1f}  -  {r['place']}",fontsize=19,weight="bold",color="#1f3b57")
    ax0.text(0.06,0.45,f"{r['_t']:%Y-%m-%d %H:%M:%S} UTC     {cat}",fontsize=12,color="#333")
    ax0.text(0.06,0.18,f"Distance {r['_dist']:.0f} km   -   depth {r['depth']} km   -   "
             f"recorded SNR {r['_snr']:.0f}x   -   source {r['src'].upper()}",
             fontsize=11,color="#333")

    if t is None or len(t)<10:
        axw=fig.add_axes([0.1,0.4,0.8,0.3]); axw.axis("off")
        axw.text(0.5,0.5,"(waveform data unavailable)",ha="center")
        pdf.savefig(fig); plt.close(fig); return

    yf=bp(detrend(y,type="linear"),fs,lo,hi)
    # full window
    ax1=fig.add_axes([0.10,0.50,0.82,0.26])
    ax1.plot(t,yf,lw=0.4,color="#1f3b57")
    ax1.axvline(0,color="#d62728",lw=1.2,ls="--")
    ax1.text(0,ax1.get_ylim()[1]*0.92,"  CASH trigger",color="#d62728",fontsize=9,va="top")
    ax1.set_xlim(-pre,post); ax1.set_xlabel("seconds relative to detection")
    ax1.set_ylabel("ground motion (counts)")
    ax1.set_title(f"Recorded waveform   (band-pass {lo}-{hi} Hz)",fontsize=11)
    ax1.grid(alpha=0.25)

    # zoom on first ~ arrival
    zlo,zhi=(-10,40) if r["_dist"]<300 else (-20,120)
    mz=(t>=zlo)&(t<=zhi)
    ax2=fig.add_axes([0.10,0.16,0.82,0.24])
    ax2.plot(t[mz],yf[mz],lw=0.6,color="#1f3b57")
    ax2.axvline(0,color="#d62728",lw=1.2,ls="--")
    ax2.set_xlim(zlo,zhi); ax2.set_xlabel("seconds relative to detection")
    ax2.set_ylabel("ground motion (counts)")
    ax2.set_title("Zoom on arrival",fontsize=11); ax2.grid(alpha=0.25)

    peak=np.max(np.abs(yf))
    fig.text(0.10,0.085,f"Peak filtered amplitude {peak:.0f} counts.  "
             f"Wave travel time origin->CASH was {r['_lag']:.0f} s "
             f"({r['_dist']/max(r['_lag'],1):.1f} km/s apparent).",fontsize=9,color="#444")
    pdf.savefig(fig); plt.close(fig)

def main():
    rows=load_matched()
    bytime={r["time"]:r for r in rows}
    import eqmap, detectability_distance as dd
    with PdfPages(OUT) as pdf:
        page_cover(pdf,rows)
        table_page(pdf,rows)
        try:
            eqmap.map_page(pdf)          # world map summarising main detections
        except Exception as e:
            print(f"  WARN map page skipped: {e}")
        try:
            dd.detectability_page(pdf)   # detectability vs distance + core shadow
        except Exception as e:
            print(f"  WARN detectability page skipped: {e}")
        for tstr in FEATURED:
            r=bytime.get(tstr)
            if r and r["_det"]:
                event_page(pdf,r)
            else:
                print(f"  WARN featured event not found/detected: {tstr}")
    print(f"DONE -> {OUT}")

if __name__=="__main__":
    main()

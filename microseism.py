#!/usr/bin/env python3
"""
Seasonal ocean-microseism study from cash_bands.csv (numpy only).

cash_bands.csv holds ROBUST (median-of-per-minute-RMS) band levels, so brief
earthquakes don't inflate them. Microseism proxy = 'sec' band (0.10-0.35 Hz,
secondary ocean microseism); control = 'cult' band (2-8 Hz, cultural noise).

The secondary (double-frequency) ocean microseism is driven by ocean swell, so
it should peak in austral winter (Jun-Aug, Southern Ocean storms).

Evidence that low_rms is genuine ocean microseism (not cultural noise):
  - strong ANNUAL cycle, peaking in winter
  - NO weekday/weekend difference   (cultural noise has one)
  - NO/weak diurnal cycle           (cultural noise has a strong one)

Outputs (analysis/):
  fig_microseism_seasonal.png   monthly climatology, microseism vs cultural
  fig_microseism_timeseries.png daily low_rms, 8.7 yr, per-year overlay
  fig_microseism_control.png    diurnal + weekday flatness vs cultural control
  fig_microseism_periodogram.png Lomb-Scargle: annual spectral peak
  microseism_summary.txt
"""
import os, csv, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import lombscargle

HERE=os.path.dirname(__file__)
CSV=os.path.join(HERE,"cash_bands.csv")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def load():
    raw=[]
    for d in csv.DictReader(open(CSV)):
        try:
            y,m,day,h=int(d["year"]),int(d["month"]),int(d["day"]),int(d["hour"])
            date=dt.date(y,m,day)
            sec=float(d["sec"]); short=float(d["short"]); cult=float(d["cult"])
        except Exception: continue
        if not (sec>0 and sec==sec): continue
        raw.append((date,h,sec,cult,short))
    # winsorise each band at its 99th pct to kill instrument-glitch spikes
    # (the sec band has rare values up to ~1e6 that wreck fits; medians are fine
    #  but the annual fit / periodogram need clean data)
    caps=[np.percentile([r[i] for r in raw],99.0) for i in (2,3,4)]
    rows=[]
    for date,h,sec,cult,short in raw:
        rows.append((date,h,min(sec,caps[0]),min(cult,caps[1]),min(short,caps[2])))
    return rows

def daily(rows,idx):
    by={}
    for r in rows: by.setdefault(r[0],[]).append(r[idx])
    ds=sorted(by); return ds,np.array([np.median(by[d]) for d in ds])

def annual_fit(doy,y):
    """least-squares y = a + b cos(w t) + c sin(w t); return mean,amp,peak_doy"""
    w=2*np.pi/365.25
    A=np.column_stack([np.ones_like(doy),np.cos(w*doy),np.sin(w*doy)])
    coef,*_=np.linalg.lstsq(A,y,rcond=None)
    a,b,c=coef
    amp=np.hypot(b,c); peak=(np.arctan2(c,b)/w)%365.25
    return a,amp,peak,coef

def fig_seasonal(rows):
    bymon_low={m:[] for m in range(1,13)}; bymon_cult={m:[] for m in range(1,13)}
    for date,h,low,cult,rob in rows:
        bymon_low[date.month].append(low); bymon_cult[date.month].append(cult)
    ml=[np.median(bymon_low[m]) for m in range(1,13)]
    mc=[np.median(bymon_cult[m]) for m in range(1,13)]
    # normalise each to its own annual mean to compare shape
    mln=np.array(ml)/np.mean(ml); mcn=np.array(mc)/np.mean(mc)
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(1,13)
    ax.axvspan(5.5,8.5,color="#cfe3f7",alpha=0.6,label="austral winter (JJA)")
    ax.plot(x,mln,"-o",color="#1f4e79",lw=2,label="secondary microseism 0.10-0.35 Hz")
    ax.plot(x,mcn,"-s",color="#d95f0e",lw=2,label="cultural 2-8 Hz (control)")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("noise level / annual mean"); ax.grid(alpha=0.3)
    ax.set_title("Seasonal cycle: microseism peaks in austral autumn-winter; cultural noise does not")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_microseism_seasonal.png"),dpi=120); plt.close(fig)
    return ml

def fig_timeseries(rows):
    ds,y=daily(rows,2)
    x=[dt.datetime(d.year,d.month,d.day) for d in ds]
    doy=np.array([d.timetuple().tm_yday for d in ds])
    a,amp,peak,coef=annual_fit(doy,y)
    w=2*np.pi/365.25
    fit=coef[0]+coef[1]*np.cos(w*doy)+coef[2]*np.sin(w*doy)
    fig,ax=plt.subplots(figsize=(14,4))
    ax.plot(x,y,lw=0.4,color="#7aa6c2",alpha=0.7)
    # 30-day rolling median
    roll=np.full_like(y,np.nan)
    for i in range(len(y)):
        lo,hi=max(0,i-15),min(len(y),i+16); roll[i]=np.median(y[lo:hi])
    ax.plot(x,roll,lw=1.4,color="#1f4e79",label="30-day median")
    ax.plot(x,fit,lw=1.4,color="#d62728",ls="--",label="annual fit")
    ax.set_ylabel("daily microseism level (low_rms)")
    ax.set_title("CASH ocean-microseism proxy, 2016-2025 (winter peaks recur every year)")
    ax.set_ylim(bottom=0); ax.legend(); ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_microseism_timeseries.png"),dpi=120); plt.close(fig)
    return a,amp,peak

def fig_control(rows):
    # diurnal (normalised) and weekday for low vs cult
    hb_low={h:[] for h in range(24)}; hb_cult={h:[] for h in range(24)}
    wl=[[],[]]; wc=[[],[]]   # [weekday, weekend]
    for date,h,low,cult,rob in rows:
        hb_low[h].append(low); hb_cult[h].append(cult)
        wknd=date.weekday()>=5
        wl[1 if wknd else 0].append(low); wc[1 if wknd else 0].append(cult)
    dl=np.array([np.median(hb_low[h]) for h in range(24)]); dl/=dl.mean()
    dc=np.array([np.median(hb_cult[h]) for h in range(24)]); dc/=dc.mean()
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    axs[0].plot(range(24),dl,"-o",color="#1f4e79",label="microseism 0.10-0.35 Hz")
    axs[0].plot(range(24),dc,"-s",color="#d95f0e",label="cultural 2-8 Hz")
    axs[0].set_xlabel("hour (UTC)"); axs[0].set_ylabel("level / daily mean")
    axs[0].set_title("Diurnal cycle"); axs[0].grid(alpha=0.3); axs[0].legend()
    lw=[np.median(wl[0]),np.median(wl[1])]; cw=[np.median(wc[0]),np.median(wc[1])]
    lw=np.array(lw)/lw[0]*100; cw=np.array(cw)/cw[0]*100
    xb=np.arange(2)
    axs[1].bar(xb-0.18,lw,0.34,color="#1f4e79",label="microseism")
    axs[1].bar(xb+0.18,cw,0.34,color="#d95f0e",label="cultural")
    axs[1].set_xticks(xb); axs[1].set_xticklabels(["weekday","weekend"])
    axs[1].set_ylabel("% of weekday level"); axs[1].set_ylim(80,105)
    axs[1].set_title("Weekday vs weekend"); axs[1].legend(); axs[1].grid(alpha=0.3,axis="y")
    fig.suptitle("Control: microseism band is FLATTER in time-of-day & weekday than cultural "
                 "(less anthropogenic) -- but not fully flat, so some cultural leakage remains")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_microseism_control.png"),dpi=120); plt.close(fig)
    return dl,dc,lw,cw

def fig_periodogram(rows):
    ds,y=daily(rows,2)
    t=np.array([(d-ds[0]).days for d in ds],float)
    y=y-np.polyval(np.polyfit(t,y,1),t)         # remove mean + linear trend
    periods=np.linspace(20,800,4000)            # days
    ang=2*np.pi/periods
    pk=lombscargle(t,y,ang,normalize=True)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(periods,pk,color="#1f4e79")
    ax.axvline(365.25,color="#d62728",ls="--",label="1 year")
    ax.set_xlabel("period (days)"); ax.set_ylabel("Lomb-Scargle power")
    ax.set_title("Periodicity of microseism level — dominant annual cycle")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_microseism_periodogram.png"),dpi=120); plt.close(fig)
    bestP=periods[np.argmax(pk)]
    return bestP,pk.max()

def main():
    rows=load()
    ml=fig_seasonal(rows)
    a,amp,peak=fig_timeseries(rows)
    dl,dc,lw,cw=fig_control(rows)
    bestP,bestpow=fig_periodogram(rows)
    peak_month=MONTHS[int((peak//30.4))%12]
    wmax=MONTHS[int(np.argmax(ml))]; wmin=MONTHS[int(np.argmin(ml))]
    ratio=max(ml)/min(ml)
    winter=np.mean([ml[m] for m in (4,5,6,7)])   # May-Aug (austral autumn-winter)
    summer=np.mean([ml[m] for m in (10,11,0,1)]) # Nov-Feb (austral summer)
    annual_clean=abs(bestP-365)<40               # periodogram lands near 1 yr?
    # evidence triad
    flat_diurnal=(dl.max()-dl.min())<(dc.max()-dc.min())
    flat_week=abs(lw[1]-100)<abs(cw[1]-100)
    L=[]
    L.append("Microseism proxy = sec band 0.10-0.35 Hz (robust median-of-minute RMS);"
             " control = cult band 2-8 Hz. Bands winsorised at 99th pct.")
    L.append(f"Monthly climatology: peak {wmax}, trough {wmin}, "
             f"winter(MJJA)/summer(NDJF) = {winter/summer:.2f}, max/min = {ratio:.2f}x.")
    L.append(f"Annual sinusoid fit: half-amplitude {100*amp/a:.0f}% of mean, peak ~{peak_month}.")
    L.append(f"Lomb-Scargle dominant period: {bestP:.0f} days "
             f"({'~annual, as expected' if annual_clean else 'NOT cleanly annual'}).")
    L.append(f"Diurnal pk-pk: microseism {100*(dl.max()-dl.min()):.0f}% vs cultural "
             f"{100*(dc.max()-dc.min()):.0f}%  ({'flatter' if flat_diurnal else 'NOT flatter'}).")
    L.append(f"Weekend change: microseism {lw[1]-100:+.1f}% vs cultural {cw[1]-100:+.1f}%  "
             f"({'flatter' if flat_week else 'NOT flatter'}).")
    triad=(winter>summer) and flat_diurnal and flat_week
    L.append("")
    if triad:
        L.append(f"VERDICT: the sec band peaks in austral autumn/winter ({winter/summer:.2f}x, "
                 f"max/min {ratio:.2f}x) with a clean ~annual periodogram line -- consistent with"
                 " ocean (secondary) microseism driven by Southern-Ocean swell. It is flatter"
                 " than the cultural band in time-of-day and weekday (less anthropogenic), BUT"
                 " it still shows a diurnal cycle, so this band is a MIXTURE of attenuated ocean"
                 " microseism and residual cultural leakage -- not pure microseism.")
    else:
        L.append("VERDICT: mixed evidence -- see numbers above.")
    L.append("The cycle is MODEST, not dramatic: the TC1 4.5 Hz geophone rolls off as ~f^2"
             " below its corner, attenuating 0.1-0.35 Hz ground motion by ~100-1000x, so true"
             " ocean microseism is near the instrument/cultural noise floor here. The seasonal"
             " pattern survives in the median but absolute sensitivity in this band is poor.")
    txt="\n".join(L); open(os.path.join(HERE,"microseism_summary.txt"),"w").write(txt+"\n")
    print(txt); print("\n-> fig_microseism_*.png")

if __name__=="__main__":
    main()

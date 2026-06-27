#!/usr/bin/env python3
"""
Noise / lockdown study + completeness map from cash_hourly_metrics.csv.
numpy-only (no pandas).

Produces (in analysis/):
  fig_completeness.png   year x day-of-year heatmap of hourly coverage
  fig_daily_noise.png    long-term daily robust noise with 2020 lockdown marked
  fig_diurnal.png        hour-of-day (UTC) noise, all years
  fig_weekly.png         weekday vs weekend noise
  fig_lockdown_2020.png  zoom on Oct2019-Sep2020 with NZ lockdown window
  summary.txt            key numbers
"""
import os, csv, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "cash_hourly_metrics.csv")

LD_START = dt.date(2020, 3, 26)   # NZ COVID-19 Alert Level 4
LD_END   = dt.date(2020, 5, 13)   # end of Level 3

def fnum(s):
    try:
        v = float(s)
        return v
    except Exception:
        return np.nan

def load():
    rows = []
    with open(CSV) as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                y, m, day, h = int(d["year"]), int(d["month"]), int(d["day"]), int(d["hour"])
                date = dt.date(y, m, day)
            except Exception:
                continue
            npts = fnum(d["npts"])
            rr = fnum(d["robust_rms"])
            rows.append(dict(date=date, hour=h, npts=npts,
                             full=fnum(d["full_rms"]), robust=rr,
                             low=fnum(d["low_rms"]), cult=fnum(d["cult_rms"]),
                             clip=fnum(d["clip_frac"]), dead=fnum(d["dead"]),
                             valid=(npts==npts and npts>0 and rr==rr and rr>0)))
    return rows

# ---- daily aggregation helpers ----------------------------------------------
def daily_median(rows, key="robust"):
    by = {}
    for r in rows:
        if r["valid"]:
            by.setdefault(r["date"], []).append(r[key])
    dates = sorted(by)
    vals = np.array([np.median(by[d]) for d in dates])
    return dates, vals

def median_in_range(rows, a, b, key="robust"):
    v = [r[key] for r in rows if r["valid"] and a <= r["date"] < b]
    return float(np.median(v)) if v else None

# ---- figures -----------------------------------------------------------------
def fig_completeness(rows):
    years = sorted({r["date"].year for r in rows})
    yi = {y:i for i,y in enumerate(years)}
    cnt = np.zeros((len(years), 366))
    for r in rows:
        if r["valid"]:
            doy = r["date"].timetuple().tm_yday
            cnt[yi[r["date"].year], doy-1] += 1
    grid = cnt/24.0
    grid[grid==0] = np.nan
    fig, ax = plt.subplots(figsize=(14,4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    ax.set_xticks([0,31,59,90,120,151,181,212,243,273,304,334])
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_xlabel("month"); ax.set_title("CASH data completeness — fraction of 24 hourly files present per day")
    fig.colorbar(im, ax=ax, label="hours present / 24", shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_completeness.png"), dpi=120); plt.close(fig)

def fig_daily_noise(rows):
    dates, vals = daily_median(rows)
    x = [dt.datetime(d.year,d.month,d.day) for d in dates]
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(x, vals, lw=0.5, color="#1f4e79", alpha=0.8)
    roll = rolling_median(vals, 30)
    ax.plot(x, roll, lw=1.8, color="#d62728", label="30-day median")
    ax.axvspan(dt.datetime(2020,3,26), dt.datetime(2020,5,13), color="orange", alpha=0.35, label="NZ lockdown 2020")
    ax.set_ylabel("daily robust noise (counts RMS)")
    ax.set_title("CASH long-term ambient noise (daily median of robust hourly RMS)")
    ax.set_ylim(bottom=0); ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_daily_noise.png"), dpi=120); plt.close(fig)

def fig_diurnal(rows):
    byh = {h:[] for h in range(24)}
    for r in rows:
        if r["valid"]: byh[r["hour"]].append(r["robust"])
    med = [np.median(byh[h]) if byh[h] else np.nan for h in range(24)]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(24), med, color="#2c7fb8")
    ax.set_xlabel("hour (UTC)   —   NZ local = UTC +12 (+13 in summer)")
    ax.set_ylabel("median robust noise"); ax.set_title("Diurnal noise cycle (all years)")
    ax.set_xticks(range(0,24,2))
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_diurnal.png"), dpi=120); plt.close(fig)

def fig_weekly(rows):
    byd = {i:[] for i in range(7)}
    for r in rows:
        if r["valid"]: byd[r["date"].weekday()].append(r["robust"])
    med = [np.median(byd[i]) for i in range(7)]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(7), med, color=["#2c7fb8"]*5+["#d95f0e"]*2)
    ax.set_xticks(range(7)); ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_ylabel("median robust noise"); ax.set_title("Weekday vs weekend noise (UTC days)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_weekly.png"), dpi=120); plt.close(fig)

def fig_lockdown(rows):
    a, b = dt.date(2019,10,1), dt.date(2020,9,30)
    sub = [r for r in rows if r["valid"] and a <= r["date"] <= b]
    dates, vals = daily_median(sub)
    if not dates: return
    x = [dt.datetime(d.year,d.month,d.day) for d in dates]
    roll = rolling_median(vals, 7)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(x, vals, lw=0.6, color="#1f4e79", alpha=0.7)
    ax.plot(x, roll, lw=2, color="#d62728", label="7-day median")
    ax.axvspan(dt.datetime(2020,3,26), dt.datetime(2020,5,13), color="orange", alpha=0.35, label="NZ lockdown")
    ax.set_title("Zoom: ambient noise around the 2020 NZ lockdown")
    ax.set_ylabel("daily robust noise"); ax.set_ylim(bottom=0); ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_lockdown_2020.png"), dpi=120); plt.close(fig)

def rolling_median(vals, win):
    out = np.full(len(vals), np.nan)
    half = win//2
    for i in range(len(vals)):
        lo, hi = max(0,i-half), min(len(vals),i+half+1)
        seg = vals[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= max(3, win//3):
            out[i] = np.median(seg)
    return out

def summary(rows):
    L=[]
    tot=len(rows); valid=sum(1 for r in rows if r["valid"])
    dmin=min(r["date"] for r in rows); dmax=max(r["date"] for r in rows)
    span=(dmax-dmin).days; expected=span*24
    L.append(f"Files in CSV: {tot}")
    L.append(f"Valid hours: {valid} ({100*valid/tot:.1f}%)")
    L.append(f"Date span: {dmin} .. {dmax} ({span} days)")
    L.append(f"Coverage vs continuous 24/7: {100*valid/expected:.1f}%")
    L.append(f"Dead/flat hours: {sum(1 for r in rows if r['dead']==1)}")
    L.append(f"Clipped hours (clip_frac>0.5%): {sum(1 for r in rows if r['clip']==r['clip'] and r['clip']>0.005)}")
    base=median_in_range(rows, dt.date(2019,9,1), dt.date(2020,3,1))
    lock=median_in_range(rows, dt.date(2020,3,26), dt.date(2020,5,13))
    if base and lock:
        L.append(f"Median robust noise pre-COVID (Sep19-Mar20): {base:.2f}")
        L.append(f"Median robust noise during lockdown: {lock:.2f}")
        L.append(f"Lockdown change: {100*(lock-base)/base:+.1f}%")
    byh={h:[] for h in range(24)}
    for r in rows:
        if r["valid"]: byh[r["hour"]].append(r["robust"])
    med={h:np.median(v) for h,v in byh.items() if v}
    qh=min(med,key=med.get); nh=max(med,key=med.get)
    L.append(f"Quietest UTC hour {qh}:00 ({med[qh]:.2f}); noisiest {nh}:00 ({med[nh]:.2f}); ratio {med[nh]/med[qh]:.2f}x")
    wd=[r["robust"] for r in rows if r["valid"] and r["date"].weekday()<5]
    we=[r["robust"] for r in rows if r["valid"] and r["date"].weekday()>=5]
    wdm, wem=np.median(wd), np.median(we)
    L.append(f"Weekday median {wdm:.2f} vs weekend {wem:.2f} ({100*(wem-wdm)/wdm:+.1f}%)")
    txt="\n".join(L)
    open(os.path.join(HERE,"summary.txt"),"w").write(txt+"\n")
    print(txt)

def main():
    rows=load()
    fig_completeness(rows); fig_daily_noise(rows); fig_diurnal(rows)
    fig_weekly(rows); fig_lockdown(rows); summary(rows)
    print("\nFigures written to analysis/*.png")

if __name__=="__main__":
    main()

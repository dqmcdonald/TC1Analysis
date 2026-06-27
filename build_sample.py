#!/usr/bin/env python3
"""
Build the bundled sample dataset (run once, needs the full CASH archive).

Copies the SAC hours for a handful of representative events into sample_data/CASH/
and writes sample_data/cash_detected_catalog.csv (those events, with lat/lon
embedded) so a fresh clone can run tc1_explorer.py out of the box.
"""
import os, csv, shutil, datetime as dt
import config, eqmap

HERE=os.path.dirname(os.path.abspath(__file__))
SRC=config.DATA_ROOT
OUTDIR=os.path.join(HERE,"sample_data"); OUTCASH=os.path.join(OUTDIR,"CASH")
COMB=os.path.join(HERE,"combined_catalog.csv")

# (origin, pre_seconds, post_seconds) -- chosen to span each event's useful window
EVENTS=[
 ("2023-09-19T21:14:49",4000, 800),   # M6.0 Geraldine        local crustal
 ("2020-12-22T11:36:46",4000, 600),   # M3.8 Christchurch     strong local
 ("2020-03-15T15:28:58",4000, 800),   # M5.1 Culverden        local
 ("2022-01-12T10:49:36",4000,1800),   # M5.8 Stratford        regional deep
 ("2021-03-04T19:28:33",4000,1800),   # M8.1 Kermadec         great regional
 ("2017-11-19T22:43:29",4000,1800),   # M7.0 New Caledonia    teleseism
 ("2018-08-19T00:19:40",4000,1800),   # M8.2 Fiji             deep teleseism
 ("2025-03-28T06:20:52",4000,5600),   # M7.7 Myanmar          far teleseism + surface
]

def hours(o,pre,post):
    h=(o-dt.timedelta(seconds=pre)).replace(minute=0,second=0,microsecond=0)
    end=o+dt.timedelta(seconds=post); out=[]
    while h<=end: out.append(h); h+=dt.timedelta(hours=1)
    return out

def main():
    if os.path.isdir(OUTDIR): shutil.rmtree(OUTDIR)
    os.makedirs(OUTCASH)
    coords=eqmap.load_coords()
    byt={r["time"]:r for r in csv.DictReader(open(COMB))}
    n_files=0; rows=[]
    for tstr,pre,post in EVENTS:
        o=dt.datetime.strptime(tstr,"%Y-%m-%dT%H:%M:%S")
        for h in hours(o,pre,post):
            rel=os.path.join(str(h.year),str(h.month),str(h.day),f"{h.hour}.sac")
            s=os.path.join(SRC,rel); d=os.path.join(OUTCASH,rel)
            if os.path.exists(s):
                os.makedirs(os.path.dirname(d),exist_ok=True); shutil.copy2(s,d); n_files+=1
        r=byt.get(tstr)
        if r:
            c=coords.get(tstr[:19]); r=dict(r)
            r["lat"]="" if not c else round(c[0],4); r["lon"]="" if not c else round(c[1],4)
            rows.append(r)
    cols=list(rows[0].keys())
    with open(os.path.join(OUTDIR,"cash_detected_catalog.csv"),"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=cols); w.writeheader(); w.writerows(rows)
    sz=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(OUTDIR) for f in fs)
    print(f"sample: {len(rows)} events, {n_files} SAC files, {sz/1e6:.1f} MB -> {OUTDIR}")

if __name__=="__main__":
    main()

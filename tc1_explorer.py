#!/usr/bin/env python3
"""
TC1 Detection Explorer — a GUI to browse CASH's detected-earthquake catalogue.

Features
  - sortable table (click a column header) of detected earthquakes
  - filter by place name (substring), magnitude range, distance range
  - select a row -> the epicentre is shown on a CASH-centred map, and the
    recorded trace is plotted, annotated with magnitude / distance / depth /
    detection method / SNR and TauP P & S arrival times.

Run:  python tc1_explorer.py        (needs the tf venv: obspy, cartopy, tkinter)
Reads cash_detected_catalog.csv (run combined_catalog.py first) + the SAC archive.
"""
import os, csv, datetime as dt
import numpy as np
from scipy.signal import detrend
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config
import generate_report as G        # get_trace / read_sac / bp (numpy/scipy only)
import phases as ph                # TauP tt_p / tt_s
import eqmap                       # load_coords

HERE=os.path.dirname(os.path.abspath(__file__))
CAT=os.path.join(HERE,"cash_detected_catalog.csv")

# ---------------------------------------------------------------- data
def load_rows():
    coords=eqmap.load_coords(); rows=[]
    if not os.path.exists(CAT): return rows
    for r in csv.DictReader(open(CAT)):
        try: m=float(r["mag"]); dist=float(r["dist_km"]); depth=float(r["depth"] or 0)
        except Exception: continue
        c=coords.get(r["time"][:19])
        rows.append(dict(time=r["time"],mag=m,dist=dist,depth=depth,
            method=r["method"],place=r["place"],regime=r["regime"],
            lat=(c[0] if c else None),lon=(c[1] if c else None),
            body_snr=r.get("body_snr",""),surf_score=r.get("surface_score",""),
            body_lag=r.get("body_lag","")))
    return rows

# ---------------------------------------------------------------- plotting
def plot_event(fig, ev):
    fig.clf()
    # ----- map (azimuthal-equidistant, centred on CASH) -----
    axm=fig.add_subplot(1,2,1,projection=ccrs.AzimuthalEquidistant(
        central_longitude=config.CASH_LON,central_latitude=config.CASH_LAT))
    axm.set_global()
    axm.add_feature(cfeature.OCEAN,facecolor="#dceaf2")
    axm.add_feature(cfeature.LAND,facecolor="#efe9dc")
    axm.coastlines(resolution="110m",color="#9aa7b0",linewidth=0.5)
    pc=ccrs.PlateCarree()
    axm.scatter([config.CASH_LON],[config.CASH_LAT],marker="*",s=300,c="#cc0000",
                edgecolors="white",linewidths=0.8,transform=pc,zorder=6)
    if ev["lat"] is not None:
        axm.plot([config.CASH_LON,ev["lon"]],[config.CASH_LAT,ev["lat"]],
                 transform=ccrs.Geodetic(),color="#d9a441",lw=1.2,zorder=4)
        col={"body":"#2c7fb8","surface":"#1a9850","body+surface":"#7b3294"}.get(ev["method"],"#d62728")
        axm.scatter([ev["lon"]],[ev["lat"]],s=120,c=col,edgecolors="white",
                    linewidths=0.6,transform=pc,zorder=7)
    axm.set_title(f"M{ev['mag']:.1f}  {ev['place']}\n{ev['dist']:.0f} km from CASH",fontsize=10)

    # ----- trace -----
    axt=fig.add_subplot(1,2,2)
    o=dt.datetime.strptime(ev["time"],"%Y-%m-%dT%H:%M:%S")
    dist,depth=ev["dist"],ev["depth"]
    P=ph.tt_p(dist,depth); S=ph.tt_s(dist,depth)
    if dist<300:    band=(1.5,9.0)
    elif dist<1500: band=(1.0,8.0)
    else:           band=(0.8,6.0)
    pre=60; post=S+200
    t,y,fs=G.get_trace(o,pre,post)
    if t is None or len(t)<10:
        axt.text(0.5,0.5,"waveform unavailable",ha="center",va="center",transform=axt.transAxes)
    else:
        yb=G.bp(detrend(y,type="linear"),fs,band[0],band[1])
        axt.plot(t,yb,lw=0.4,color="#1f3b57")
        for tt,c,nm in [(P,"#1a9850","P"),(S,"#d62728","S")]:
            if -pre<=tt<=post:
                axt.axvline(tt,color=c,ls="--",lw=1.1)
                axt.text(tt,axt.get_ylim()[1]*0.9,f" {nm}",color=c,fontsize=9,weight="bold")
        axt.set_xlim(max(-pre,P-90),post)
        axt.set_xlabel("seconds after origin (UTC)"); axt.set_ylabel("counts")
        peak=float(np.max(np.abs(yb)))
        info=(f"M{ev['mag']:.1f}   {ev['dist']:.0f} km   depth {ev['depth']:.0f} km\n"
              f"method: {ev['method']}\n"
              f"SNR body {ev['body_snr'] or '-'} / surf {ev['surf_score'] or '-'}\n"
              f"TauP  P {P:.0f}s   S {S:.0f}s   S-P {S-P:.0f}s\n"
              f"peak {peak:.0f} counts  (band {band[0]}-{band[1]} Hz)")
        axt.text(0.98,0.97,info,ha="right",va="top",transform=axt.transAxes,fontsize=8,
                 family="monospace",bbox=dict(boxstyle="round",fc="#fffbe6",ec="#ccc",alpha=0.9))
    axt.set_title(f"{ev['time']} UTC — recorded trace",fontsize=10)
    fig.tight_layout()

# ---------------------------------------------------------------- GUI
COLS=[("time","Time (UTC)",150,False),("mag","Mag",55,True),("dist","Dist km",75,True),
      ("depth","Depth",60,True),("method","Method",95,False),("place","Location",260,False)]

class Explorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TC1 Detection Explorer — CASH")
        self.geometry("1320x780")
        self.all_rows=load_rows()
        self.rows=list(self.all_rows)
        self._build()
        self.populate()

    def _build(self):
        # filter bar
        top=ttk.Frame(self,padding=6); top.pack(side=tk.TOP,fill=tk.X)
        self.v={k:tk.StringVar() for k in("name","mmin","mmax","dmin","dmax")}
        def field(lbl,key,w=8):
            ttk.Label(top,text=lbl).pack(side=tk.LEFT,padx=(8,2))
            e=ttk.Entry(top,textvariable=self.v[key],width=w); e.pack(side=tk.LEFT)
            e.bind("<Return>",lambda _:self.apply_filter())
        field("Name contains:","name",18)
        field("Mag ≥","mmin",5); field("≤","mmax",5)
        field("Dist(km) ≥","dmin",7); field("≤","dmax",7)
        ttk.Button(top,text="Apply",command=self.apply_filter).pack(side=tk.LEFT,padx=8)
        ttk.Button(top,text="Reset",command=self.reset_filter).pack(side=tk.LEFT)
        self.count=ttk.Label(top,text=""); self.count.pack(side=tk.RIGHT,padx=10)

        # main split: table | plots
        pan=ttk.Panedwindow(self,orient=tk.HORIZONTAL); pan.pack(fill=tk.BOTH,expand=True)
        left=ttk.Frame(pan); pan.add(left,weight=1)
        self.tree=ttk.Treeview(left,columns=[c[0] for c in COLS],show="headings",selectmode="browse")
        for key,lbl,w,num in COLS:
            self.tree.heading(key,text=lbl,command=lambda k=key,n=num:self.sort_by(k,n))
            self.tree.column(key,width=w,anchor=("center" if num else "w"))
        sb=ttk.Scrollbar(left,orient=tk.VERTICAL,command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); sb.pack(side=tk.RIGHT,fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>",self.on_select)

        right=ttk.Frame(pan); pan.add(right,weight=2)
        self.fig=Figure(figsize=(8,7))
        self.canvas=FigureCanvasTkAgg(self.fig,master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self._sort=(None,False)
        self.item2ev={}

    def populate(self):
        self.tree.delete(*self.tree.get_children()); self.item2ev={}
        for ev in self.rows:
            iid=self.tree.insert("","end",values=(ev["time"],f"{ev['mag']:.1f}",
                f"{ev['dist']:.0f}",f"{ev['depth']:.0f}",ev["method"],ev["place"]))
            self.item2ev[iid]=ev
        self.count.config(text=f"{len(self.rows)} of {len(self.all_rows)} detections")

    def sort_by(self,key,numeric):
        rev = not self._sort[1] if self._sort[0]==key else False
        self.rows.sort(key=lambda e:(e[key] if numeric else str(e[key]).lower()),reverse=rev)
        self._sort=(key,rev); self.populate()

    def apply_filter(self):
        def num(s):
            try: return float(s)
            except Exception: return None
        nm=self.v["name"].get().strip().lower()
        mn,mx=num(self.v["mmin"].get()),num(self.v["mmax"].get())
        dn,dx=num(self.v["dmin"].get()),num(self.v["dmax"].get())
        out=[]
        for e in self.all_rows:
            if nm and nm not in e["place"].lower(): continue
            if mn is not None and e["mag"]<mn: continue
            if mx is not None and e["mag"]>mx: continue
            if dn is not None and e["dist"]<dn: continue
            if dx is not None and e["dist"]>dx: continue
            out.append(e)
        self.rows=out
        if self._sort[0]: self.rows.sort(key=lambda e:(e[self._sort[0]]
            if dict((c[0],c[3]) for c in COLS)[self._sort[0]] else str(e[self._sort[0]]).lower()),
            reverse=self._sort[1])
        self.populate()

    def reset_filter(self):
        for var in self.v.values(): var.set("")
        self.rows=list(self.all_rows); self.populate()

    def on_select(self,_):
        sel=self.tree.selection()
        if not sel: return
        ev=self.item2ev.get(sel[0])
        if not ev: return
        try:
            plot_event(self.fig,ev); self.canvas.draw()
        except Exception as e:
            self.fig.clf(); ax=self.fig.add_subplot(111)
            ax.text(0.5,0.5,f"plot error:\n{e}",ha="center",va="center",transform=ax.transAxes)
            self.canvas.draw()

def main():
    Explorer().mainloop()

if __name__=="__main__":
    main()

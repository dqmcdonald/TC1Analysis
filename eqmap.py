#!/usr/bin/env python3
"""
Station-centred map summarising CASH's main earthquake detections (cartopy).

Azimuthal-equidistant projection centred on CASH: distance from the centre = true
great-circle distance, so range rings (30/60/90 deg) read directly and every
gold arc is a real great-circle path. Events coloured by detection method, sized
by magnitude. Coordinates recovered from cached raw catalogues.

map_page(pdf) draws into an existing PdfPages (used by generate_report.py).
Run directly -> analysis/CASH_detection_map.png + .pdf  (use the tf venv python).
"""
import os, csv, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE=os.path.dirname(__file__)
CATS=os.path.join(HERE,"catalogs")
COMB=os.path.join(HERE,"combined_catalog.csv")
CASH_LAT,CASH_LON=-43.567,172.622
MAP_MINMAG=5.5

def load_coords():
    d={}
    for f in glob.glob(os.path.join(CATS,"usgs_*.csv")):
        for r in csv.DictReader(open(f)):
            try: d[r["time"][:19]]=(float(r["latitude"]),float(r["longitude"]))
            except Exception: pass
    for f in glob.glob(os.path.join(CATS,"geonet_*.txt")):
        for line in open(f):
            if line.startswith("#") or "|" not in line: continue
            p=line.split("|")
            try: d[p[1][:19]]=(float(p[2]),float(p[3]))
            except Exception: pass
    return d

def load_detected():
    coords=load_coords(); evs=[]; nomatch=0; nlocal=0
    for r in csv.DictReader(open(COMB)):
        if r["detected_any"]!="1": continue
        c=coords.get(r["time"][:19])
        if c is None: nomatch+=1; continue
        if float(r["dist_km"])<150: nlocal+=1
        evs.append(dict(lat=c[0],lon=c[1],mag=float(r["mag"]),method=r["method"],
                        dist=float(r["dist_km"]),place=r["place"]))
    return evs,nomatch,nlocal

def ring(lat0,lon0,d_deg,n=361):
    p0,l0,d=math.radians(lat0),math.radians(lon0),math.radians(d_deg)
    az=np.radians(np.linspace(0,360,n))
    lat=np.arcsin(np.sin(p0)*np.cos(d)+np.cos(p0)*np.sin(d)*np.cos(az))
    lon=l0+np.arctan2(np.sin(az)*np.sin(d)*np.cos(p0),np.cos(d)-np.sin(p0)*np.sin(lat))
    return np.degrees(lon),np.degrees(lat)

def dest(lat0,lon0,d_deg,az_deg):
    p0,l0,d,az=map(math.radians,[lat0,lon0,d_deg,az_deg])
    lat=math.asin(math.sin(p0)*math.cos(d)+math.cos(p0)*math.sin(d)*math.cos(az))
    lon=l0+math.atan2(math.sin(az)*math.sin(d)*math.cos(p0),math.cos(d)-math.sin(p0)*math.sin(lat))
    return math.degrees(lon),math.degrees(lat)

def map_page(pdf):
    evs,nomatch,nlocal=load_detected()
    big=[e for e in evs if e["mag"]>=MAP_MINMAG]
    print(f"detected with coords: {len(evs)} (unmatched {nomatch}); plotting M>={MAP_MINMAG}: {len(big)}")
    style={"body+surface":("#7b3294","both methods"),
           "body":("#2c7fb8","body-wave"),"surface":("#1a9850","surface-wave")}
    pc=ccrs.PlateCarree()
    proj=ccrs.AzimuthalEquidistant(central_longitude=CASH_LON,central_latitude=CASH_LAT)
    fig=plt.figure(figsize=(10.2,10.8))
    ax=fig.add_axes([0.02,0.05,0.96,0.88],projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.OCEAN,facecolor="#dceaf2")
    ax.add_feature(cfeature.LAND,facecolor="#efe9dc")
    ax.coastlines(resolution="110m",color="#9aa7b0",linewidth=0.5)
    # range rings
    for rr in (30,60,90):
        lon,lat=ring(CASH_LAT,CASH_LON,rr)
        ax.plot(lon,lat,transform=pc,color="#9b9b9b",lw=0.6,ls=":",zorder=3)
        lo,la=dest(CASH_LAT,CASH_LON,rr,158)
        ax.text(lo,la,f"{rr}°",transform=pc,fontsize=8,color="#666",zorder=6,
                ha="center",va="center",bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.7))
    # great-circle arcs to the 8 largest (true geodesics)
    for e in sorted(big,key=lambda e:-e["mag"])[:8]:
        ax.plot([CASH_LON,e["lon"]],[CASH_LAT,e["lat"]],transform=ccrs.Geodetic(),
                color="#d9a441",lw=0.9,alpha=0.75,zorder=4)
    # events by method
    for meth,(col,lbl) in style.items():
        sel=[e for e in big if e["method"]==meth]
        if not sel: continue
        ax.scatter([e["lon"] for e in sel],[e["lat"] for e in sel],transform=pc,
                   s=[(e["mag"]-3)**2.4*2.4 for e in sel],c=col,alpha=0.8,
                   edgecolors="white",linewidths=0.3,zorder=5,label=f"{lbl} ({len(sel)})")
    # CASH
    ax.scatter([CASH_LON],[CASH_LAT],marker="*",s=500,c="#cc0000",transform=pc,
               edgecolors="white",linewidths=1.0,zorder=8,label="CASH (Christchurch)")
    for e in sorted(big,key=lambda e:-e["mag"])[:7]:
        ax.text(e["lon"],e["lat"],f" M{e['mag']:.1f}",transform=pc,fontsize=7.5,
                color="#222",zorder=9)
    # magnitude size legend proxies
    for mg in (6,7,8):
        ax.scatter([],[],s=(mg-3)**2.4*2.4,c="#888",edgecolors="white",linewidths=0.3,label=f"M{mg}")
    ax.set_title("CASH earthquake detections, 2016-2025 — global reach of one backyard seismometer",
                 fontsize=14,weight="bold",pad=12)
    ax.legend(loc="lower left",fontsize=8.5,framealpha=0.95,ncol=2,scatterpoints=1)
    fig.text(0.5,0.025,f"{len(big)} quakes M>={MAP_MINMAG} (+{nlocal} local at CASH).  "
             "Rings = 30/60/90° great-circle distance.  Colour = method, size = magnitude.",
             ha="center",fontsize=8.5,color="#555")
    pdf.savefig(fig); return fig

def main():
    with PdfPages(os.path.join(HERE or ".","CASH_detection_map.pdf")) as pdf:
        fig=map_page(pdf)
    fig.savefig(os.path.join(HERE or ".","CASH_detection_map.png"),dpi=130)
    plt.close(fig); print("wrote CASH_detection_map.pdf + .png")

if __name__=="__main__":
    main()

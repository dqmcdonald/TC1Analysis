#!/usr/bin/env python3
"""
CASH detectability vs distance & magnitude, with the Earth's core shadow.

Shows the core shadow is a P-blind BAND (not a max range): surface waves bypass
the core and give near-global reach, amplitude-limited only.

detectability_page(pdf) draws into an existing PdfPages (used by generate_report).
Run directly -> analysis/fig_detectability_distance.png
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=os.path.dirname(__file__)

def _build():
    rows=list(csv.DictReader(open(os.path.join(HERE,"combined_catalog.csv"))))
    for r in rows: r["deg"]=float(r["dist_km"])/111.19; r["m"]=float(r["mag"])
    det=[r for r in rows if r["detected_any"]=="1"]
    # empirical detection floor: 5th-pct magnitude detected per 12-deg bin
    edges=np.arange(0,181,12); cx=[]; floor=[]
    for i in range(len(edges)-1):
        b=[r["m"] for r in det if edges[i]<=r["deg"]<edges[i+1]]
        if len(b)>=3: cx.append((edges[i]+edges[i+1])/2); floor.append(np.percentile(b,5))
    cx=np.array(cx); floor=np.array(floor)
    a=np.median(floor[cx>=15]-1.66*np.log10(cx[cx>=15]))   # fit M = a + 1.66 log10(deg)
    xx=np.linspace(2,180,300); thr=a+1.66*np.log10(xx)

    fig,ax=plt.subplots(figsize=(11.0,7.2))
    ax.axvspan(103,142,color="#f2c9c0",alpha=0.6,zorder=0,label="P-wave shadow (103–142°)")
    ax.axvspan(103,182,facecolor="none",hatch="//",edgecolor="#e0c4c4",lw=0,zorder=0)
    ax.axvline(98,ls=":",c="#888",lw=1); ax.axvline(145,ls=":",c="#888",lw=1)
    ax.axvline(180,ls="-",c="#444",lw=1)
    style={"body+surface":("#7b3294","both methods"),"body":("#2c7fb8","body-wave"),
           "surface":("#1a9850","surface-wave")}
    nd=[r for r in rows if r["detected_any"]!="1"]
    ax.scatter([r["deg"] for r in nd],[r["m"] for r in nd],s=6,c="#dadada",zorder=1,label="not detected")
    for meth,(c,l) in style.items():
        s=[r for r in det if r["method"]==meth]
        ax.scatter([r["deg"] for r in s],[r["m"] for r in s],s=18,c=c,alpha=0.8,zorder=3,label=l)
    ax.plot(xx,thr,"k--",lw=1.6,zorder=4,label=f"detection floor ≈ {a:.1f} + 1.66·log₁₀Δ")
    ax.text(122,3.15,"direct P absent\n(only weak Pdiff)",ha="center",fontsize=8.5,color="#7a3b30")
    ax.text(96,8.85,"direct P limit ~98°",rotation=90,va="top",ha="right",fontsize=8,color="#666")
    ax.text(143,8.85,"PKP (through core) ~145°",rotation=90,va="top",ha="right",fontsize=8,color="#666")
    ax.text(179,3.0,"antipode 180°",rotation=90,va="bottom",ha="right",fontsize=8,color="#444")
    ax.text(52,2.95,"S-wave shadow for ALL distances >103° (liquid outer core blocks S)",
            fontsize=8.5,color="#999")
    ax.set_xlim(0,182); ax.set_ylim(2.8,9.0)
    ax.set_xlabel("epicentral distance from CASH (degrees)"); ax.set_ylabel("magnitude")
    ax.set_title("CASH detectability vs distance & magnitude — surface waves bypass the core shadow",
                 fontsize=13,weight="bold")
    ax.legend(loc="upper left",fontsize=8.5,ncol=2,framealpha=0.95); ax.grid(alpha=0.25)
    fig.text(0.5,0.012,"Core shadow blanks direct P (103–142°) & S (>103°), but surface waves reach "
             f"the antipode — max range is amplitude-limited: Δ_max ≈ 10^((M−{a:.1f})/1.66)°.",
             ha="center",fontsize=8.5,color="#555")
    fig.tight_layout(rect=[0,0.03,1,1])
    return fig,a

def detectability_page(pdf):
    fig,a=_build(); pdf.savefig(fig); return fig

def main():
    fig,a=_build()
    fig.savefig(os.path.join(HERE,"fig_detectability_distance.png"),dpi=130)
    print(f"fit: M_threshold ≈ {a:.2f} + 1.66·log10(deg)")
    print("wrote fig_detectability_distance.png")

if __name__=="__main__":
    main()

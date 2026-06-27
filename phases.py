#!/usr/bin/env python3
"""
Wave-type / phase analysis for selected CASH events.

Single vertical component => we identify phases by TRAVEL TIME (P first, then S,
then surface waves) and by FREQUENCY (P highest, S lower, surface lowest), NOT by
3-component particle-motion polarization (which we don't have).

For each event we render: body-wave-band waveform, long-period waveform (to expose
surface waves), and a spectrogram -- all with predicted P / S / surface markers.
For local events we also auto-pick P and S and turn the S-P time into a distance.

Travel-time model:
  regional (<1500 km): apparent Pn 8.0, Sn 4.45, surface 3.0 km/s
  teleseismic (>1500 km): iasp91-ish surface-focus P & S tables; surface 3.6 km/s
  (deep sources shift body-wave times earlier & suppress surface waves -- flagged)

Output: analysis/CASH_wave_analysis.pdf  (+ per-event PNGs)
"""
import os, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt, detrend, spectrogram, hilbert, lfilter
import generate_report as G   # reuse read_sac / get_trace / bp

HERE=os.path.dirname(__file__)
OUT=os.path.join(HERE,"CASH_wave_analysis.pdf")

# Body-wave arrival times from obspy TauP (iasp91): depth-corrected and accurate
# for teleseismic and deep events (replaces the old hand-built tables). Surface
# waves are not modelled by TauP, so they stay as a Rayleigh group-velocity marker.
from obspy.taup import TauPyModel
_MODEL=TauPyModel(model="iasp91")
VP_LOC,VS_LOC=5.9,3.45        # crustal velocities for the local S-P -> distance demo
VP_REG,VS_REG=8.0,4.45
VSURF=3.0                     # Rayleigh group-velocity marker (waves are dispersive)

def crustal_v(dist): return (VP_LOC,VS_LOC) if dist<150 else (VP_REG,VS_REG)
def _taup_first(dist,depth,phases):
    # explicit P/S branch phases only -- avoids fast converted phases like sP/pS
    try:
        arr=_MODEL.get_travel_times(source_depth_in_km=max(float(depth),0.0),
            distance_in_degree=dist/111.19,phase_list=phases)
    except Exception:
        arr=[]
    return min((a.time for a in arr),default=None)
def tt_p(dist,depth=10.0):
    t=_taup_first(dist,depth,["P","Pn","Pg","Pdiff","p"])
    return t if t is not None else dist/crustal_v(dist)[0]
def tt_s(dist,depth=10.0):
    t=_taup_first(dist,depth,["S","Sn","Sg","Sdiff","s"])
    return t if t is not None else dist/crustal_v(dist)[1]
def tt_pp(dist,depth):        # depth phase pP (prominent for deep sources)
    try:
        arr=_MODEL.get_travel_times(source_depth_in_km=max(float(depth),0.0),
            distance_in_degree=dist/111.19,phase_list=["pP"])
        return min((a.time for a in arr),default=None)
    except Exception: return None
def tt_surf(dist): return dist/VSURF

def envelope(y): return np.abs(hilbert(y))

def recursive_sta_lta(x,nsta,nlta):
    sq=x*x; cs,cl=1.0/nsta,1.0/nlta
    sta=lfilter([cs],[1,-(1-cs)],sq); lta=lfilter([cl],[1,-(1-cl)],sq)
    lta=np.maximum(lta,1e-12); r=sta/lta; r[:int(nlta)]=0; return r

# events: (origin_str, label, kind)  kind: local | regional | tele_shallow | tele_deep
EVENTS=[
    ("2023-09-19T21:14:49","M6.0  45 km N of Geraldine","local"),
    ("2022-01-12T10:49:36","M5.8  20 km E of Stratford (189 km deep)","regional"),
    ("2017-11-19T22:43:29","M7.0  New Caledonia (2500 km, 10 km deep)","tele_shallow"),
    ("2018-08-19T00:19:40","M8.2  Fiji 2018 (2960 km, 600 km deep)","tele_deep"),
]

def get_event(tstr):
    for r in G.load_matched():
        if r["time"]==tstr: return r
    return None

def bands_for(kind):
    if kind=="local":     return (1.5,9.0),(0.5,2.0)
    if kind=="regional":  return (1.0,8.0),(0.3,1.5)
    return (0.8,6.0),(0.04,0.4)   # teleseismic: long-period to expose surface waves

def event_page(pdf, tstr, label, kind):
    r=get_event(tstr)
    if r is None: print("  missing",tstr); return
    o=dt.datetime.strptime(tstr,"%Y-%m-%dT%H:%M:%S")
    dist=float(r["dist_km"]); depth=float(r["depth"])
    P,S,SURF=tt_p(dist,depth),tt_s(dist,depth),tt_surf(dist)
    PP=tt_pp(dist,depth) if depth>100 else None   # pP depth phase (deep sources)
    pre=20; post=max(SURF*1.25, S*1.3, 90)+60
    t,y,fs=G.get_trace(o,pre,post)            # t relative to origin
    if t is None: print("  no trace",tstr); return
    y=detrend(y,type="linear")
    (blo,bhi),(llo,lhi)=bands_for(kind)
    yb=G.bp(y,fs,blo,bhi)
    yl=G.bp(y,fs,llo,lhi)

    fig=plt.figure(figsize=(8.27,11.69))
    fig.suptitle(label,fontsize=16,weight="bold",x=0.5,y=0.975,color="#1f3b57")
    deep=depth>100
    sub=(f"distance {dist:.0f} km    depth {depth:.0f} km     |     TauP   P {P:.0f}s"
         + (f"   pP {PP:.0f}s" if PP else "") + f"    S {S:.0f}s    surface {SURF:.0f}s")
    fig.text(0.5,0.948,sub,ha="center",fontsize=9.5,color="#444")
    if deep:
        fig.text(0.5,0.930,"deep source: body-wave times incl. the pP depth phase are from "
                 "TauP (depth-corrected); surface waves are physically suppressed",
                 ha="center",fontsize=8.5,style="italic",color="#777")

    # P, pP (deep), S from TauP -- accurate for teleseismic & deep events.
    def marks(ax, show_surf=True):
        for tt,c,nm in [(P,"#1a9850","P"),(PP,"#3366cc","pP"),(S,"#d62728","S")]:
            if tt is not None and -pre<=tt<=post:
                ax.axvline(tt,color=c,lw=1.3,ls="--")
                ax.text(tt,ax.get_ylim()[1]*0.86,f" {nm}",color=c,fontsize=10,weight="bold")
        if show_surf and not deep and -pre<=SURF<=post:
            ax.axvline(SURF,color="#7b3294",lw=1.3,ls="--")
            ax.text(SURF,ax.get_ylim()[1]*0.86," surface",color="#7b3294",fontsize=10,weight="bold")

    # panel 1: body-wave band
    ax1=fig.add_axes([0.10,0.66,0.82,0.22])
    ax1.plot(t,yb,lw=0.4,color="#1f3b57"); ax1.set_xlim(-pre,post)
    ax1.set_title(f"Body-wave band {blo}-{bhi} Hz  (P & S arrivals)",fontsize=11)
    ax1.set_ylabel("counts"); ax1.grid(alpha=0.25); marks(ax1,show_surf=False)

    # local: auto-pick P onset and S onset, measure S-P -> distance
    if kind=="local":
        rr=recursive_sta_lta(yb,int(1.0*fs),int(20*fs))
        i0=int(np.argmax(t>=0))
        trig=np.where(rr[i0:]>6)[0]
        tP=t[i0+trig[0]] if len(trig) else P
        # S ONSET: smooth the envelope (~4 s), then walk FORWARD from P to the
        # first strong sustained energy (P is weak on a vertical geophone, so the
        # first arrival above 25% of the peak is the S onset, not the coda max).
        w=max(1,int(4*fs))
        env=np.convolve(envelope(yb),np.ones(w)/w,mode="same")
        gpk=env[t>=tP].max()
        cand=np.where((t>tP+3.0)&(env>0.25*gpk))[0]
        if len(cand):
            tS=t[cand[0]]
            vp,vs=crustal_v(dist); dest=(tS-tP)/(1/vs-1/vp)
            ax1.axvline(tP,color="#1a9850",lw=1.0,alpha=0.7)
            ax1.axvline(tS,color="#d62728",lw=1.0,alpha=0.7)
            gate="" if 0.5*dist<dest<2.0*dist else "  [approx]"
            fig.text(0.10,0.635,f"Auto-picked S-P = {tS-tP:.0f} s  (P={tP:.0f}s, S={tS:.0f}s)"
                     f"   =>   distance {dest:.0f} km   vs catalog {dist:.0f} km{gate}",
                     fontsize=10,color="#063",weight="bold")

    # panel 2: long-period band (surface waves)
    ax2=fig.add_axes([0.10,0.37,0.82,0.20])
    ax2.plot(t,yl,lw=0.5,color="#5a3d8a"); ax2.set_xlim(-pre,post)
    ax2.set_title(f"Long-period band {llo}-{lhi} Hz  (surface / ground waves)",fontsize=11)
    ax2.set_ylabel("counts"); ax2.grid(alpha=0.25); marks(ax2)

    # panel 3: spectrogram
    ax3=fig.add_axes([0.10,0.08,0.82,0.21])
    nper=512 if kind.startswith("tele") else 256
    f_,t_,Sxx=spectrogram(y,fs,nperseg=nper,noverlap=int(nper*0.9))
    t_=t_-pre
    Sxx=10*np.log10(Sxx+1e-9)
    fmax=6
    fm=f_<=fmax
    ax3.pcolormesh(t_,f_[fm],Sxx[fm],shading="auto",cmap="magma")
    ax3.set_xlim(-pre,post); ax3.set_ylim(0,fmax)
    ax3.set_title("Spectrogram (P=high freq, S=mid, surface=low freq)",fontsize=11)
    ax3.set_xlabel("seconds after origin"); ax3.set_ylabel("Hz")
    for tt,c in [(P,"#7CFC00"),(PP,"#66ccff"),(S,"#ff5555")]:
        if tt is not None and -pre<=tt<=post: ax3.axvline(tt,color=c,lw=1.0,ls="--")
    if not deep and -pre<=SURF<=post: ax3.axvline(SURF,color="#cf9fff",lw=1.0,ls="--")
    if kind=="tele_deep":
        fig.text(0.5,0.020,"No surface-wave train: the deep (600 km) source radiates little "
                 "surface-wave energy (cf. shallow New Caledonia).",ha="center",
                 fontsize=8.5,style="italic",color="#555")
    pdf.savefig(fig); plt.close(fig)

def intro_page(pdf):
    fig=plt.figure(figsize=(8.27,11.69)); ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0.88),1,0.12,color="#1f3b57"))
    ax.text(0.5,0.94,"CASH — Seismic Wave-Type Analysis",ha="center",va="center",
            fontsize=22,color="white",weight="bold")
    txt=[
     "Can a single vertical-component backyard seismometer separate seismic phases?",
     "",
     "What we CAN do:",
     "  - Identify P, S and surface waves by their TRAVEL TIMES (P arrives first,",
     "    then S, then the slower surface waves). P/S/pP arrivals below are",
     "    computed with obspy TauP (iasp91), depth-corrected per event.",
     "  - Identify them by FREQUENCY: P is highest-frequency, S lower, surface",
     "    waves lowest -- visible in the spectrograms.",
     "  - For local quakes, measure the S-P time and convert it to distance.",
     "",
     "What we CANNOT do here:",
     "  - True P-vs-S discrimination by particle motion needs THREE components",
     "    (vertical + 2 horizontal). CASH is vertical-only, so we infer phases",
     "    from timing and frequency, not from polarization.",
     "  - Teleseismic surface waves have 15-30 s periods; the 4.5 Hz geophone",
     "    rolls off steeply below its corner, so distant surface waves are",
     "    strongly attenuated. Whether you see them depends on source DEPTH:",
     "    shallow quakes make strong surface waves, deep quakes make almost none.",
     "",
     "The five events below run from a local crustal quake out to a great distant",
     "earthquake, chosen to show each wave type -- including the shallow-vs-deep",
     "surface-wave contrast, and (last) the M7.7 Myanmar quake 10,669 km away whose",
     "surface waves CASH recorded even though the body-wave detector missed it.",
    ]
    y=0.84
    for ln in txt:
        b=ln.endswith(":") and not ln.startswith(" ")
        ax.text(0.09,y,ln,fontsize=11.5,va="top",weight="bold" if b else "normal",
                color="#1f3b57" if b else "black")
        y-=0.030
    pdf.savefig(fig); plt.close(fig)

def main():
    import myanmar_2025
    with PdfPages(OUT) as pdf:
        intro_page(pdf)
        for tstr,label,kind in EVENTS:
            event_page(pdf,tstr,label,kind)
        myanmar_2025.myanmar_page(pdf)   # 5th: extreme teleseism, surface waves only
    print("DONE ->",OUT)

if __name__=="__main__":
    main()

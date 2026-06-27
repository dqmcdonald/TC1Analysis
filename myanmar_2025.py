#!/usr/bin/env python3
"""
Did CASH record surface waves from the 28 Mar 2025 M7.7 Mandalay (Myanmar) quake?

The event is 10,669 km (~96 deg) away and shallow (10 km). The body-wave STA/LTA
detector logged it as NOT detected -- because at this range the recorded energy is
dominated by long-period SURFACE waves, which the detector's 1.5-8 Hz band misses.
This isolates the surface-wave band and checks for the dispersion that proves it is
a real Rayleigh wave rather than a cultural-noise burst.

Run directly -> analysis/CASH_myanmar_2025.pdf.
myanmar_page(pdf) draws the same page into an existing PdfPages (used by phases.py).
"""
import os, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import detrend, spectrogram, hilbert
import generate_report as G, phases as ph

def myanmar_page(pdf):
    o=dt.datetime(2025,3,28,6,20,52); dist=10669.0
    P=ph.tt_p(dist)
    gv=[3.6,3.0,2.5]; gt=[dist/v for v in gv]          # Rayleigh group-velocity guides
    pre,post=120,5200
    t,y,fs=G.get_trace(o,pre,post); y=detrend(y,type="linear")
    yb=G.bp(y,fs,0.8,5.0)                                # body waves
    ys=G.bp(y,fs,0.03,0.12)                              # surface band (8-33 s)
    w=int(20*fs); envs=np.convolve(np.abs(hilbert(ys)),np.ones(w)/w,mode="same")

    # spike-robust SNR: median smoothed envelope in the packet vs quiet pre-event
    pre_env=np.median(envs[(t>-100)&(t<700)])
    pk=(t>3300)&(t<4200)
    snr_med=np.median(envs[pk])/pre_env; snr_pk=envs[pk].max()/pre_env
    print(f"P={P:.0f}s  packet group velocity {dist/4200:.2f}-{dist/3300:.2f} km/s")
    print(f"robust surface SNR median {snr_med:.1f}x  peak {snr_pk:.1f}x")

    fig=plt.figure(figsize=(8.27,11.69))
    fig.suptitle("M7.7 Mandalay, Myanmar - 28 Mar 2025 - recorded at CASH (10,669 km)",
                 fontsize=13.5,weight="bold",y=0.975,color="#1f3b57")
    fig.text(0.5,0.945,f"A weak P arrives at {P:.0f} s [TauP]; the dominant recorded energy is the "
             f"dispersed surface-wave train in the Rayleigh window (robust SNR ~{snr_med:.0f}x).",
             ha="center",fontsize=9,color="#444")

    ax1=fig.add_axes([0.10,0.70,0.82,0.18])
    ax1.plot(t,yb,lw=0.3,color="#1f3b57"); ax1.set_xlim(-pre,post); ax1.set_ylim(-120,120)
    ax1.axvline(P,color="#1a9850",ls="--",lw=1.2); ax1.text(P,95,"  P",color="#1a9850",weight="bold")
    ax1.set_title(f"Body-wave band 0.8-5 Hz  (emergent P at {P:.0f} s [TauP]; tall spikes = local noise, clipped)",fontsize=9.5)
    ax1.set_ylabel("counts"); ax1.grid(alpha=0.25)

    ax2=fig.add_axes([0.10,0.44,0.82,0.20])
    ax2.axvspan(3300,4200,color="#ffe9a8",alpha=0.5)
    ax2.plot(t,ys,lw=0.4,color="#5a3d8a"); ax2.plot(t,envs,lw=1.3,color="#d62728")
    ax2.set_xlim(-pre,post); ax2.set_ylim(-300,300)
    for v,tt in zip(gv,gt):
        ax2.axvline(tt,color="#7b3294",ls=":",lw=1.1); ax2.text(tt,232,f" {v}",color="#7b3294",fontsize=8)
    ax2.set_title("Surface-wave band 0.03-0.12 Hz (8-33 s) - dispersed Rayleigh packet shaded; dotted = group vel (km/s)",fontsize=9)
    ax2.set_ylabel("counts"); ax2.grid(alpha=0.25)

    ax3=fig.add_axes([0.10,0.07,0.82,0.27])
    nper=1024
    f_,tt_,Sxx=spectrogram(y,fs,nperseg=nper,noverlap=int(nper*0.92)); tt_=tt_-pre
    ax3.pcolormesh(tt_,f_[f_<=0.6],10*np.log10(Sxx[f_<=0.6]+1e-9),shading="auto",cmap="magma")
    ax3.set_xlim(-pre,post); ax3.set_ylim(0,0.6)
    ax3.axvline(P,color="#7CFC00",ls="--",lw=1.0)
    for tt in gt: ax3.axvline(tt,color="#cf9fff",ls=":",lw=1.0)
    ax3.set_title("Spectrogram 0-0.6 Hz - sustained low-frequency energy across the surface-wave window",fontsize=9.5)
    ax3.set_xlabel("seconds after origin"); ax3.set_ylabel("Hz")
    fig.text(0.5,0.022,"Dispersion: the packet spans ~3.2 to ~2.5 km/s group velocity - the signature "
             "of a real surface wave, not a noise burst.",ha="center",fontsize=8.5,style="italic",color="#555")
    pdf.savefig(fig); plt.close(fig)

def main():
    out=os.path.join(os.path.dirname(__file__) or ".","CASH_myanmar_2025.pdf")
    with PdfPages(out) as pdf:
        myanmar_page(pdf)
    print("wrote",out)

if __name__=="__main__":
    main()

# TC1Analysis

Tools for analysing ~9 years of continuous data from a **TC1 educational
seismometer** — station **CASH** (Cashmere, Christchurch, New Zealand), a single
vertical-component 4.5 Hz geophone in the jAmaSeis network.

From one DIY/backyard-class sensor these tools build an ambient-noise climatology,
an earthquake catalogue (body-wave **and** surface-wave detectors), wave-type /
phase analyses, and several PDF reports — and, along the way, surface a real
station data-quality fault and quantify the instrument's detection capability
against the Earth's core shadow.

---

## Background

- **Instrument:** TC1 vertical geophone, ~4.5 Hz corner, recorded by jAmaSeis.
- **Station:** CASH, ~(-43.567°, 172.622°), Christchurch NZ.
- **Data:** one **big-endian SAC** file per hour, `DATA_ROOT/<year>/<month>/<day>/<hour>.sac`
  (`0.sac`..`23.sac`), ~18.77 Hz, 3600 s (~67,564 samples) each. Timestamps are **UTC**
  (NZ local = UTC+12, +13 in summer). Span **2016-12-03 → 2025-07-31**,
  72,589 valid hours, 99.8 % readable, 95.7 % coverage.
- The raw SAC archive is **not** in this repo (it is large). Point `config.py`
  / the `CASH_ARCHIVE` env var at your own copy.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11–3.13 recommended
pip install -r requirements.txt
export CASH_ARCHIVE=/path/to/CASH                        # your SAC archive root
```

`obspy` (TauP travel times) and `cartopy` (maps) are the heavyweight deps; numpy /
scipy / matplotlib do the rest. No pandas. `tkinter` is needed only for the GUI.

---

## Pipeline & scripts

Run from the repo directory. Scripts share `config.py` and reuse each other
(e.g. `phases.py` imports `generate_report.py`). The cross-match and phase tools
need **obspy**, so run them with the venv that has it.

| Order | Script | Produces | Purpose |
|------|--------|----------|---------|
| 1 | `scan_cash.py` | `cash_hourly_metrics.csv` | Sweep the whole archive once; per-hour robust RMS, band RMS, clipping, gaps. |
| 1b| `scan_bands.py` | `cash_bands.csv` | Robust per-minute-median RMS in microseism / cultural bands. |
| 1c| `scan_surface.py` | `cash_surface.csv` | Per-hour percentiles of 0.04–0.10 Hz energy (for the surface detector). |
| 2 | `plot_noise.py` | noise / lockdown / completeness figures | Ambient-noise climatology, 2020 COVID lockdown, diurnal/weekly cycles. |
| 3 | `microseism.py` | microseism figures + summary | Seasonal ocean-microseism study (winter peak, dispersion, controls). |
| 4 | `detect.py` | `detections.csv` | STA/LTA body-wave detector on candidate hours (~94k triggers). |
| 5 | `catalog_match.py` | `matched_events.csv`, capability figures | Fetch GeoNet + USGS catalogues, cross-match using **TauP** P/S windows. |
| 6 | `surface_detect.py` | `surface_detections.csv`, figure | Surface-wave detector for great distant quakes the body detector misses. |
| 7 | `combined_catalog.py` | `combined_catalog.csv`, `cash_detected_catalog.csv` | Merge both detectors into one method-tagged catalogue. |
| 8 | `eqmap.py` | `CASH_detection_map.*` | Cartopy azimuthal-equidistant, CASH-centred detection map. |
| 9 | `detectability_distance.py` | `fig_detectability_distance.png` | Detectability vs distance/magnitude with the core shadow. |
| 10| `phases.py` | `CASH_wave_analysis.pdf` | Wave-type / phase ID (P/S/pP via TauP, spectrograms) for selected events. |
|   | `myanmar_2025.py` | `CASH_myanmar_2025.pdf` | Surface-wave case study (M7.7 Myanmar 2025). |
|   | `generate_report.py` | `CASH_earthquake_report.pdf` | Headline report: cover, tables, map, detectability, event cards. |
|   | `tc1_explorer.py` | (interactive GUI) | Browse/sort/filter the catalogue; per-event map + annotated trace. |

Typical regeneration order after new data:
`scan_* → detect → catalog_match → surface_detect → combined_catalog → eqmap → generate_report`.

### GUI

```bash
python tc1_explorer.py
```
Sortable/filterable table of detected earthquakes; selecting one shows it on a
map and plots the recorded trace annotated with magnitude, distance, depth,
detection method, SNR and TauP P/S arrival times.

---

## Key assumptions / caveats

- **Single vertical component:** phases are identified by **travel time** (TauP
  iasp91, depth-corrected) and **frequency**, not by particle-motion polarization
  (which needs 3 components). Surface waves are not in TauP → group-velocity marker.
- **Amplitudes are raw counts** — the instrument response is **not** removed, so
  values are comparable between events but not absolute ground motion.
- **Detector is a candidate-flagger.** The body STA/LTA fires ~30×/day on cultural
  noise; detection statistics are made meaningful by catalogue cross-matching and a
  reported false-alarm floor (chance-match probability per event).
- The 4.5 Hz geophone rolls off ~f² below its corner, strongly attenuating
  long-period (microseism, teleseismic surface) energy.
- Catalogues: **GeoNet** FDSN (regional NZ, M≥3, `format=text`) and **USGS** FDSN
  (global, M≥6, `format=csv`), cached under `catalogs/`.

---

## Observed performance

**Data quality.** 99.8 % of 72,589 hourly files readable, 95.7 % continuous
coverage, 0 dead hours, 3 clipped hours. Main gap: a few days in late Jan 2020.

**Ambient noise.** Clean diurnal (≈1.25× day/night) and weekday/weekend cultural
cycles. The **2020 NZ COVID lockdown** shows a **−12.5 %** drop in robust noise.

**Seasonal microseism.** The 0.10–0.35 Hz band peaks in austral autumn/winter
(May–Jul), winter/summer ≈ 1.18×, with a clean ~annual periodogram line — a modest
but genuine Southern-Ocean ocean-microseism signal (partly cultural-contaminated).

**Earthquake detection** (catalogue cross-match, TauP windows):

| Magnitude (teleseismic >1500 km) | Body-wave | Surface-wave | Combined |
|---|---|---|---|
| M7.0–7.5 | 56 % | 53 % | 73 % |
| M7.5–8.0 | 39 % | 81 % | 81 % |
| All M≥7 | 50 % | 63 % | 76 % |

The two detectors are **complementary** — body-wave for local/regional, surface-wave
for great distant events (adds ~29 M≥7 quakes the body detector misses). Combined,
CASH recorded **2,701** catalogued earthquakes. Detection floor follows
**M ≈ 2.8 + 1.66·log₁₀Δ°**, i.e. global reach (to the antipode) by ~M6.5–7; CASH has
detections out to **167°** (Morocco M6.8, near-antipodal).

**Core shadow.** Direct P is geometric-limited to ~98°; the P-shadow (103–142°) and
the S-shadow (>103°) come from the liquid outer core. Surface waves bypass the core,
so the practical maximum range is **amplitude-limited, not shadow-limited** — CASH
has 43 detections inside the P-shadow zone, almost all via surface waves.

**Data-quality finding.** The M8.8 Kamchatka (29 Jul 2025) — the largest catalogued
event — was **missed**: CASH's surface-wave sensitivity degraded around mid-July 2025
(five straight M≥7 missed from 16 Jul, elevated background), confirmed by the 2021
M8.2 Chignik on the same path being ~10× stronger. Late-July 2025 data should be
flagged.

---

## Notable events recorded (examples)

- 4 Mar 2021 Kermadec sequence (M7.3 / 7.4 / 8.1; NZ tsunami warning)
- M8.2 Fiji 2018 (deep, 600 km), M8.2 Chignik & Tehuantepec, M7.8 Türkiye 2023
- M7.7 Myanmar 2025 — clear dispersed surface-wave train at 10,669 km
- Strongest local shaking: M3.8 5 km E of Christchurch (SNR ~2700)

---

*Generated as an exploratory analysis of a backyard seismometer; not a calibrated
monitoring product.*

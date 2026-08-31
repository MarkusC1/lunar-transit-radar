# 🌕 Lunar Transit Radar PRO

[![Live Demo](https://img.shields.io/badge/Live%20Radar-lunar--transit--radar.onrender.com-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://lunar-transit-radar.onrender.com/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Framework-Flask-black?logo=flask)
![NASA JPL](https://img.shields.io/badge/Astrometry-NASA%20JPL%20DE421-orange)
![WGS84](https://img.shields.io/badge/Geodesy-WGS84%204D-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**Lunar Transit Radar PRO** is a real-time 4D aeronautical and astrophotography radar that predicts commercial aircraft transits across the Moon. Combining live **ADS-B transponder telemetry** with high-precision **NASA JPL topocentric lunar ephemeris**, the engine computes sub-second Time of Closest Approach (TCA), lunar chord transit duration, and exact 3D vertical angular offsets.

---

## ✨ Features

- **🔭 Topocentric Lunar Astrometry (NASA JPL DE421):** Real-time lunar position (Azimuth / Elevation) and apparent angular diameter computed via `Skyfield` with observer elevation and parallax correction.
- **✈️ 4D Geodetic Flight Propagation:** Aircraft vectors are integrated along the **WGS84 ellipsoid**, compensating for Earth's curvature drop (~180 m at 50 km) in the observer's local **ENU (East-North-Up)** tangent plane.
- **📐 Exact 3D Vertical Offset (ΔAlt):** Calculates whether an aircraft passes **Above (+)** or **Below (-)** the Moon in degrees, equivalent lunar diameters, and physical meters at closest approach.
- **⏱️ Live TCA Countdowns for Every Flight:** Continuous sub-second countdown (`T-02:45.3`) towards the optical lunar line of sight for all aircraft in the sector.
- **📉 Real-Time Climb/Descent Detection:** Integrates ADS-B vertical rate (`geom_rate` / `baro_rate`) or deduces empirical vertical speed (Δh / Δt) from historical states in memory.
- **⛰️ Automatic Topographic DEM Lookup:** Queries satellite elevation (NASA SRTM / Copernicus via Open-Meteo) with customizable building/rooftop height.
- **⚡ 60 FPS Monotonic Dead-Reckoning:** Smooth forward glide animation on Leaflet without stuttering, reverse jumps, or flickering.
- **🌑 Automatic Bypass When Moon is Down:** If the Moon is below the horizon, transit computation loops are automatically bypassed (0% CPU) and replaced by a live moonrise countdown.

---

## 🛠️ Tech Stack

- **Backend:** Python 3, Flask, Skyfield, Requests, Gunicorn
- **Frontend:** HTML5, Leaflet.js, Tailwind CSS, CARTO Dark Matter HD Basemaps
- **Ephemeris Data:** NASA JPL `de421.bsp`
- **Telemetry Feeds:** Open decentralized ADS-B networks (`airplanes.live`, `adsb.lol`)

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
git clone https://github.com/MarkusC1/lunar-transit-radar.git
cd lunar-transit-radar

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the application
python app.py

### 4. Open in browser
Navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).

---

## ☁️ Cloud Deployment (Render.com)

This application is production-ready for deployment on **Render.com** (Free Web Service tier):

1. **Connect GitHub Repo:** Link your repository on [Render.com](https://render.com).
2. **Environment:** `Python 3`
3. **Build Command:**
   `pip install -r requirements.txt`
4. **Start Command:**
   `gunicorn -b 0.0.0.0:$PORT app:app --workers 1 --threads 8 --timeout 120`
5. **Instance Type:** `Free`

---

## 📸 Astrophotography & Camera Guide

When targeting a lunar aircraft transit:

| Parameter | Recommended Setting |
| :--- | :--- |
| **Shutter Speed** | `1/500s` to `1/2000s` (freezes aircraft motion) |
| **Aperture** | `f/6` - `f/11` (sharpest optical sweet spot for telephotos) |
| **ISO** | `100` - `400` (low noise, the lowest ISO possible) |
| **Drive Mode** | Continuous High-Speed Burst  |

---

## 📄 Open Data Acknowledgements

- **NASA JPL:** Planetary and Lunar Ephemeris DE421.
- **Skyfield:** Astronomy library by Brandon Rhodes.
- **Airplanes.live & ADS-B Exchange / adsb.lol:** Community ADS-B feeder data.
- **CARTO & OpenStreetMap:** High-resolution dark cartography basemaps.
- **Open-Meteo:** Digital Elevation Model API (NASA SRTM / Copernicus).

---

## ⚖️ License

Distributed under the **MIT License**. Free for personal, academic, and non-commercial astrophotography use.

import math
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, render_template_string
from skyfield import almanac
from skyfield.api import load, wgs84

app = Flask(__name__)

# =========================================================================
# 1. NASA JPL Planetary Ephemeris
# =========================================================================
ts = load.timescale()
eph = load('de421.bsp')
sun = eph['sun']
moon = eph['moon']
earth = eph['earth']

# Constantes físicas y geodésicas WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
R_EARTH = 6371000.0

SUN_RADIUS_KM = 696340.0
MOON_RADIUS_KM = 1737.4

WINGSPANS = {
    'A318': 34.1, 'A319': 35.8, 'A320': 35.8, 'A321': 35.8,
    'A332': 60.3, 'A333': 60.3, 'A339': 64.0, 'A359': 64.7, 'A35K': 64.7,
    'A388': 79.8, 'B737': 35.8, 'B738': 35.8, 'B739': 35.8, 'B38M': 35.9,
    'B744': 64.4, 'B748': 68.4, 'B752': 38.0, 'B763': 47.6, 'B772': 60.9,
    'B77W': 64.8, 'B788': 60.1, 'B789': 60.1, 'B78X': 60.1, 'E190': 28.7,
    'E195': 28.7, 'CRJ9': 24.9, 'AT76': 27.0
}

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat) ** 2))
    x = (n + h_m) * math.cos(lat) * math.cos(lon)
    y = (n + h_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * math.sin(lat)
    return x, y, z

def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0_m):
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    sin_l, cos_l = math.sin(lat0), math.cos(lat0)
    sin_o, cos_o = math.sin(lon0), math.cos(lon0)
    
    e = -sin_o * dx + cos_o * dy
    n = -sin_l * cos_o * dx - sin_l * sin_o * dy + cos_l * dz
    u = cos_l * cos_o * dx + cos_l * sin_o * dy + sin_l * dz
    return e, n, u

def enu_to_az_alt(e, n, u):
    ground = math.hypot(e, n)
    az = (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
    alt = math.degrees(math.atan2(u, ground))
    slant = math.sqrt(e**2 + n**2 + u**2)
    return az, alt, slant

def angular_separation(az1, alt1, az2, alt2):
    r1, r2 = math.radians(alt1), math.radians(alt2)
    cos_d = math.sin(r1)*math.sin(r2) + math.cos(r1)*math.cos(r2)*math.cos(math.radians(az1 - az2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))

def propagate_geodetic_position(lat_deg, lon_deg, ground_speed_ms, track_deg, dt_seconds):
    d = ground_speed_ms * dt_seconds
    d_r = d / R_EARTH
    track_r = math.radians(track_deg)
    lat_r = math.radians(lat_deg)
    lon_r = math.radians(lon_deg)

    lat_future_r = math.asin(
        math.sin(lat_r) * math.cos(d_r) + 
        math.cos(lat_r) * math.sin(d_r) * math.cos(track_r)
    )
    lon_future_r = lon_r + math.atan2(
        math.sin(track_r) * math.sin(d_r) * math.cos(lat_r),
        math.cos(d_r) - math.sin(lat_r) * math.sin(lat_future_r)
    )
    return math.degrees(lat_future_r), math.degrees(lon_future_r)

# =========================================================================
# GESTIÓN DE MEMORIA CACHÉ SMART (COMPATIBLE CON GUNICORN)
# =========================================================================
CACHE = {
    'lat': 0.0,
    'lon': 0.0,
    'timestamp': 0.0,
    'aircraft': [],
    'source': 'airplanes.live'
}
HTTP_SESSION = requests.Session()

def get_live_aircraft(cur_lat, cur_lon):
    now = time.time()
    if now - CACHE['timestamp'] < 2.5 and abs(cur_lat - CACHE['lat']) < 0.05 and abs(cur_lon - CACHE['lon']) < 0.05:
        return CACHE['aircraft'], CACHE['source'], max(0.0, now - CACHE['timestamp'])

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AstronomicalRadar/18.0'}
    
    # 1. airplanes.live
    try:
        url = f"https://api.airplanes.live/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            if ac is not None and len(ac) > 0:
                CACHE['lat'] = cur_lat
                CACHE['lon'] = cur_lon
                CACHE['timestamp'] = now
                CACHE['aircraft'] = ac
                CACHE['source'] = 'airplanes.live'
                return ac, 'airplanes.live', 0.0
    except Exception:
        pass

    # 2. adsb.lol de respaldo
    try:
        url = f"https://api.adsb.lol/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            if ac is not None and len(ac) > 0:
                CACHE['lat'] = cur_lat
                CACHE['lon'] = cur_lon
                CACHE['timestamp'] = now
                CACHE['aircraft'] = ac
                CACHE['source'] = 'adsb.lol'
                return ac, 'adsb.lol', 0.0
    except Exception:
        pass

    return CACHE['aircraft'], CACHE['source'], max(0.0, now - CACHE['timestamp'])

# =========================================================================
# ENDPOINT PRINCIPAL (SOPORTA SOL, LUNA Y MODO AUTO)
# =========================================================================
@app.route('/api/data')
def get_data():
    try:
        lat = float(request.args.get('lat', 41.6079))
        lon = float(request.args.get('lon', 2.2876))
        alt = float(request.args.get('alt', 145.0))
        target_req = request.args.get('target', 'auto').lower()
        now_epoch = time.time()

        raw_ac, source_feed, cache_age_sec = get_live_aircraft(lat, lon)

        t_now = ts.now()
        topos_loc = wgs84.latlon(lat, lon, elevation_m=alt)
        obs_loc = earth + topos_loc
        
        # 1. Cálculo Astrométrico de Luna y Sol
        app_moon = obs_loc.at(t_now).observe(moon).apparent()
        m_alt0, m_az0, m_dist0 = app_moon.altaz()
        moon_az0 = float(m_az0.degrees)
        moon_alt0 = float(m_alt0.degrees)
        moon_radius_deg = float(math.degrees(math.asin(MOON_RADIUS_KM / m_dist0.km)))

        app_sun = obs_loc.at(t_now).observe(sun).apparent()
        s_alt0, s_az0, s_dist0 = app_sun.altaz()
        sun_az0 = float(s_az0.degrees)
        sun_alt0 = float(s_alt0.degrees)
        sun_radius_deg = float(math.degrees(math.asin(SUN_RADIUS_KM / s_dist0.km)))

        sun_is_visible = bool(sun_alt0 > 0)
        moon_is_visible = bool(moon_alt0 > 0)

        # 2. Selección del cuerpo activo (Sol o Luna)
        if target_req == 'sun':
            active_target = 'sun'
        elif target_req == 'moon':
            active_target = 'moon'
        else: # 'auto'
            if sun_is_visible:
                active_target = 'sun'
            elif moon_is_visible:
                active_target = 'moon'
            else:
                # Si ambos están bajo el horizonte, tomar el que esté más cerca de salir
                active_target = 'sun' if sun_alt0 > moon_alt0 else 'moon'

        if active_target == 'sun':
            body_obj = sun
            body_name = "Sun"
            body_symbol = "☀️"
            body_az0 = sun_az0
            body_alt0 = sun_alt0
            body_radius_deg = sun_radius_deg
            body_is_visible = sun_is_visible
        else:
            body_obj = moon
            body_name = "Moon"
            body_symbol = "🌕"
            body_az0 = moon_az0
            body_alt0 = moon_alt0
            body_radius_deg = moon_radius_deg
            body_is_visible = moon_is_visible

        # 3. Cálculo de salida / puesta del astro activo
        next_event_str = "--:--"
        next_event_seconds = 0
        event_type = "SET" if body_is_visible else "RISE"

        try:
            t_search_end = ts.from_datetime(datetime.now(timezone.utc) + timedelta(hours=36))
            f_rise_set = almanac.risings_and_settings(eph, body_obj, topos_loc)
            times_rs, events_rs = almanac.find_discrete(t_now, t_search_end, f_rise_set)

            now_utc = datetime.now(timezone.utc)
            for t_e, ev in zip(times_rs, events_rs):
                dt_e = t_e.utc_datetime()
                if body_is_visible and ev == 0:
                    event_type = "SET"
                    next_event_str = dt_e.strftime('%H:%M UTC')
                    next_event_seconds = max(0, int((dt_e - now_utc).total_seconds()))
                    break
                elif not body_is_visible and ev == 1:
                    event_type = "RISE"
                    next_event_str = dt_e.strftime('%H:%M UTC')
                    next_event_seconds = max(0, int((dt_e - now_utc).total_seconds()))
                    break
        except Exception:
            pass

        # Derivadas de posición del astro
        d_az_dt, d_alt_dt = 0.0, 0.0
        if body_is_visible:
            dt_future = datetime.now(timezone.utc) + timedelta(seconds=300)
            t_300 = ts.from_datetime(dt_future)
            app_300 = obs_loc.at(t_300).observe(body_obj).apparent()
            b_alt300, b_az300, _ = app_300.altaz()
            d_az_dt = (float(b_az300.degrees) - body_az0) / 300.0
            d_alt_dt = (float(b_alt300.degrees) - body_alt0) / 300.0

        aircraft_results = []
        body_diameter_deg = 2.0 * body_radius_deg

        for ac in raw_ac:
            raw_lat = ac.get('lat')
            raw_lon = ac.get('lon')
            alt_val = ac.get('alt_geom') or ac.get('alt_baro')
            track = ac.get('track')
            gs = ac.get('gs', 0)
            vr_raw = ac.get('geom_rate', ac.get('baro_rate', 0))
            model_icao = str(ac.get('t', 'A320')).strip().upper()

            if None in (raw_lat, raw_lon, alt_val, track) or alt_val == 'ground':
                continue

            try:
                alt_m = float(alt_val) * 0.3048
                speed_ms = float(gs) * 0.514444
                vr_ms = float(vr_raw) * 0.00508 if vr_raw else 0.0
                vr_fpm = int(float(vr_raw)) if vr_raw else 0
                track_val = float(track)
                ac_lat = float(raw_lat)
                ac_lon = float(raw_lon)
            except (ValueError, TypeError):
                continue

            e0, n0, u0 = ecef_to_enu(*geodetic_to_ecef(ac_lat, ac_lon, alt_m), lat, lon, alt)
            cur_az, cur_alt, cur_range = enu_to_az_alt(e0, n0, u0)

            # Si el astro está bajo el horizonte
            if not body_is_visible:
                aircraft_results.append({
                    'callsign': str(ac.get('flight') or ac.get('hex', 'UNKNOWN')).strip(),
                    'model': model_icao,
                    'reg': str(ac.get('r', '')).strip().upper(),
                    'lat': ac_lat,
                    'lon': ac_lon,
                    'alt_ft': int(float(alt_val)),
                    'track': float(track_val),
                    'speed_kt': int(float(gs)) if gs else 0,
                    'speed_ms': float(speed_ms),
                    'vr_fpm': vr_fpm,
                    'azimuth': float(round(cur_az, 1)),
                    'elevation': float(round(cur_alt, 1)),
                    'distance_km': float(round(cur_range / 1000.0, 1)),
                    'current_sep': 99.0,
                    'min_sep': 99.0,
                    'vertical_offset_deg': 0.0,
                    'vertical_offset_m': 0,
                    'vertical_body_diams': 0.0,
                    'vertical_dir_text': '',
                    'tca_seconds': 0.0,
                    'tca_epoch': float(now_epoch),
                    'transit_duration_s': 0.0,
                    'position_descriptor': f"{body_name} Below Horizon",
                    'is_transit': False,
                    'is_close': False
                })
                continue

            # Si el astro es visible en el cielo
            cur_sep = angular_separation(cur_az, cur_alt, body_az0, body_alt0)

            def evaluate_aircraft_state_at_t(t_val):
                p_lat, p_lon = propagate_geodetic_position(ac_lat, ac_lon, speed_ms, track_val, t_val)
                p_alt_m = alt_m + (vr_ms * t_val)
                et, nt, ut = ecef_to_enu(*geodetic_to_ecef(p_lat, p_lon, p_alt_m), lat, lon, alt)
                p_az, p_alt, p_range = enu_to_az_alt(et, nt, ut)
                if p_alt <= 0: return 999.0, p_az, p_alt, p_range
                b_az_t = body_az0 + d_az_dt * t_val
                b_alt_t = body_alt0 + d_alt_dt * t_val
                sep = angular_separation(p_az, p_alt, b_az_t, b_alt_t)
                return sep, p_az, p_alt, p_range

            best_t = 0.0
            min_sep = cur_sep
            best_p_alt = cur_alt
            best_p_range = cur_range
            best_b_alt = body_alt0

            for step in range(1, 101):
                t_cand = float(step * 3)
                sep_val, p_az, p_alt, p_range = evaluate_aircraft_state_at_t(t_cand)
                if sep_val < min_sep:
                    min_sep = sep_val
                    best_t = t_cand
                    best_p_alt = p_alt
                    best_p_range = p_range
                    best_b_alt = body_alt0 + d_alt_dt * t_cand

            if best_t > 0:
                t_start = max(0.0, best_t - 3.0)
                t_end = min(300.0, best_t + 3.0)
                for f in range(int((t_end - t_start) / 0.1) + 1):
                    t_cand = t_start + (f * 0.1)
                    sep_val, p_az, p_alt, p_range = evaluate_aircraft_state_at_t(t_cand)
                    if sep_val < min_sep:
                        min_sep = sep_val
                        best_t = t_cand
                        best_p_alt = p_alt
                        best_p_range = p_range
                        best_b_alt = body_alt0 + d_alt_dt * t_cand

            vertical_offset_deg = float(round(best_p_alt - best_b_alt, 3))
            vertical_body_diams = float(round(abs(vertical_offset_deg) / max(0.01, body_diameter_deg), 1))
            vertical_offset_m = int(best_p_range * math.tan(math.radians(vertical_offset_deg)))

            is_transit = True if min_sep <= body_radius_deg else False
            is_close = True if (min_sep > body_radius_deg and min_sep <= (body_radius_deg * 3.5)) else False

            transit_duration_s = 0.0
            if is_transit and best_t > 0:
                chord_deg = 2.0 * math.sqrt(max(0.0, (body_radius_deg ** 2) - (min_sep ** 2)))
                dt_delta = 0.5
                sep_prev, _, _, _ = evaluate_aircraft_state_at_t(max(0.0, best_t - dt_delta))
                sep_post, _, _, _ = evaluate_aircraft_state_at_t(best_t + dt_delta)
                ang_speed_deg_s = max(0.1, math.hypot((sep_post - sep_prev) / (2 * dt_delta), (speed_ms / best_p_range) * (180 / math.pi)))
                transit_duration_s = round(chord_deg / ang_speed_deg_s, 2)

            if is_transit:
                if abs(vertical_offset_deg) <= (body_radius_deg * 0.35):
                    position_descriptor = f"{body_name} Center"
                elif vertical_offset_deg > 0:
                    position_descriptor = "Northern Limb (Above)"
                else:
                    position_descriptor = "Southern Limb (Below)"
            else:
                dir_txt = "Above" if vertical_offset_deg > 0 else "Below"
                position_descriptor = f"{abs(vertical_offset_deg):.2f}° {dir_txt} ({vertical_body_diams} {body_symbol})"

            dir_txt_clean = "Above" if vertical_offset_deg > 0 else "Below"

            aircraft_results.append({
                'callsign': str(ac.get('flight') or ac.get('hex', 'UNKNOWN')).strip(),
                'model': model_icao,
                'reg': str(ac.get('r', '')).strip().upper(),
                'lat': ac_lat,
                'lon': ac_lon,
                'alt_ft': int(float(alt_val)),
                'track': float(track_val),
                'speed_kt': int(float(gs)) if gs else 0,
                'speed_ms': float(speed_ms),
                'vr_fpm': vr_fpm,
                'azimuth': float(round(cur_az, 1)),
                'elevation': float(round(cur_alt, 1)),
                'distance_km': float(round(cur_range / 1000.0, 1)),
                'current_sep': float(round(cur_sep, 2)),
                'min_sep': float(round(min_sep, 3)),
                'vertical_offset_deg': vertical_offset_deg,
                'vertical_offset_m': vertical_offset_m,
                'vertical_body_diams': vertical_body_diams,
                'vertical_dir_text': dir_txt_clean,
                'tca_seconds': float(round(best_t, 1)),
                'tca_epoch': float(now_epoch + best_t) if best_t > 0 else float(now_epoch),
                'transit_duration_s': transit_duration_s,
                'position_descriptor': position_descriptor,
                'is_transit': is_transit,
                'is_close': is_close
            })

        return jsonify({
            'source_feed': source_feed,
            'server_time': now_epoch,
            'observer_altitude_used_m': alt,
            'target_mode': target_req,
            'body': {
                'type': active_target,
                'name': body_name,
                'symbol': body_symbol,
                'azimuth': float(round(body_az0, 2)),
                'elevation': float(round(body_alt0, 2)),
                'radius_deg': float(round(body_radius_deg, 3)),
                'visible': body_is_visible,
                'event_type': event_type,
                'next_event_str': next_event_str,
                'next_event_seconds': next_event_seconds
            },
            'sun_summary': {'az': round(sun_az0, 1), 'alt': round(sun_alt0, 1), 'visible': sun_is_visible},
            'moon_summary': {'az': round(moon_az0, 1), 'alt': round(moon_alt0, 1), 'visible': moon_is_visible},
            'aircraft': aircraft_results
        })

    except Exception as e:
        print(f"[ERROR ENGINE] {e}")
        return jsonify({
            'body': {'type': 'sun', 'name': 'Sun', 'symbol': '☀️', 'azimuth': 0, 'elevation': -90, 'radius_deg': 0.26, 'visible': False},
            'aircraft': []
        })

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# =========================================================================
# WEB UI (COMPATIBLE CON MODO SOLAR Y LUNAR)
# =========================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASTRONOMICAL TRANSIT RADAR PRO (Solar & Lunar)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0f19; color: #f8fafc; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        #map { height: calc(100vh - 115px); width: 100%; border-radius: 12px; }
        .leaflet-container { background: #0b0f19 !important; }
        
        .obs-target { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; }
        .obs-ring { position: absolute; width: 32px; height: 32px; border-radius: 50%; background: rgba(6, 182, 212, 0.25); border: 2px solid #06b6d4; animation: pulse-ring 2s infinite ease-out; }
        .obs-dot { width: 10px; height: 10px; border-radius: 50%; background: #22d3ee; border: 2px solid #ffffff; box-shadow: 0 0 14px #06b6d4; z-index: 10; }
        @keyframes pulse-ring { 0% { transform: scale(0.5); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }
        
        .astro-marker { display: flex; align-items: center; justify-content: center; font-size: 24px; filter: drop-shadow(0 0 12px #f59e0b); }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0b1120; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    </style>
</head>
<body class="p-2 flex flex-col h-screen overflow-hidden">
    <header class="bg-slate-900/90 backdrop-blur border border-slate-800 px-3 py-2 rounded-xl mb-1.5 flex justify-between items-center gap-2 shadow-xl">
        <div class="flex items-center gap-2.5">
            <span id="header-symbol" class="text-2xl animate-pulse">☀️</span>
            <div>
                <div class="flex items-center gap-1.5">
                    <h1 class="text-xs font-black text-amber-400 tracking-wider">TRANSIT RADAR PRO</h1>
                    <span id="feed-badge" class="text-[8px] px-1 py-0.2 bg-emerald-950 text-emerald-300 border border-emerald-700 rounded font-bold">ONLINE</span>
                </div>
                <span id="astro-horizon-status" class="text-[10px] text-slate-400 font-bold">Computing ephemerides...</span>
            </div>
        </div>
        
        <!-- TARGET SELECTOR (AUTO / SUN / MOON) -->
        <div class="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800">
            <button id="btn-target-auto" onclick="setTargetMode('auto')" class="text-[10px] px-2 py-1 rounded font-bold transition text-cyan-400 bg-slate-800">⚡ Auto</button>
            <button id="btn-target-sun" onclick="setTargetMode('sun')" class="text-[10px] px-2 py-1 rounded font-bold transition text-slate-400 hover:text-amber-300">☀️ Sun</button>
            <button id="btn-target-moon" onclick="setTargetMode('moon')" class="text-[10px] px-2 py-1 rounded font-bold transition text-slate-400 hover:text-cyan-300">🌔 Moon</button>
        </div>

        <div class="flex items-center gap-2 text-xs">
            <div class="bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800 flex items-center gap-2">
                <span id="astro-coords" class="text-amber-300 font-bold text-xs">Az: --° | Alt: --°</span>
                <span id="astro-event-badge" class="text-[10px] px-1.5 py-0.5 bg-slate-900 text-slate-300 rounded border border-slate-700 font-mono">--:--</span>
            </div>

            <div class="bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800 flex items-center gap-2">
                <span id="obs-coords" class="text-cyan-400 font-bold text-xs">41.6079, 2.2876</span>
                <span id="obs-alt-badge" class="text-emerald-300 font-bold text-xs">⛰️ 145m</span>
                <button id="lock-btn" onclick="toggleLocationLock()" class="text-[10px] px-1.5 py-0.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 font-bold transition">
                    🔒
                </button>
            </div>
        </div>

        <div class="flex items-center gap-1.5">
            <button onclick="toggleSettingsModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 font-bold transition">
                ⚙️ Settings
            </button>
            <button onclick="toggleAboutModal()" class="bg-indigo-950/80 hover:bg-indigo-900 text-indigo-300 border border-indigo-700/80 text-xs px-2.5 py-1.5 rounded-lg font-bold transition">
                ℹ️ About
            </button>
            <button id="audio-btn" onclick="toggleAudio()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition">
                🔇
            </button>
            <button onclick="locateUser()" class="bg-cyan-600 hover:bg-cyan-500 text-white text-xs px-2.5 py-1.5 rounded-lg font-bold transition shadow-lg shadow-cyan-600/30">
                📍 GPS
            </button>
        </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-4 gap-1.5 flex-grow overflow-hidden">
        <div class="lg:col-span-3 rounded-xl overflow-hidden border border-slate-800 relative shadow-2xl">
            <div id="map"></div>
            
            <div class="absolute top-2.5 right-2.5 z-[1000] bg-slate-950/90 backdrop-blur p-2 rounded-xl text-[9px] border border-slate-800 flex flex-col gap-1 shadow-2xl">
                <span class="font-bold text-slate-400 uppercase text-[8px] mb-0.5">Altitude</span>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#ef4444]"></span> &lt; 3k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#f97316]"></span> 3k - 10k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#eab308]"></span> 10k - 18k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#22c55e]"></span> 18k - 28k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#06b6d4]"></span> 28k - 36k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#a855f7]"></span> &gt; 36k ft</div>
            </div>

            <div class="absolute bottom-2.5 left-2.5 z-[1000] bg-slate-950/90 backdrop-blur px-2.5 py-1.5 rounded-lg text-[11px] border border-slate-800 text-slate-300 flex items-center gap-2">
                <span id="footer-vector-beam" class="text-amber-400 font-bold">─ ─ ─</span>
                <span id="footer-astro-info">Optical Line of Sight</span>
            </div>
        </div>

        <div class="bg-slate-900/95 border border-slate-800 rounded-xl p-2.5 overflow-y-auto flex flex-col gap-2 shadow-2xl">
            <div class="flex justify-between items-center border-b border-slate-800 pb-1.5">
                <h2 class="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                    📡 Sector Traffic (<span id="plane-count">0</span>)
                </h2>
            </div>
            
            <div id="alerts-container" class="flex flex-col gap-1.5 overflow-y-auto max-h-[calc(100vh-170px)]">
                <div class="text-xs text-slate-500 text-center py-8">Tracking aircraft in sector...</div>
            </div>
        </div>
    </div>

    <!-- MODAL SETTINGS -->
    <div id="settings-modal" class="fixed inset-0 z-[2000] bg-black/70 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-4 max-w-sm w-full shadow-2xl flex flex-col gap-3">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <h3 class="font-bold text-sm text-cyan-400">⚙️ Settings & Calibration</h3>
                <button onclick="toggleSettingsModal()" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <div class="flex flex-col gap-1">
                <label class="text-xs text-slate-300 font-bold">🏢 Rooftop / Building Height</label>
                <div class="flex items-center gap-2">
                    <input id="building-offset" type="number" value="0" min="0" max="200" onchange="updateBuildingOffset(this.value)" class="w-full bg-slate-950 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold">
                    <span class="text-slate-400 text-xs">meters</span>
                </div>
            </div>

            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-xs text-slate-300 font-bold">
                    <span>⏱️ Latency Calibration Offset</span>
                    <span id="calib-val" class="text-amber-400 font-bold">0.0s</span>
                </div>
                <input id="calib-slider" type="range" min="-4.0" max="4.0" step="0.2" value="0.0" oninput="updateCalibration(this.value)" class="w-full h-2 bg-slate-950 rounded-lg cursor-pointer accent-amber-400">
            </div>

            <div class="flex justify-between items-center pt-2 border-t border-slate-800">
                <button onclick="toggleMapLayer()" id="layer-btn" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 font-bold">
                    🗺️ Switch Basemap
                </button>
                <button onclick="toggleSettingsModal()" class="bg-cyan-600 hover:bg-cyan-500 text-xs text-white px-4 py-1.5 rounded-lg font-bold">
                    Done
                </button>
            </div>
        </div>
    </div>

    <!-- MODAL ABOUT -->
    <div id="about-modal" class="fixed inset-0 z-[2000] bg-black/80 backdrop-blur-md hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-5 max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl flex flex-col gap-3.5">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <div>
                    <h3 class="font-black text-base text-amber-400">ℹ️ About Astronomical Transit Radar PRO</h3>
                    <p class="text-[10px] text-slate-400">Solar & Lunar Transits with NASA JPL Ephemerides</p>
                </div>
                <button onclick="toggleAboutModal()" class="text-slate-400 hover:text-white font-bold text-lg">✕</button>
            </div>

            <div class="text-xs text-slate-300 flex flex-col gap-3 leading-relaxed">
                <div class="bg-amber-950/40 p-3 rounded-xl border border-amber-700/60 text-amber-200">
                    <h4 class="font-bold text-amber-400 mb-1">⚠️ Solar Transit Safety Warning</h4>
                    <p>Never observe or photograph solar transits through an optical instrument (telescope, binoculars, telephoto lens) without a certified <b>full-aperture solar filter (ND5.0 / Baader Solar Safety Film)</b>. Irreversible eye and sensor damage will occur instantly without adequate protection.</p>
                </div>

                <div class="bg-slate-950/70 p-3 rounded-xl border border-slate-800">
                    <h4 class="font-bold text-cyan-400 mb-1">1. Dual Astrometric Engine (Sun & Moon)</h4>
                    <p>Real-time calculation of topocentric coordinates with topocentric parallax using the <b>NASA JPL DE421</b> planetary ephemeris. Supports automated daytime (☀️) and nighttime (🌔) switching with 4D kinematic extrapolation.</p>
                </div>

                <div class="bg-slate-950/70 p-3 rounded-xl border border-slate-800">
                    <h4 class="font-bold text-cyan-400 mb-1">2. Precision Vertical Offsets</h4>
                    <p>Calculates the angular difference ($\Delta\text{Alt} = \text{Alt}_\text{aircraft} - \text{Alt}_\text{astro}$) at the exact instant of closest approach (TCA), denoting passes relative to the Northern/Southern solar or lunar limbs.</p>
                </div>
            </div>

            <div class="text-right pt-1">
                <button onclick="toggleAboutModal()" class="bg-slate-800 hover:bg-slate-700 text-xs text-white px-4 py-1.5 rounded-lg font-bold">
                    Close
                </button>
            </div>
        </div>
    </div>

    <script>
        let savedLat = localStorage.getItem('obs_lat');
        let savedLon = localStorage.getItem('obs_lon');
        let savedBuilding = localStorage.getItem('obs_building_m');
        let savedCalib = localStorage.getItem('obs_calib');
        let currentTargetMode = localStorage.getItem('obs_target_mode') || 'auto';

        let observerLat = savedLat ? parseFloat(savedLat) : 41.6079;
        let observerLon = savedLon ? parseFloat(savedLon) : 2.2876;
        let terrainElevationM = 145.0;
        let buildingOffsetM = savedBuilding ? parseFloat(savedBuilding) : 0.0;
        let timingCalibrationSec = savedCalib ? parseFloat(savedCalib) : 0.0;

        let serverClockDelta = 0.0;
        let isLocationLocked = true;
        let activeBody = { type: 'sun', name: 'Sun', symbol: '☀️', azimuth: 0, elevation: 0, visible: false };
        let audioEnabled = false;
        let audioContext = null;
        let lastBeepedFlight = 0;
        let lastAnimTime = 0;

        let map, obsMarker, astroLine, astroIconMarker;
        let currentBaseTileLayer, isSatelliteMode = false;
        
        const planesState = {};
        let activeAircraftData = [];

        const CARTO_KEY = 'cb1_2l65_1_3a8e83de8b889ec5e4e98278';

        map = L.map('map', { preferCanvas: true }).setView([observerLat, observerLon], 10);

        const cartoDarkLayer = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png?key=${CARTO_KEY}`, {
            subdomains: 'abcd',
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
        });

        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 18,
            attribution: 'Esri Satellite &bull; High-Res'
        });

        currentBaseTileLayer = cartoDarkLayer;
        currentBaseTileLayer.addTo(map);

        function toggleMapLayer() {
            map.removeLayer(currentBaseTileLayer);
            if (!isSatelliteMode) {
                currentBaseTileLayer = satelliteLayer;
                document.getElementById('layer-btn').innerText = "🗺️ Basemap: Satellite HD";
            } else {
                currentBaseTileLayer = cartoDarkLayer;
                document.getElementById('layer-btn').innerText = "🗺️ Basemap: Dark HD";
            }
            isSatelliteMode = !isSatelliteMode;
            currentBaseTileLayer.addTo(map);
        }

        const obsCustomIcon = L.divIcon({
            className: 'obs-container',
            html: '<div class="obs-target"><div class="obs-ring"></div><div class="obs-dot"></div></div>',
            iconSize: [32, 32], iconAnchor: [16, 16]
        });

        obsMarker = L.marker([observerLat, observerLon], { draggable: false, icon: obsCustomIcon }).addTo(map);

        obsMarker.on('drag', function(e) {
            const pos = e.target.getLatLng();
            observerLat = pos.lat; observerLon = pos.lng;
            renderAstroVector();
        });

        obsMarker.on('dragend', function (e) {
            const pos = e.target.getLatLng();
            saveAndSetObserverPos(pos.lat, pos.lng);
        });

        map.on('click', function(e) {
            if (!isLocationLocked) {
                obsMarker.setLatLng(e.latlng);
                saveAndSetObserverPos(e.latlng.lat, e.latlng.lng);
            }
        });

        function setTargetMode(mode) {
            currentTargetMode = mode;
            localStorage.setItem('obs_target_mode', mode);
            updateTargetButtons();
            fetchData();
        }

        function updateTargetButtons() {
            const bAuto = document.getElementById('btn-target-auto');
            const bSun = document.getElementById('btn-target-sun');
            const bMoon = document.getElementById('btn-target-moon');
            
            [bAuto, bSun, bMoon].forEach(b => {
                b.className = "text-[10px] px-2 py-1 rounded font-bold transition text-slate-400 hover:text-white";
            });

            if (currentTargetMode === 'auto') {
                bAuto.className = "text-[10px] px-2 py-1 rounded font-bold transition text-cyan-300 bg-slate-800 border border-slate-700";
            } else if (currentTargetMode === 'sun') {
                bSun.className = "text-[10px] px-2 py-1 rounded font-bold transition text-amber-300 bg-amber-950 border border-amber-700";
            } else if (currentTargetMode === 'moon') {
                bMoon.className = "text-[10px] px-2 py-1 rounded font-bold transition text-cyan-300 bg-cyan-950 border border-cyan-700";
            }
        }

        function toggleLocationLock() {
            isLocationLocked = !isLocationLocked;
            const btn = document.getElementById('lock-btn');
            if (isLocationLocked) {
                obsMarker.dragging.disable();
                btn.innerText = "🔒";
                btn.className = "text-[10px] px-1.5 py-0.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 font-bold transition";
            } else {
                obsMarker.dragging.enable();
                btn.innerText = "🔓";
                btn.className = "text-[10px] px-1.5 py-0.5 bg-amber-600 text-white rounded font-bold transition animate-pulse";
            }
        }

        function toggleSettingsModal() {
            const m = document.getElementById('settings-modal');
            m.classList.toggle('hidden');
            m.classList.toggle('flex');
        }

        function toggleAboutModal() {
            const m = document.getElementById('about-modal');
            m.classList.toggle('hidden');
            m.classList.toggle('flex');
        }

        async function fetchTerrainElevation(lat, lon) {
            try {
                const res = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}`);
                const data = await res.json();
                if (data.elevation && data.elevation.length > 0) {
                    terrainElevationM = parseFloat(data.elevation[0]);
                    updateAltitudeDisplay();
                }
            } catch (err) {
                updateAltitudeDisplay();
            }
        }

        function updateAltitudeDisplay() {
            const totalAlt = terrainElevationM + buildingOffsetM;
            document.getElementById('obs-alt-badge').innerText = `⛰️ ${totalAlt.toFixed(0)}m`;
        }

        function updateBuildingOffset(val) {
            buildingOffsetM = Math.max(0, parseFloat(val) || 0);
            localStorage.setItem('obs_building_m', buildingOffsetM.toString());
            updateAltitudeDisplay();
            fetchData();
        }

        async function saveAndSetObserverPos(lat, lon) {
            observerLat = lat; observerLon = lon;
            localStorage.setItem('obs_lat', lat.toString());
            localStorage.setItem('obs_lon', lon.toString());
            document.getElementById('obs-coords').innerText = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
            
            fetchTerrainElevation(lat, lon);
            renderAstroVector();
            fetchData();
        }

        function locateUser() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(pos => {
                    saveAndSetObserverPos(pos.coords.latitude, pos.coords.longitude);
                    map.setView([pos.coords.latitude, pos.coords.longitude], 11);
                    obsMarker.setLatLng([pos.coords.latitude, pos.coords.longitude]);
                });
            }
        }

        function updateCalibration(val) {
            timingCalibrationSec = parseFloat(val);
            localStorage.setItem('obs_calib', timingCalibrationSec.toString());
            document.getElementById('calib-val').innerText = (timingCalibrationSec >= 0 ? '+' : '') + timingCalibrationSec.toFixed(1) + 's';
        }

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('audio-btn');
            if (audioEnabled) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                btn.innerText = "🔔";
                btn.className = "bg-emerald-600 text-white text-xs px-2 py-1.5 rounded-lg font-bold transition";
                playTone(880, 0.1);
            } else {
                btn.innerText = "🔇";
                btn.className = "bg-slate-800 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition";
            }
        }

        function playTone(freq, duration) {
            if (!audioEnabled || !audioContext) return;
            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();
            osc.frequency.value = freq;
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.15, audioContext.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioContext.destination);
            osc.start();
            osc.stop(audioContext.currentTime + duration);
        }

        function getAltitudeColor(altFt) {
            if (altFt < 3000) return '#ef4444';
            if (altFt < 10000) return '#f97316';
            if (altFt < 18000) return '#eab308';
            if (altFt < 28000) return '#22c55e';
            if (altFt < 36000) return '#06b6d4';
            return '#a855f7';
        }

        function renderAstroVector() {
            if (!activeBody.visible) {
                if (astroLine) map.removeLayer(astroLine);
                if (astroIconMarker) map.removeLayer(astroIconMarker);
                return;
            }

            const distKm = 55;
            const radAz = (activeBody.azimuth * Math.PI) / 180;
            const dLat = (distKm * Math.cos(radAz)) / 111.0;
            const dLon = (distKm * Math.sin(radAz)) / (111.0 * Math.cos(observerLat * Math.PI / 180));
            const endPoint = [observerLat + dLat, observerLon + dLon];

            const lineColor = activeBody.type === 'sun' ? '#f59e0b' : '#38bdf8';

            if (astroLine) {
                astroLine.setLatLngs([[observerLat, observerLon], endPoint]);
                astroLine.setStyle({ color: lineColor });
            } else {
                astroLine = L.polyline([[observerLat, observerLon], endPoint], {
                    color: lineColor, weight: 2, dashArray: '5, 8', opacity: 0.95
                }).addTo(map);
            }

            if (astroIconMarker) {
                astroIconMarker.setLatLng(endPoint);
                const el = astroIconMarker.getElement();
                if (el) el.innerHTML = `<div>${activeBody.symbol}</div>`;
            } else {
                const astroDiv = L.divIcon({
                    className: 'astro-marker', html: `<div>${activeBody.symbol}</div>`, iconSize: [26, 26], iconAnchor: [13, 13]
                });
                astroIconMarker = L.marker(endPoint, { icon: astroDiv }).addTo(map);
            }
        }

        // =========================================================================
        // DATA FETCH & SMOOTH RENDERING
        // =========================================================================
        async function fetchData() {
            try {
                const totalObserverAlt = terrainElevationM + buildingOffsetM;
                const res = await fetch(`/api/data?lat=${observerLat}&lon=${observerLon}&alt=${totalObserverAlt}&target=${currentTargetMode}`);
                const data = await res.json();
                
                if (data.server_time) {
                    serverClockDelta = (Date.now() / 1000.0) - data.server_time;
                }

                activeBody = data.body;

                if (data.source_feed) {
                    document.getElementById('feed-badge').innerText = data.source_feed.toUpperCase();
                }

                document.getElementById('header-symbol').innerText = activeBody.symbol;
                const horizonStatusEl = document.getElementById('astro-horizon-status');
                const badgeEl = document.getElementById('astro-event-badge');
                const footerInfoEl = document.getElementById('footer-astro-info');

                if (activeBody.visible) {
                    document.getElementById('astro-coords').innerText = `${activeBody.name}: Az ${activeBody.azimuth}° | Alt +${activeBody.elevation}°`;
                    horizonStatusEl.innerText = `${activeBody.symbol} ${activeBody.name} Visible (${activeBody.event_type === 'SET' ? 'Sets' : 'Rises'} at ${activeBody.next_event_str})`;
                    horizonStatusEl.className = activeBody.type === 'sun' ? "text-[10px] text-amber-300 font-bold" : "text-[10px] text-cyan-300 font-bold";
                    badgeEl.innerText = `${activeBody.event_type}: ${activeBody.next_event_str}`;
                    badgeEl.className = activeBody.type === 'sun' ? "text-[10px] px-1.5 py-0.5 bg-amber-950 text-amber-300 rounded border border-amber-800 font-mono" : "text-[10px] px-1.5 py-0.5 bg-cyan-950 text-cyan-300 rounded border border-cyan-800 font-mono";
                    footerInfoEl.innerText = `Line of sight to the ${activeBody.name} (Visible)`;
                } else {
                    document.getElementById('astro-coords').innerText = `${activeBody.name} Hidden (${activeBody.elevation}°)`;
                    horizonStatusEl.innerText = `🌑 ${activeBody.name} Below Horizon (${activeBody.event_type === 'RISE' ? 'Rises' : 'Sets'} at ${activeBody.next_event_str})`;
                    horizonStatusEl.className = "text-[10px] text-slate-400 font-bold";
                    badgeEl.innerText = `${activeBody.event_type}: ${activeBody.next_event_str}`;
                    badgeEl.className = "text-[10px] px-1.5 py-0.5 bg-slate-900 text-slate-400 rounded border border-slate-700 font-mono";
                    footerInfoEl.innerText = `${activeBody.name} below horizon (Rises at ${activeBody.next_event_str})`;
                }

                renderAstroVector();

                activeAircraftData = data.aircraft || [];
                document.getElementById('plane-count').innerText = activeAircraftData.length;
                const currentCallsigns = new Set();
                const nowSec = (Date.now() / 1000.0) - serverClockDelta;

                activeAircraftData.forEach(plane => {
                    const cs = plane.callsign;
                    currentCallsigns.add(cs);

                    const color = getAltitudeColor(plane.alt_ft);
                    const glow = (plane.is_transit && activeBody.visible) ? 'filter: drop-shadow(0 0 12px #ef4444);' : (plane.is_close && activeBody.visible ? 'filter: drop-shadow(0 0 8px #f59e0b);' : '');

                    const planeHtml = `
                        <div style="transform: rotate(${plane.track}deg); width: 24px; height: 24px; display:flex; align-items:center; justify-content:center; ${glow}">
                            <svg viewBox="0 0 24 24" width="22" height="22" fill="${color}">
                                <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
                            </svg>
                        </div>
                    `;

                    if (!planesState[cs]) {
                        const marker = L.marker([plane.lat, plane.lon], {
                            icon: L.divIcon({ className: 'p-icon', html: planeHtml, iconSize: [24, 24], iconAnchor: [12, 12] })
                        }).addTo(map);

                        marker.bindPopup(`
                            <div style="font-family: monospace; font-size: 11px; color:#000;">
                                <b>FLIGHT:</b> ${plane.callsign} (${plane.model})<br>
                                <b>REG:</b> ${plane.reg || 'N/A'}<br>
                                <b>ALTITUDE:</b> ${plane.alt_ft} ft (${plane.vr_fpm > 300 ? '↗' : (plane.vr_fpm < -300 ? '↘' : '→')} ${plane.vr_fpm} ft/min)<br>
                                <b>SPEED:</b> ${plane.speed_kt} kt | <b>DIST:</b> ${plane.distance_km} km<br>
                                ${activeBody.visible ? `<b>VERT OFFSET:</b> ${plane.vertical_offset_deg > 0 ? '+' : ''}${plane.vertical_offset_deg}° (${plane.vertical_dir_text})<br><b>SEPARATION:</b> ${plane.min_sep}°<br><b>TCA:</b> ${plane.tca_seconds}s` : `<b>HEADING:</b> ${plane.track}°`}
                            </div>
                        `);

                        planesState[cs] = {
                            marker: marker,
                            curLat: plane.lat,
                            curLon: plane.lon,
                            speedMs: plane.speed_ms,
                            trackRad: (plane.track * Math.PI) / 180.0,
                            lastSeenTime: nowSec,
                            data: plane
                        };
                    } else {
                        const st = planesState[cs];
                        st.curLat = (st.curLat + plane.lat) / 2.0;
                        st.curLon = (st.curLon + plane.lon) / 2.0;
                        st.speedMs = plane.speed_ms;
                        st.trackRad = (plane.track * Math.PI) / 180.0;
                        st.lastSeenTime = nowSec;
                        st.data = plane;

                        st.marker.setIcon(L.divIcon({ className: 'p-icon', html: planeHtml, iconSize: [24, 24], iconAnchor: [12, 12] }));
                    }
                });

                for (let cs in planesState) {
                    const st = planesState[cs];
                    const ageSinceSeen = nowSec - st.lastSeenTime;
                    if (ageSinceSeen > 15.0) {
                        map.removeLayer(st.marker);
                        delete planesState[cs];
                    } else if (!currentCallsigns.has(cs)) {
                        const el = st.marker.getElement();
                        if (el) el.style.opacity = '0.45';
                    } else {
                        const el = st.marker.getElement();
                        if (el) el.style.opacity = '1.0';
                    }
                }

            } catch (err) {
                console.error("Error fetchData:", err);
            }
        }

        // 60 FPS Continuous Frame Animation
        function animateFrame() {
            const now = performance.now() / 1000.0;
            const dt = Math.min(0.08, Math.max(0.001, now - (lastAnimTime || now)));
            lastAnimTime = now;

            for (const cs in planesState) {
                const p = planesState[cs];
                const dist = p.speedMs * dt;
                p.curLat += (dist * Math.cos(p.trackRad)) / 111139.0;
                p.curLon += (dist * Math.sin(p.trackRad)) / (111139.0 * Math.cos(p.curLat * Math.PI / 180.0));
                p.marker.setLatLng([p.curLat, p.curLon]);
            }

            requestAnimationFrame(animateFrame);
        }

        // Live HUD Countdowns & Geometry Telemetry
        function updateHUDCountdowns() {
            const container = document.getElementById('alerts-container');
            if (!activeAircraftData || activeAircraftData.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-500 text-center py-8">Tracking aircraft in sector...</div>';
                return;
            }

            const now = (Date.now() / 1000.0) - serverClockDelta;
            activeAircraftData.sort((a, b) => a.min_sep - b.min_sep);

            let html = '';
            activeAircraftData.forEach(plane => {
                const remaining = Math.max(0.0, (plane.tca_epoch - now) + timingCalibrationSec);
                const isTransit = plane.is_transit && activeBody.visible;
                const isClose = plane.is_close && activeBody.visible;

                const mins = Math.floor(remaining / 60);
                const secs = (remaining % 60).toFixed(1).padStart(4, '0');
                const timerStr = remaining > 0 ? `T-${mins.toString().padStart(2, '0')}:${secs}` : `CROSSING!`;

                let cardBorder = 'border-slate-800 bg-slate-950/60';
                let tagHtml = `<span class="bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded text-[8px] font-bold">EN ROUTE</span>`;
                let statusDetails = '';

                if (!activeBody.visible) {
                    tagHtml = `<span class="bg-slate-900 text-slate-500 px-1.5 py-0.5 rounded text-[8px] font-bold">${activeBody.name.toUpperCase()} DOWN</span>`;
                    statusDetails = `
                        <span>Heading: <b>${plane.track}°</b></span>
                        <span>Speed: <b>${plane.speed_kt} kt</b></span>
                    `;
                } else if (isTransit) {
                    cardBorder = 'border-red-500 bg-red-950/70 shadow-lg shadow-red-950/60';
                    tagHtml = `<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[9px] font-black animate-pulse">🎯 TRANSIT (${plane.transit_duration_s}s)!</span>`;
                    
                    if (remaining > 0 && remaining <= 15.0 && lastBeepedFlight !== plane.callsign) {
                        playTone(1200, 0.25);
                        lastBeepedFlight = plane.callsign;
                    }
                    
                    statusDetails = `
                        <span>Intersection: <b class="text-red-400 font-bold">${plane.position_descriptor}</b></span>
                        <span>Sep: <b class="text-red-400 font-black">${plane.min_sep}°</b></span>
                    `;
                } else {
                    if (isClose) {
                        cardBorder = 'border-amber-500 bg-amber-950/60';
                        tagHtml = `<span class="bg-amber-600 text-white px-2 py-0.5 rounded text-[8px] font-bold">⚠️ CLOSE PASS</span>`;
                    }
                    
                    const sign = plane.vertical_offset_deg > 0 ? '+' : '';
                    const vertColor = isClose ? 'text-amber-300' : 'text-slate-200';
                    statusDetails = `
                        <span class="col-span-2">Vert Offset: <b class="${vertColor}">${sign}${plane.vertical_offset_deg}° ${plane.vertical_dir_text} (${plane.vertical_body_diams} ${activeBody.symbol} / ${sign}${plane.vertical_offset_m}m)</b></span>
                    `;
                }

                html += `
                    <div onclick="focusPlane('${plane.callsign}')" 
                         class="p-2.5 rounded-xl border ${cardBorder} text-xs flex flex-col gap-1 transition cursor-pointer hover:border-cyan-500">
                        <div class="flex justify-between items-center">
                            <div class="flex items-center gap-1.5">
                                <span class="font-black text-sm ${isTransit ? 'text-red-400' : 'text-slate-100'}">${plane.callsign}</span>
                                <span class="px-1.5 py-0.2 bg-slate-800 text-cyan-300 border border-slate-700 rounded text-[9px] font-bold">${plane.model}</span>
                            </div>
                            ${tagHtml}
                        </div>
                        
                        <div class="grid grid-cols-2 gap-1 text-[10.5px] text-slate-300">
                            <span>Alt: <b>${(plane.alt_ft || 0).toLocaleString()} ft</b> ${plane.vr_fpm > 300 ? '↗' : (plane.vr_fpm < -300 ? '↘' : '→')}</span>
                            <span>V/S: <b>${plane.vr_fpm} ft/min</b></span>
                            <span>Dist: <b>${plane.distance_km} km</b></span>
                            <span>TCA: <b class="${isTransit ? 'text-red-400 font-black text-xs' : (isClose ? 'text-amber-300' : 'text-cyan-300')} font-mono">${timerStr}</b></span>
                            ${statusDetails}
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }

        function focusPlane(callsign) {
            if (planesState[callsign]) {
                map.setView([planesState[callsign].curLat, planesState[callsign].curLon], 11);
                planesState[callsign].marker.openPopup();
            }
        }

        document.getElementById('calib-slider').value = timingCalibrationSec.toString();
        document.getElementById('calib-val').innerText = (timingCalibrationSec >= 0 ? '+' : '') + timingCalibrationSec.toFixed(1) + 's';
        document.getElementById('building-offset').value = buildingOffsetM.toString();
        document.getElementById('obs-coords').innerText = `${observerLat.toFixed(4)}, ${observerLon.toFixed(4)}`;

        updateTargetButtons();
        fetchData();
        fetchTerrainElevation(observerLat, observerLon);
        setInterval(fetchData, 2500);
        setInterval(updateHUDCountdowns, 100);
        requestAnimationFrame(animateFrame);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print(f" [OK] ASTRONOMICAL TRANSIT RADAR PRO (SOLAR & LUNAR)")
    print(f" [OK] Running on port: {port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
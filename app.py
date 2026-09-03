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
# 1. EFEMÉRIDES NASA JPL Y CONSTANTES GEODÉSICAS WGS84
# =========================================================================
ts = load.timescale()
eph = load('de421.bsp')
moon = eph['moon']
sun = eph['sun']
earth = eph['earth']

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
R_EARTH = 6371000.0

MOON_RADIUS_KM = 1737.4
SUN_RADIUS_KM = 696340.0

WINGSPANS = {
    'A318': 34.1, 'A319': 35.8, 'A320': 35.8, 'A321': 35.8,
    'A332': 60.3, 'A333': 60.3, 'A339': 64.0, 'A359': 64.7, 'A35K': 64.7,
    'A388': 79.8, 'B737': 35.8, 'B738': 35.8, 'B739': 35.8, 'B38M': 35.9,
    'B744': 64.4, 'B748': 68.4, 'B752': 38.0, 'B763': 47.6, 'B772': 60.9,
    'B77W': 64.8, 'B788': 60.1, 'B789': 60.1, 'B78X': 60.1, 'E190': 28.7,
    'E195': 28.7, 'CRJ9': 24.9, 'AT76': 27.0, 'CONC': 25.6, 'A400': 42.4,
    'C17': 51.75, 'C130': 40.4, 'B52': 56.4
}

def get_wingspan(model_icao):
    return WINGSPANS.get(model_icao, 35.0)

def calculate_atmosphere(alt_m):
    p_mbar = 1013.25 * math.pow((1.0 - 2.25577e-5 * max(0.0, alt_m)), 5.25588)
    t_c = 15.0 - (0.0065 * alt_m)
    return max(300.0, p_mbar), t_c

def diff_angle_deg(a, b):
    """Diferencia angular modular correcta para evitar errores en 0/360 deg"""
    return (a - b + 180.0) % 360.0 - 180.0

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
# GESTIÓN DE CACHÉ DE VUELOS (COMPATIBLE CON GUNICORN / RENDER)
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

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) LunarTransitRadar/25.0'}
    
    # 1. airplanes.live
    try:
        url = f"https://api.airplanes.live/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            if ac:
                CACHE['lat'] = cur_lat
                CACHE['lon'] = cur_lon
                CACHE['timestamp'] = now
                CACHE['aircraft'] = ac
                CACHE['source'] = 'airplanes.live'
                return ac, 'airplanes.live', 0.0
    except Exception:
        pass

    # 2. adsb.lol
    try:
        url = f"https://api.adsb.lol/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            if ac:
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
# MOTOR ASTROMÉTRICO LUNAR (CON SOPORTE SOLAR SECUNDARIO)
# =========================================================================
@app.route('/api/data')
def get_data():
    try:
        lat = float(request.args.get('lat', 41.6079))
        lon = float(request.args.get('lon', 2.2876))
        alt = float(request.args.get('alt', 145.0))
        now_epoch = time.time()

        raw_ac, source_feed, cache_age_sec = get_live_aircraft(lat, lon)

        t_now = ts.now()
        topos_loc = wgs84.latlon(lat, lon, elevation_m=alt)
        obs_loc = earth + topos_loc
        p_mbar, t_c = calculate_atmosphere(alt)

        # 1. Astrometría Lunar (Principal) con Refracción ISA
        app_moon = obs_loc.at(t_now).observe(moon).apparent()
        m_alt, m_az, m_dist = app_moon.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
        moon_az0 = float(m_az.degrees)
        moon_alt0 = float(m_alt.degrees)
        moon_radius_deg = float(math.degrees(math.asin(MOON_RADIUS_KM / m_dist.km)))
        moon_is_visible = bool(moon_alt0 > -0.5)

        # 2. Astrometría Solar (Secundaria) con Refracción ISA
        app_sun = obs_loc.at(t_now).observe(sun).apparent()
        s_alt, s_az, s_dist = app_sun.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
        sun_az0 = float(s_az.degrees)
        sun_alt0 = float(s_alt.degrees)
        sun_radius_deg = float(math.degrees(math.asin(SUN_RADIUS_KM / s_dist.km)))
        sun_is_visible = bool(sun_alt0 > -0.5)

        # Derivadas angulares continuas con corrección de wrap-around (300s)
        dt_future = datetime.now(timezone.utc) + timedelta(seconds=300)
        t_300 = ts.from_datetime(dt_future)
        
        d_az_dt_moon, d_alt_dt_moon = 0.0, 0.0
        if moon_is_visible:
            app_m300 = obs_loc.at(t_300).observe(moon).apparent()
            ma300, mz300, _ = app_m300.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
            d_az_dt_moon = diff_angle_deg(float(mz300.degrees), moon_az0) / 300.0
            d_alt_dt_moon = (float(ma300.degrees) - moon_alt0) / 300.0

        d_az_dt_sun, d_alt_dt_sun = 0.0, 0.0
        if sun_is_visible:
            app_s300 = obs_loc.at(t_300).observe(sun).apparent()
            sa300, sz300, _ = app_s300.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
            d_az_dt_sun = diff_angle_deg(float(sz300.degrees), sun_az0) / 300.0
            d_alt_dt_sun = (float(sa300.degrees) - sun_alt0) / 300.0

        # Salida / Puesta de Luna y Sol
        def get_next_event(body_obj, is_vis):
            try:
                t_end = ts.from_datetime(datetime.now(timezone.utc) + timedelta(hours=36))
                f_rs = almanac.risings_and_settings(eph, body_obj, topos_loc)
                times_rs, events_rs = almanac.find_discrete(t_now, t_end, f_rs)
                now_utc = datetime.now(timezone.utc)
                for t_e, ev in zip(times_rs, events_rs):
                    dt_e = t_e.utc_datetime()
                    if is_vis and ev == 0:
                        return "SET", dt_e.strftime('%H:%M UTC'), max(0, int((dt_e - now_utc).total_seconds()))
                    elif not is_vis and ev == 1:
                        return "RISE", dt_e.strftime('%H:%M UTC'), max(0, int((dt_e - now_utc).total_seconds()))
            except Exception:
                pass
            return ("SET" if is_vis else "RISE"), "--:--", 0

        moon_ev_type, moon_ev_str, moon_ev_sec = get_next_event(moon, moon_is_visible)
        sun_ev_type, sun_ev_str, sun_ev_sec = get_next_event(sun, sun_is_visible)

        aircraft_results = []

        for ac in raw_ac:
            raw_lat = ac.get('lat')
            raw_lon = ac.get('lon')
            alt_val = ac.get('alt_geom') or ac.get('alt_baro')
            track = ac.get('track')
            gs = ac.get('gs', 0)
            vr_raw = ac.get('geom_rate', ac.get('baro_rate', 0))
            model_icao = str(ac.get('t', 'A320')).strip().upper()
            wingspan_m = get_wingspan(model_icao)

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

            # Optimización TCA por Sección Áurea
            def compute_body_intercept(body_type, b_az0, b_alt0, b_rad, d_az, d_alt, is_vis):
                if not is_vis:
                    return {
                        'target': body_type,
                        'is_transit': False, 'is_close': False,
                        'min_sep': 99.0, 'current_sep': 99.0,
                        'tca_seconds': 0.0, 'tca_epoch': float(now_epoch),
                        'vertical_offset_deg': 0.0, 'vertical_offset_m': 0,
                        'vertical_body_diams': 0.0, 'vertical_dir_text': '',
                        'transit_duration_s': 0.0,
                        'angular_size_arcsec': 0.0, 'disk_coverage_pct': 0.0,
                        'position_descriptor': "Below Horizon"
                    }

                cur_sep = angular_separation(cur_az, cur_alt, b_az0, b_alt0)

                def eval_t(t_val):
                    p_lat, p_lon = propagate_geodetic_position(ac_lat, ac_lon, speed_ms, track_val, t_val)
                    p_alt_m = alt_m + (vr_ms * t_val)
                    et, nt, ut = ecef_to_enu(*geodetic_to_ecef(p_lat, p_lon, p_alt_m), lat, lon, alt)
                    p_az, p_alt, p_range = enu_to_az_alt(et, nt, ut)
                    if p_alt <= 0: return 999.0, p_az, p_alt, p_range
                    b_az_t = (b_az0 + d_az * t_val) % 360.0
                    b_alt_t = b_alt0 + d_alt * t_val
                    return angular_separation(p_az, p_alt, b_az_t, b_alt_t), p_az, p_alt, p_range

                # Fase 1: Barrido global (Paso 2.0s de 0 a 300s)
                best_t = 0.0
                min_sep = cur_sep
                best_p_alt, best_p_range, best_b_alt = cur_alt, cur_range, b_alt0

                for step in range(0, 151):
                    t_cand = float(step * 2.0)
                    sep_val, p_az, p_alt, p_range = eval_t(t_cand)
                    if sep_val < min_sep:
                        min_sep, best_t = sep_val, t_cand
                        best_p_alt, best_p_range = p_alt, p_range
                        best_b_alt = b_alt0 + d_alt * t_cand

                # Optimización fina por Sección Áurea (Golden Section Search)
                a = max(0.0, best_t - 2.5)
                b = min(300.0, best_t + 2.5)
                phi = (1.0 + math.sqrt(5.0)) / 2.0
                resphi = 2.0 - phi

                x1 = a + resphi * (b - a)
                x2 = b - resphi * (b - a)
                f1, _, _, _ = eval_t(x1)
                f2, _, _, _ = eval_t(x2)

                for _ in range(14):
                    if f1 < f2:
                        b = x2
                        x2 = x1
                        f2 = f1
                        x1 = a + resphi * (b - a)
                        f1, _, _, _ = eval_t(x1)
                    else:
                        a = x1
                        x1 = x2
                        f1 = f2
                        x2 = b - resphi * (b - a)
                        f2, _, _, _ = eval_t(x2)

                t_opt = (a + b) / 2.0
                sep_opt, _, opt_p_alt, opt_p_range = eval_t(t_opt)
                if sep_opt < min_sep:
                    min_sep = sep_opt
                    best_t = t_opt
                    best_p_alt = opt_p_alt
                    best_p_range = opt_p_range
                    best_b_alt = b_alt0 + d_alt * t_opt

                vert_offset_deg = round(best_p_alt - best_b_alt, 3)
                body_diam_deg = 2.0 * b_rad
                vert_body_diams = round(abs(vert_offset_deg) / max(0.01, body_diam_deg), 1)
                vert_offset_m = int(best_p_range * math.tan(math.radians(vert_offset_deg)))

                is_transit = (min_sep <= b_rad)
                is_close = (min_sep > b_rad and min_sep <= (b_rad * 3.5))

                transit_dur = 0.0
                if is_transit and best_t > 0:
                    chord_deg = 2.0 * math.sqrt(max(0.0, (b_rad ** 2) - (min_sep ** 2)))
                    dt_d = 0.5
                    sep_p, _, _, _ = eval_t(max(0.0, best_t - dt_d))
                    sep_n, _, _, _ = eval_t(best_t + dt_d)
                    ang_speed = max(0.1, math.hypot((sep_n - sep_p) / (2 * dt_d), (speed_ms / best_p_range) * (180 / math.pi)))
                    transit_dur = round(chord_deg / ang_speed, 2)

                ang_size_rad = 2.0 * math.atan2(wingspan_m, 2.0 * max(100.0, best_p_range))
                ang_size_arcsec = round(math.degrees(ang_size_rad) * 3600.0, 1)
                disk_coverage_pct = round((ang_size_arcsec / (body_diam_deg * 3600.0)) * 100.0, 1)

                symbol = "🌕" if body_type == 'moon' else "☀️"
                name = "Lunar" if body_type == 'moon' else "Solar"

                if is_transit:
                    if abs(vert_offset_deg) <= (b_rad * 0.35):
                        pos_desc = f"{name} Center"
                    elif vert_offset_deg > 0:
                        pos_desc = "Northern Limb (Above)"
                    else:
                        pos_desc = "Southern Limb (Below)"
                else:
                    dir_txt = "Above" if vert_offset_deg > 0 else "Below"
                    pos_desc = f"{abs(vert_offset_deg):.2f}° {dir_txt} ({vert_body_diams} {symbol})"

                dir_clean = "Above" if vert_offset_deg > 0 else "Below"

                return {
                    'target': body_type,
                    'is_transit': is_transit,
                    'is_close': is_close,
                    'min_sep': round(min_sep, 3),
                    'current_sep': round(cur_sep, 2),
                    'tca_seconds': round(best_t, 1),
                    'tca_epoch': float(now_epoch + best_t) if best_t > 0 else float(now_epoch),
                    'vertical_offset_deg': vert_offset_deg,
                    'vertical_offset_m': vert_offset_m,
                    'vertical_body_diams': vert_body_diams,
                    'vertical_dir_text': dir_clean,
                    'transit_duration_s': transit_dur,
                    'angular_size_arcsec': ang_size_arcsec,
                    'disk_coverage_pct': disk_coverage_pct,
                    'position_descriptor': pos_desc
                }

            moon_data = compute_body_intercept('moon', moon_az0, moon_alt0, moon_radius_deg, d_az_dt_moon, d_alt_dt_moon, moon_is_visible)
            sun_data = compute_body_intercept('sun', sun_az0, sun_alt0, sun_radius_deg, d_az_dt_sun, d_alt_dt_sun, sun_is_visible)

            if moon_is_visible:
                primary = moon_data
            elif sun_is_visible:
                primary = sun_data
            else:
                primary = moon_data if moon_data['min_sep'] <= sun_data['min_sep'] else sun_data

            aircraft_results.append({
                'callsign': str(ac.get('flight') or ac.get('hex', 'UNKNOWN')).strip(),
                'model': model_icao,
                'wingspan_m': wingspan_m,
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
                'moon': moon_data,
                'sun': sun_data,
                'primary': primary
            })

        return jsonify({
            'source_feed': source_feed,
            'server_time': now_epoch,
            'observer_altitude_used_m': alt,
            'moon': {
                'name': 'Moon', 'symbol': '🌕',
                'azimuth': round(moon_az0, 2), 'elevation': round(moon_alt0, 2),
                'radius_deg': round(moon_radius_deg, 3), 'visible': moon_is_visible,
                'event_type': moon_ev_type, 'next_event_str': moon_ev_str, 'next_event_seconds': moon_ev_sec
            },
            'sun': {
                'name': 'Sun', 'symbol': '☀️',
                'azimuth': round(sun_az0, 2), 'elevation': round(sun_alt0, 2),
                'radius_deg': round(sun_radius_deg, 3), 'visible': sun_is_visible,
                'event_type': sun_ev_type, 'next_event_str': sun_ev_str, 'next_event_seconds': sun_ev_sec
            },
            'aircraft': aircraft_results
        })

    except Exception as e:
        print(f"[ENGINE ERROR] {e}")
        return jsonify({'error': str(e), 'aircraft': []})

# =========================================================================
# RUTAS DE INDEXACIÓN (GOOGLE SEARCH CONSOLE) Y FRONTEND
# =========================================================================
@app.route('/google92a4c5b46b2ec0bf.html')
def google_verification():
    return 'google-site-verification: google92a4c5b46b2ec0bf.html'

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# =========================================================================
# WEB UI (LUNAR TRANSIT RADAR PRO)
# =========================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="google-site-verification" content="google92a4c5b46b2ec0bf" />
    
    <!-- SEO & META DESCRIPCIÓN PARA GOOGLE -->
    <meta name="description" content="Lunar Transit Radar PRO: Radar en tiempo real para predicción y seguimiento de tránsitos de aeronaves frente a la Luna y el Sol. Astrometría NASA JPL DE421 y telemetría 4D. Creado por Marc Garrido.">
    <meta name="keywords" content="lunar transit radar, transito lunar avion, solar transit radar, astrofotografia, marc garrido, radar aviones luna">
    <meta name="author" content="Marc Garrido">
    
    <!-- OPEN GRAPH / REDES SOCIALES -->
    <meta property="og:title" content="Lunar Transit Radar PRO">
    <meta property="og:description" content="Predicción en tiempo real de tránsitos de aeronaves frente a la Luna y el Sol con astrometría NASA JPL DE421. Creado por Marc Garrido.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://lunar-transit-radar.onrender.com/">
    
    <title>Lunar Transit Radar PRO</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #060913; color: #f8fafc; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .map-container { height: calc(100dvh - 120px); width: 100%; border-radius: 12px; }
        .leaflet-container { background: #060913 !important; }
        
        .obs-target { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; }
        .obs-ring { position: absolute; width: 32px; height: 32px; border-radius: 50%; background: rgba(6, 182, 212, 0.25); border: 2px solid #06b6d4; animation: pulse-ring 2s infinite ease-out; }
        .obs-dot { width: 10px; height: 10px; border-radius: 50%; background: #22d3ee; border: 2px solid #ffffff; box-shadow: 0 0 14px #06b6d4; z-index: 10; }
        @keyframes pulse-ring { 0% { transform: scale(0.5); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }
        
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #060913; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    </style>
</head>
<body class="p-1.5 md:p-2 flex flex-col h-[100dvh] overflow-hidden select-none">
    
    <!-- TOP STATUS & CONTROL BAR -->
    <header class="bg-slate-900/90 backdrop-blur border border-slate-800 px-3 py-2 rounded-xl mb-1.5 flex flex-wrap justify-between items-center gap-2 shadow-2xl">
        <div class="flex items-center gap-2">
            <div class="flex items-center gap-1.5">
                <span class="text-2xl animate-pulse">🌔</span>
                <div>
                    <h1 class="text-xs font-black text-amber-400 tracking-wider">LUNAR RADAR PRO</h1>
                    <div class="flex items-center gap-1">
                        <span id="feed-badge" class="text-[8px] px-1 py-0.2 bg-emerald-950 text-emerald-300 border border-emerald-700 rounded font-bold">ONLINE</span>
                        <span class="text-[8px] px-1 py-0.2 bg-indigo-950 text-indigo-300 border border-indigo-700 rounded font-mono">DE421 JPL</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- CELESTIAL BODIES (LUNA PRINCIPAL / SOL SECUNDARIO) -->
        <div class="flex items-center gap-1.5 text-xs">
            <div id="moon-status-card" class="bg-slate-950 px-2.5 py-1 rounded-lg border border-cyan-900/50 flex items-center gap-1.5 cursor-pointer hover:border-cyan-500 transition" onclick="setFilterMode('moon')">
                <span class="text-sm">🌕</span>
                <span id="moon-coords" class="text-cyan-300 font-bold text-[11px]">Moon: Az --° | Alt --°</span>
                <span id="moon-badge" class="text-[9px] px-1 bg-cyan-950 text-cyan-400 rounded border border-cyan-800 font-mono">--:--</span>
            </div>

            <div id="sun-status-card" class="bg-slate-950 px-2.5 py-1 rounded-lg border border-amber-900/30 flex items-center gap-1.5 cursor-pointer hover:border-amber-500 transition opacity-80" onclick="setFilterMode('sun')">
                <span class="text-sm">☀️</span>
                <span id="sun-coords" class="text-amber-300 font-bold text-[11px]">Sun: Az --° | Alt --°</span>
                <span id="sun-badge" class="text-[9px] px-1 bg-amber-950 text-amber-400 rounded border border-amber-800 font-mono">--:--</span>
            </div>

            <div class="hidden sm:flex bg-slate-950 px-2 py-1 rounded-lg border border-slate-800 items-center gap-1.5">
                <span id="obs-coords" class="text-cyan-400 font-bold text-[11px]">41.6079, 2.2876</span>
                <span id="obs-alt-badge" class="text-emerald-300 font-bold text-[11px]">⛰️ 145m</span>
                <button id="lock-btn" onclick="toggleLocationLock()" class="text-[10px] px-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 font-bold transition">
                    🔒
                </button>
            </div>
        </div>

        <!-- TARGET MODE SWITCH & TOOLS -->
        <div class="flex items-center gap-1.5">
            <div class="bg-slate-950 p-0.5 rounded-lg border border-slate-800 flex">
                <button id="btn-flt-moon" onclick="setFilterMode('moon')" class="text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">🌔 Moon</button>
                <button id="btn-flt-sun" onclick="setFilterMode('sun')" class="text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-amber-300">☀️ Sun</button>
                <button id="btn-flt-all" onclick="setFilterMode('all')" class="text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-white">Dual</button>
            </div>

            <button onclick="toggleAboutModal()" class="bg-indigo-950/80 hover:bg-indigo-900 text-indigo-300 border border-indigo-700/80 text-xs px-2 py-1.5 rounded-lg font-bold transition" title="Acerca de">
                ℹ️
            </button>
            <button id="voice-btn" onclick="toggleVoice()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition" title="Voice Alerts">
                🗣️
            </button>
            <button id="audio-btn" onclick="toggleAudio()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition" title="Sonar Audio">
                🔇
            </button>
            <button onclick="toggleSettingsModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition">
                ⚙️
            </button>
            <button onclick="locateUser()" class="bg-cyan-600 hover:bg-cyan-500 text-white text-xs px-2.5 py-1.5 rounded-lg font-bold transition shadow-lg shadow-cyan-600/30">
                📍
            </button>
        </div>
    </header>

    <!-- MAIN INTERFACE GRID -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-1.5 flex-grow overflow-hidden">
        
        <!-- MAP RADAR SECTION -->
        <div class="lg:col-span-3 rounded-xl overflow-hidden border border-slate-800 relative shadow-2xl flex flex-col">
            <div id="map" class="map-container"></div>
            
            <!-- ALTITUDE COLOR LEGEND -->
            <div class="hidden sm:flex absolute top-2.5 right-2.5 z-[1000] bg-slate-950/90 backdrop-blur p-2 rounded-xl text-[9px] border border-slate-800 flex-col gap-1 shadow-2xl">
                <span class="font-bold text-slate-400 uppercase text-[8px] mb-0.5">Flight Level</span>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#ef4444]"></span> &lt; 3k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#f97316]"></span> 3k - 10k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#eab308]"></span> 10k - 18k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#22c55e]"></span> 18k - 28k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#06b6d4]"></span> 28k - 36k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#a855f7]"></span> &gt; 36k ft</div>
            </div>

            <!-- FOOTER INFO BAR -->
            <div class="absolute bottom-2.5 left-2.5 z-[1000] bg-slate-950/90 backdrop-blur px-2.5 py-1.5 rounded-lg text-[11px] border border-slate-800 text-slate-300 flex items-center gap-2">
                <span id="footer-vector-indicator" class="font-bold text-cyan-400">🌕──────</span>
                <span id="footer-astro-info">Optical Line of Sight to the Moon</span>
            </div>
        </div>

        <!-- TELEMETRY & ALERTS HUD -->
        <div class="bg-slate-900/95 border border-slate-800 rounded-xl p-2.5 overflow-y-auto flex flex-col gap-2 shadow-2xl max-h-[40vh] lg:max-h-full">
            <div class="flex justify-between items-center border-b border-slate-800 pb-1.5">
                <h2 class="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <span>📡 Sector Traffic</span>
                    <span id="plane-count" class="bg-cyan-950 text-cyan-300 px-1.5 py-0.2 rounded-full text-[9px] border border-cyan-800">0</span>
                </h2>
                <span id="filter-indicator" class="text-[9px] text-cyan-400 font-mono font-bold">TARGET: 🌕 MOON</span>
            </div>
            
            <div id="alerts-container" class="flex flex-col gap-1.5 overflow-y-auto">
                <div class="text-xs text-slate-500 text-center py-8">Tracking lunar intercept traffic...</div>
            </div>
        </div>
    </div>

    <!-- MODAL: ACERCA DE (ABOUT / AUTORÍA) -->
    <div id="about-modal" class="fixed inset-0 z-[2000] bg-black/80 backdrop-blur-md hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-5 max-w-lg w-full shadow-2xl flex flex-col gap-3.5">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <div>
                    <h3 class="font-black text-base text-amber-400">ℹ️ Acerca de Lunar Transit Radar PRO</h3>
                    <p class="text-[10px] text-slate-400">Astrometría NASA JPL DE421 & Cinemática de Tránsitos 4D</p>
                </div>
                <button onclick="toggleAboutModal()" class="text-slate-400 hover:text-white font-bold text-lg">✕</button>
            </div>

            <div class="text-xs text-slate-300 flex flex-col gap-3 leading-relaxed">
                <!-- TARJETA DEL CREADOR -->
                <div class="bg-gradient-to-r from-amber-950/70 via-slate-950 to-slate-950 p-3.5 rounded-xl border border-amber-600/60 flex items-center justify-between shadow-lg">
                    <div>
                        <span class="text-[10px] text-amber-400 font-black uppercase tracking-wider block mb-0.5">👨‍💻 Creador & Desarrollador</span>
                        <span class="text-sm font-black text-white tracking-wide">Marc Garrido</span>
                    </div>
                    <span class="text-2xl">🚀</span>
                </div>

                <div class="bg-slate-950/70 p-3 rounded-xl border border-slate-800">
                    <h4 class="font-bold text-cyan-400 mb-1">📐 Motor Astrométrico & Geodésico</h4>
                    <ul class="list-disc pl-4 space-y-1 text-[11px] text-slate-400">
                        <li><b>Efemérides:</b> NASA JPL DE421 con paralaje topocéntrico y refracción ISA.</li>
                        <li><b>Geodesia:</b> WGS84 Elipsoidal ($\text{ECEF} \to \text{ENU}$).</li>
                        <li><b>Optimización TCA:</b> Búsqueda de Sección Áurea a $\pm 3\text{ ms}$.</li>
                    </ul>
                </div>
            </div>

            <div class="text-right pt-2 border-t border-slate-800">
                <button onclick="toggleAboutModal()" class="bg-slate-800 hover:bg-slate-700 text-xs text-white px-4 py-1.5 rounded-lg font-bold">
                    Cerrar
                </button>
            </div>
        </div>
    </div>

    <!-- MODAL: SETTINGS & CALIBRATION -->
    <div id="settings-modal" class="fixed inset-0 z-[2000] bg-black/75 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-4 max-w-sm w-full shadow-2xl flex flex-col gap-3">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <h3 class="font-bold text-sm text-cyan-400">⚙️ Settings & Calibration</h3>
                <button onclick="toggleSettingsModal()" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <div class="flex flex-col gap-1">
                <label class="text-xs text-slate-300 font-bold">🏢 Observer / Rooftop Elevation Offset</label>
                <div class="flex items-center gap-2">
                    <input id="building-offset" type="number" value="0" min="0" max="500" onchange="updateBuildingOffset(this.value)" class="w-full bg-slate-950 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold">
                    <span class="text-slate-400 text-xs">m</span>
                </div>
            </div>

            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-xs text-slate-300 font-bold">
                    <span>⏱️ Telemetry Latency Calibration</span>
                    <span id="calib-val" class="text-amber-400 font-bold">0.0s</span>
                </div>
                <input id="calib-slider" type="range" min="-5.0" max="5.0" step="0.1" value="0.0" oninput="updateCalibration(this.value)" class="w-full h-2 bg-slate-950 rounded-lg cursor-pointer accent-amber-400">
            </div>

            <div class="flex justify-between items-center pt-2 border-t border-slate-800">
                <button onclick="toggleMapLayer()" id="layer-btn" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 font-bold">
                    🗺️ Toggle Sat Map
                </button>
                <button onclick="toggleSettingsModal()" class="bg-cyan-600 hover:bg-cyan-500 text-xs text-white px-4 py-1.5 rounded-lg font-bold">
                    Save
                </button>
            </div>
        </div>
    </div>

    <script>
        let savedLat = localStorage.getItem('obs_lat');
        let savedLon = localStorage.getItem('obs_lon');
        let savedBuilding = localStorage.getItem('obs_building_m');
        let savedCalib = localStorage.getItem('obs_calib');

        let observerLat = savedLat ? parseFloat(savedLat) : 41.6079;
        let observerLon = savedLon ? parseFloat(savedLon) : 2.2876;
        let terrainElevationM = 145.0;
        let buildingOffsetM = savedBuilding ? parseFloat(savedBuilding) : 0.0;
        let timingCalibrationSec = savedCalib ? parseFloat(savedCalib) : 0.0;

        let activeFilter = 'moon';
        let isLocationLocked = true;
        let serverClockDelta = 0.0;
        let audioEnabled = false;
        let voiceEnabled = false;
        let audioContext = null;
        let lastBeepedFlight = 0;
        
        // Registro de avisos y descubrimientos tempranos
        let detectedTransits = new Set();
        let spokenCountdowns = new Set();
        let lastAnimTime = 0;

        let sunDataGlobal = { azimuth: 0, elevation: 0, visible: false };
        let moonDataGlobal = { azimuth: 0, elevation: 0, visible: false };

        let map, obsMarker;
        let sunLine = null, sunIconMarker = null;
        let moonLine = null, moonIconMarker = null;
        let currentBaseTileLayer, isSatelliteMode = false;
        
        const planesState = {};
        let activeAircraftData = [];

        const CARTO_KEY = 'cb1_2l65_1_3a8e83de8b889ec5e4e98278';

        map = L.map('map', { preferCanvas: true, zoomControl: false }).setView([observerLat, observerLon], 10);
        L.control.zoom({ position: 'bottomright' }).addTo(map);

        const cartoDarkLayer = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png?key=${CARTO_KEY}`, {
            subdomains: 'abcd', maxZoom: 20, attribution: '&copy; CARTO &bull; DE421'
        });

        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 18, attribution: 'Esri Satellite'
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
            renderAstroVectors();
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

        function toggleLocationLock() {
            isLocationLocked = !isLocationLocked;
            const btn = document.getElementById('lock-btn');
            if (isLocationLocked) {
                obsMarker.dragging.disable();
                btn.innerText = "🔒";
                btn.className = "text-[10px] px-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 font-bold transition";
            } else {
                obsMarker.dragging.enable();
                btn.innerText = "🔓";
                btn.className = "text-[10px] px-1 bg-amber-600 text-white rounded font-bold transition animate-pulse";
            }
        }

        function toggleAboutModal() {
            const m = document.getElementById('about-modal');
            m.classList.toggle('hidden');
            m.classList.toggle('flex');
        }

        function setFilterMode(mode) {
            activeFilter = mode;
            ['btn-flt-moon', 'btn-flt-sun', 'btn-flt-all'].forEach(id => {
                const btn = document.getElementById(id);
                btn.className = "text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-white";
            });

            const footerInd = document.getElementById('footer-vector-indicator');
            const footerInfo = document.getElementById('footer-astro-info');

            if (mode === 'moon') {
                document.getElementById('btn-flt-moon').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800";
                document.getElementById('filter-indicator').innerText = "TARGET: 🌕 MOON";
                footerInd.innerHTML = `<span class="text-cyan-400 font-bold">🌕──────</span>`;
                footerInfo.innerText = "Optical Line of Sight to the Moon";
            } else if (mode === 'sun') {
                document.getElementById('btn-flt-sun').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-amber-950 text-amber-300 border border-amber-800";
                document.getElementById('filter-indicator').innerText = "TARGET: ☀️ SUN";
                footerInd.innerHTML = `<span class="text-amber-400 font-bold">☀️──────</span>`;
                footerInfo.innerText = "Optical Line of Sight to the Sun";
            } else if (mode === 'all') {
                document.getElementById('btn-flt-all').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-slate-800 text-cyan-300";
                document.getElementById('filter-indicator').innerText = "TARGET: DUAL (SUN & MOON)";
                footerInd.innerHTML = `<span class="text-cyan-400 font-bold">🌕──</span> <span class="text-amber-400 font-bold">☀️──</span>`;
                footerInfo.innerText = "Dual Optical Sight Lines Active";
            }
            
            renderAstroVectors();
            updateHUDCountdowns();
        }

        function toggleVoice() {
            voiceEnabled = !voiceEnabled;
            const btn = document.getElementById('voice-btn');
            if (voiceEnabled) {
                btn.className = "bg-purple-600 text-white text-xs px-2 py-1.5 rounded-lg font-bold transition";
                speak("Lunar voice alerts active");
            } else {
                btn.className = "bg-slate-800 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition";
            }
        }

        function speak(text) {
            if (!voiceEnabled || !('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(text);
            msg.rate = 1.05; msg.pitch = 1.0;
            window.speechSynthesis.speak(msg);
        }

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('audio-btn');
            if (audioEnabled) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                btn.innerText = "🔔";
                btn.className = "bg-emerald-600 text-white text-xs px-2 py-1.5 rounded-lg font-bold transition";
                playChime();
            } else {
                btn.innerText = "🔇";
                btn.className = "bg-slate-800 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition";
            }
        }

        // 1. Repic melòdic suau per alertes de descobriment i fites (C5-E5-G5)
        function playChime() {
            if (!audioEnabled || !audioContext) return;
            try {
                const now = audioContext.currentTime;
                [523.25, 659.25, 783.99].forEach((freq, i) => {
                    const osc = audioContext.createOscillator();
                    const gain = audioContext.createGain();
                    osc.frequency.value = freq;
                    osc.type = 'sine';
                    gain.gain.setValueAtTime(0.12, now + i * 0.09);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.09 + 0.35);
                    osc.connect(gain);
                    gain.connect(audioContext.destination);
                    osc.start(now + i * 0.09);
                    osc.stop(now + i * 0.09 + 0.35);
                });
            } catch (e) {}
        }

        // 2. Beep curt per al compte enrere final (5, 4, 3, 2, 1s)
        function playTone(freq, duration) {
            if (!audioEnabled || !audioContext) return;
            try {
                const osc = audioContext.createOscillator();
                const gain = audioContext.createGain();
                osc.frequency.value = freq;
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.16, audioContext.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + duration);
                osc.connect(gain);
                gain.connect(audioContext.destination);
                osc.start();
                osc.stop(audioContext.currentTime + duration);
            } catch (e) {}
        }

        // 3. Acord harmònic de pas al moment exacte del trànsit (T=0)
        function playTransitChord() {
            if (!audioEnabled || !audioContext) return;
            try {
                const now = audioContext.currentTime;
                [880, 1108.73, 1318.51, 1760].forEach(freq => {
                    const osc = audioContext.createOscillator();
                    const gain = audioContext.createGain();
                    osc.frequency.value = freq;
                    osc.type = 'triangle';
                    gain.gain.setValueAtTime(0.14, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
                    osc.connect(gain);
                    gain.connect(audioContext.destination);
                    osc.start(now);
                    osc.stop(now + 0.65);
                });
            } catch (e) {}
        }

        function toggleSettingsModal() {
            const m = document.getElementById('settings-modal');
            m.classList.toggle('hidden'); m.classList.toggle('flex');
        }

        async function fetchTerrainElevation(lat, lon) {
            try {
                const res = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}`);
                const data = await res.json();
                if (data.elevation && data.elevation.length > 0) {
                    terrainElevationM = parseFloat(data.elevation[0]);
                    updateAltitudeDisplay();
                }
            } catch (e) {
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

        function updateCalibration(val) {
            timingCalibrationSec = parseFloat(val);
            localStorage.setItem('obs_calib', timingCalibrationSec.toString());
            document.getElementById('calib-val').innerText = (timingCalibrationSec >= 0 ? '+' : '') + timingCalibrationSec.toFixed(1) + 's';
        }

        async function saveAndSetObserverPos(lat, lon) {
            observerLat = lat; observerLon = lon;
            localStorage.setItem('obs_lat', lat.toString());
            localStorage.setItem('obs_lon', lon.toString());
            document.getElementById('obs-coords').innerText = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
            fetchTerrainElevation(lat, lon);
            renderAstroVectors();
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

        function getAltitudeColor(altFt) {
            if (altFt < 3000) return '#ef4444';
            if (altFt < 10000) return '#f97316';
            if (altFt < 18000) return '#eab308';
            if (altFt < 28000) return '#22c55e';
            if (altFt < 36000) return '#06b6d4';
            return '#a855f7';
        }

        function renderAstroVectors() {
            const distKm = 55;
            const showMoon = (activeFilter === 'all' || activeFilter === 'moon') && moonDataGlobal.visible;
            const showSun = (activeFilter === 'all' || activeFilter === 'sun') && sunDataGlobal.visible;

            // 1. Vector Lunar
            if (showMoon) {
                const radAzM = (moonDataGlobal.azimuth * Math.PI) / 180;
                const endM = [
                    observerLat + (distKm * Math.cos(radAzM)) / 111.0,
                    observerLon + (distKm * Math.sin(radAzM)) / (111.0 * Math.cos(observerLat * Math.PI / 180))
                ];
                if (moonLine) {
                    moonLine.setLatLngs([[observerLat, observerLon], endM]);
                } else {
                    moonLine = L.polyline([[observerLat, observerLon], endM], { color: '#38bdf8', weight: 2.5, dashArray: '6, 8', opacity: 0.95 }).addTo(map);
                }
                if (moonIconMarker) {
                    moonIconMarker.setLatLng(endM);
                } else {
                    moonIconMarker = L.marker(endM, { icon: L.divIcon({ className: 'astro-m', html: '<div class="text-xl">🌕</div>', iconSize: [24, 24], iconAnchor: [12, 12] }) }).addTo(map);
                }
            } else {
                if (moonLine) { map.removeLayer(moonLine); moonLine = null; }
                if (moonIconMarker) { map.removeLayer(moonIconMarker); moonIconMarker = null; }
            }

            // 2. Vector Solar
            if (showSun) {
                const radAzS = (sunDataGlobal.azimuth * Math.PI) / 180;
                const endS = [
                    observerLat + (distKm * Math.cos(radAzS)) / 111.0,
                    observerLon + (distKm * Math.sin(radAzS)) / (111.0 * Math.cos(observerLat * Math.PI / 180))
                ];
                if (sunLine) {
                    sunLine.setLatLngs([[observerLat, observerLon], endS]);
                } else {
                    sunLine = L.polyline([[observerLat, observerLon], endS], { color: '#f59e0b', weight: 2.5, dashArray: '6, 8', opacity: 0.95 }).addTo(map);
                }
                if (sunIconMarker) {
                    sunIconMarker.setLatLng(endS);
                } else {
                    sunIconMarker = L.marker(endS, { icon: L.divIcon({ className: 'astro-m', html: '<div class="text-xl">☀️</div>', iconSize: [24, 24], iconAnchor: [12, 12] }) }).addTo(map);
                }
            } else {
                if (sunLine) { map.removeLayer(sunLine); sunLine = null; }
                if (sunIconMarker) { map.removeLayer(sunIconMarker); sunIconMarker = null; }
            }
        }

        // =========================================================================
        // CARGA DE DATOS Y RENDERIZADO REACTIVO
        // =========================================================================
        async function fetchData() {
            try {
                const totalObserverAlt = terrainElevationM + buildingOffsetM;
                const res = await fetch(`/api/data?lat=${observerLat}&lon=${observerLon}&alt=${totalObserverAlt}`);
                const data = await res.json();
                
                if (data.server_time) {
                    serverClockDelta = (Date.now() / 1000.0) - data.server_time;
                }

                moonDataGlobal = data.moon;
                sunDataGlobal = data.sun;

                if (data.source_feed) {
                    document.getElementById('feed-badge').innerText = data.source_feed.toUpperCase();
                }

                document.getElementById('moon-coords').innerText = moonDataGlobal.visible ? `Moon: Az ${moonDataGlobal.azimuth}° | Alt +${moonDataGlobal.elevation}°` : `Moon Hidden (${moonDataGlobal.elevation}°)`;
                document.getElementById('moon-badge').innerText = `${moonDataGlobal.event_type}: ${moonDataGlobal.next_event_str}`;

                document.getElementById('sun-coords').innerText = sunDataGlobal.visible ? `Sun: Az ${sunDataGlobal.azimuth}° | Alt +${sunDataGlobal.elevation}°` : `Sun Hidden (${sunDataGlobal.elevation}°)`;
                document.getElementById('sun-badge').innerText = `${sunDataGlobal.event_type}: ${sunDataGlobal.next_event_str}`;

                renderAstroVectors();

                activeAircraftData = data.aircraft || [];
                document.getElementById('plane-count').innerText = activeAircraftData.length;
                const currentCallsigns = new Set();
                const nowSec = (Date.now() / 1000.0) - serverClockDelta;

                activeAircraftData.forEach(plane => {
                    const cs = plane.callsign;
                    currentCallsigns.add(cs);

                    const color = getAltitudeColor(plane.alt_ft);
                    const isAnyTransit = (plane.moon.is_transit && moonDataGlobal.visible) || (plane.sun.is_transit && sunDataGlobal.visible);
                    const isAnyClose = (plane.moon.is_close && moonDataGlobal.visible) || (plane.sun.is_close && sunDataGlobal.visible);

                    const glow = isAnyTransit ? 'filter: drop-shadow(0 0 12px #ef4444);' : (isAnyClose ? 'filter: drop-shadow(0 0 8px #f59e0b);' : '');

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
                                <b>SPAN:</b> ${plane.wingspan_m}m | <b>ANG SIZE:</b> ${plane.moon.angular_size_arcsec}"<br>
                                <b>HEADING:</b> ${plane.track}°
                            </div>
                        `);

                        planesState[cs] = {
                            marker: marker,
                            curLat: plane.lat, curLon: plane.lon,
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
                    if (nowSec - st.lastSeenTime > 15.0) {
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

                updateHUDCountdowns();

            } catch (err) {
                console.error("Error fetchData:", err);
            }
        }

        // 60 FPS Dead Reckoning Interpolation
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

        // TELEMETRY HUD & NOTIFICACIONS ACÚSTIQUES D'ALERTA PRIMERENCA
        function updateHUDCountdowns() {
            const container = document.getElementById('alerts-container');
            if (!activeAircraftData || activeAircraftData.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-500 text-center py-8">Tracking lunar sector traffic...</div>';
                return;
            }

            const now = (Date.now() / 1000.0) - serverClockDelta;
            
            let displayList = activeAircraftData.map(p => {
                let targetData = p.moon;
                if (activeFilter === 'sun') targetData = p.sun;
                if (activeFilter === 'all') targetData = p.primary;
                return { plane: p, target: targetData };
            });

            displayList.sort((a, b) => a.target.min_sep - b.target.min_sep);

            let html = '';
            displayList.forEach(({ plane, target }) => {
                const isTargetVisible = (target.target === 'sun' ? sunDataGlobal.visible : moonDataGlobal.visible);
                const remaining = Math.max(0.0, (target.tca_epoch - now) + timingCalibrationSec);
                const isTransit = target.is_transit && isTargetVisible;
                const isClose = target.is_close && isTargetVisible;
                const sym = target.target === 'sun' ? '☀️' : '🌕';
                const bodyLabel = (target.target === 'sun' ? 'Solar' : 'Lunar');

                const mins = Math.floor(remaining / 60);
                const secs = (remaining % 60).toFixed(1).padStart(4, '0');
                const timerStr = remaining > 0 ? `T-${mins.toString().padStart(2, '0')}:${secs}` : `TRANSITING!`;

                let cardBorder = 'border-slate-800 bg-slate-950/60';
                let tagHtml = `<span class="bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded text-[8px] font-bold">${sym} EN ROUTE</span>`;

                if (isTransit) {
                    cardBorder = 'border-red-500 bg-red-950/70 shadow-lg shadow-red-950/60';
                    tagHtml = `<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[9px] font-black animate-pulse">🎯 ${sym} TRANSIT (${target.transit_duration_s}s)!</span>`;
                    
                    const flightKey = `${plane.callsign}-${target.target}`;

                    // 1. ALERTA DE DESCOBRIMENT TEMPRÀ (Immediat en detectar-se a qualsevol distància/temps)
                    if (!detectedTransits.has(flightKey)) {
                        detectedTransits.add(flightKey);
                        playChime();
                        const mRemain = Math.floor(remaining / 60);
                        const sRemain = Math.floor(remaining % 60);
                        let timeAnnouncement = mRemain > 0 ? `${mRemain} minute${mRemain > 1 ? 's' : ''} and ${sRemain} seconds` : `${sRemain} seconds`;
                        speak(`Attention. New ${bodyLabel} transit detected for flight ${plane.callsign} in ${timeAnnouncement}`);
                    }

                    // 2. FITES D'AVÍS AMB TEMPS SUFICIENT DE PREPARACIÓ
                    const wholeSec = Math.floor(remaining);
                    const alertKey = `${flightKey}-${wholeSec}`;

                    if (!spokenCountdowns.has(alertKey) && remaining > 0) {
                        if (wholeSec === 180) { // 3 minuts
                            spokenCountdowns.add(alertKey);
                            playChime();
                            speak(`${bodyLabel} transit in 3 minutes`);
                        } else if (wholeSec === 120) { // 2 minuts
                            spokenCountdowns.add(alertKey);
                            playChime();
                            speak(`${bodyLabel} transit in 2 minutes, prepare camera`);
                        } else if (wholeSec === 60) { // 1 minut
                            spokenCountdowns.add(alertKey);
                            playChime();
                            speak(`Warning: ${bodyLabel} transit in 1 minute`);
                        } else if (wholeSec === 30) { // 30 segons
                            spokenCountdowns.add(alertKey);
                            playChime();
                            speak(`${bodyLabel} transit in 30 seconds`);
                        } else if (wholeSec === 15) { // 15 segons
                            spokenCountdowns.add(alertKey);
                            speak(`${bodyLabel} transit in 15 seconds`);
                        } else if (wholeSec === 10) { // 10 segons
                            spokenCountdowns.add(alertKey);
                            speak("10 seconds");
                        } else if ([5, 4, 3, 2, 1].includes(wholeSec)) { // Cadència final de disparador
                            spokenCountdowns.add(alertKey);
                            playTone(1100 + (5 - wholeSec) * 120, 0.12);
                        }
                    }

                    // 3. MOMENT EXACTE DEL CREUAMENT (T = 0s)
                    if (remaining <= 0.2 && lastBeepedFlight !== flightKey) {
                        playTransitChord();
                        lastBeepedFlight = flightKey;
                    }
                } else if (isClose) {
                    cardBorder = 'border-amber-500 bg-amber-950/60';
                    tagHtml = `<span class="bg-amber-600 text-white px-2 py-0.5 rounded text-[8px] font-bold">⚠️ ${sym} CLOSE PASS</span>`;
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
                            <span>Alt: <b>${plane.alt_ft.toLocaleString()} ft</b></span>
                            <span>V/S: <b>${plane.vr_fpm} ft/m</b></span>
                            <span>Dist: <b>${plane.distance_km} km</b></span>
                            <span>TCA: <b class="${isTransit ? 'text-red-400 font-black' : (isClose ? 'text-amber-300' : 'text-cyan-300')} font-mono">${timerStr}</b></span>
                            <span class="col-span-2">Angular Size: <b class="text-indigo-300">${target.angular_size_arcsec}" (${target.disk_coverage_pct}% disk)</b></span>
                            <span class="col-span-2">Vert Offset: <b class="${isTransit ? 'text-red-300' : 'text-slate-200'}">${target.vertical_offset_deg}° ${target.vertical_dir_text} (${target.position_descriptor})</b></span>
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
    print(f" [OK] LUNAR TRANSIT RADAR PRO - BY MARC GARRIDO")
    print(f" [OK] Early Warning Soundscape & Multi-Minute Voice Alerts Active")
    print(f" [OK] Verification: /google92a4c5b46b2ec0bf.html")
    print(f" [OK] Server Online on port: {port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

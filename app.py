import math
import threading
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

def compute_aircraft_refraction_deg(geom_alt_deg, slant_range_m, ac_alt_m, obs_alt_m, p_mbar, t_c):
    if geom_alt_deg < -0.5:
        return 0.0
    denom = math.tan(math.radians(max(0.05, geom_alt_deg + (10.3 / (geom_alt_deg + 5.11)))))
    if abs(denom) < 1e-6:
        return 0.0
    refr_astro_arcmin = (p_mbar / 1013.25) * (288.15 / (273.15 + t_c)) * (1.02 / denom)
    refr_astro_deg = refr_astro_arcmin / 60.0

    delta_h = max(0.0, ac_alt_m - obs_alt_m)
    density_factor = 1.0 - math.exp(-delta_h / 8400.0)
    range_factor = min(1.0, slant_range_m / (slant_range_m + 2500.0))
    return refr_astro_deg * density_factor * range_factor

def diff_angle_deg(a, b):
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
    alt = math.degrees(math.atan2(u, max(1e-3, ground)))
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

    sin_lat_future = math.sin(lat_r) * math.cos(d_r) + math.cos(lat_r) * math.sin(d_r) * math.cos(track_r)
    sin_lat_future = max(-1.0, min(1.0, sin_lat_future))
    lat_future_r = math.asin(sin_lat_future)

    lon_future_r = lon_r + math.atan2(
        math.sin(track_r) * math.sin(d_r) * math.cos(lat_r),
        math.cos(d_r) - math.sin(lat_r) * math.sin(lat_future_r)
    )
    return math.degrees(lat_future_r), math.degrees(lon_future_r)

# =========================================================================
# 2. INGESTA ADS-B CON TIEMPO EXACTO POR POSICIÓN
#    Cada posición lleva su instante real de medida (_pos_t = now_API - seen_pos).
#    La edad se calcula en el momento del cálculo, así que caché, latencia de red
#    y reutilización de snapshots quedan compensadas automáticamente.
# =========================================================================
UPSTREAMS = (
    ('airplanes.live', 'https://api.airplanes.live/v2/point/{lat:.4f}/{lon:.4f}/80'),
    ('adsb.lol', 'https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/80'),
)
HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) LunarTransitRadar/27.0'}
CELL_DEG = 0.1            # rejilla de caché (~11 km) compartida entre usuarios; el radio de 80 nm lo cubre de sobra
FRESH_TTL_S = 0.9         # un snapshot más joven que esto se reutiliza
MAX_STALE_S = 10.0        # un snapshot más viejo que esto NO se sirve (evita aviones fantasma)
MAX_POS_AGE_S = 15.0      # fixes más viejos que esto se descartan (ADS-C, MLAT perdido, etc.)
REFRESH_EVERY_S = 1.0     # cadencia del refresco en segundo plano (límite de airplanes.live: 1 req/s)
ACTIVE_WINDOW_S = 30.0

HTTP_SESSION = requests.Session()
CACHE = {}
LAST_REQ = {}
CELL_LOCKS = {}
_STATE_LOCK = threading.Lock()
_refresher_started = False

def _cell_key(lat, lon):
    return (round(lat / CELL_DEG), round(lon / CELL_DEG))

def _fetch_snapshot(clat, clon):
    empty = None
    for name, tpl in UPSTREAMS:
        try:
            t0 = time.time()
            r = HTTP_SESSION.get(tpl.format(lat=clat, lon=clon), headers=HTTP_HEADERS, timeout=(1.5, 2.5))
            t1 = time.time()
            if r.status_code != 200:
                continue
            data = r.json()
            ac = data.get('ac')
            if not isinstance(ac, list):
                continue
            try:
                api_now = float(data.get('now'))
                if api_now > 1e11:
                    api_now /= 1000.0
            except (TypeError, ValueError):
                api_now = None
            # Referencia temporal: reloj de la API si es coherente con el nuestro; si no, punto medio de la petición
            ref = api_now if (api_now and abs(api_now - t1) < 10.0) else (t0 + t1) / 2.0
            ac = [a for a in ac if isinstance(a, dict)]
            for a in ac:
                try:
                    sp = float(a.get('seen_pos', a.get('seen', 0.0)) or 0.0)
                except (TypeError, ValueError):
                    sp = 0.0
                a['_pos_t'] = ref - max(0.0, sp)
            snap = {'ac': ac, 'source': name, 'fetched': t1}
            if ac:
                return snap
            if empty is None:
                empty = snap
        except Exception:
            continue
    return empty

def _refresh_cell(key, block=True):
    lock = CELL_LOCKS.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=block):
        return CACHE.get(key)
    try:
        snap = CACHE.get(key)
        if snap and time.time() - snap['fetched'] < FRESH_TTL_S:
            return snap
        new = _fetch_snapshot(key[0] * CELL_DEG, key[1] * CELL_DEG)
        if new:
            CACHE[key] = new
            return new
        return snap
    finally:
        lock.release()

def _refresher_loop():
    while True:
        time.sleep(REFRESH_EVERY_S)
        now = time.time()
        for key, seen in list(LAST_REQ.items()):
            if now - seen > 120.0:
                LAST_REQ.pop(key, None)
                CACHE.pop(key, None)
            elif now - seen <= ACTIVE_WINDOW_S:
                try:
                    _refresh_cell(key)
                except Exception:
                    pass

def _ensure_refresher():
    global _refresher_started
    if _refresher_started:
        return
    with _STATE_LOCK:
        if not _refresher_started:
            threading.Thread(target=_refresher_loop, daemon=True, name='adsb-refresh').start()
            _refresher_started = True

def get_live_aircraft(cur_lat, cur_lon):
    _ensure_refresher()
    key = _cell_key(cur_lat, cur_lon)
    LAST_REQ[key] = time.time()
    snap = CACHE.get(key)
    if not snap or time.time() - snap['fetched'] >= FRESH_TTL_S:
        snap = _refresh_cell(key, block=(snap is None))
    if not snap:
        return [], 'offline', 0.0
    age = max(0.0, time.time() - snap['fetched'])
    if age > MAX_STALE_S:
        return [], snap['source'], age
    return snap['ac'], snap['source'], age

# =========================================================================
# 3. MOTOR ASTROMÉTRICO Y GEODÉSICO DE TRÁNSITOS
# =========================================================================
@app.route('/api/data')
def get_data():
    try:
        lat = float(request.args.get('lat', 41.6079))
        lon = float(request.args.get('lon', 2.2876))
        alt = float(request.args.get('alt', 145.0))
        user_lead_sec = max(0.0, min(15.0, float(request.args.get('lead', 6.0))))
        raw_ac, source_feed, _ = get_live_aircraft(lat, lon)
        now_epoch = time.time()  # instante único de referencia (aviones, astros y tca_epoch)
        t_now = ts.from_datetime(datetime.fromtimestamp(now_epoch, timezone.utc))
        topos_loc = wgs84.latlon(lat, lon, elevation_m=alt)
        obs_loc = earth + topos_loc
        p_mbar, t_c = calculate_atmosphere(alt)

        # 1. Astrometría Lunar
        app_moon = obs_loc.at(t_now).observe(moon).apparent()
        m_alt, m_az, m_dist = app_moon.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
        moon_az0 = float(m_az.degrees)
        moon_alt0 = float(m_alt.degrees)
        moon_dist_km = max(1000.0, float(m_dist.km))
        moon_radius_deg = float(math.degrees(math.asin(min(1.0, MOON_RADIUS_KM / moon_dist_km))))
        moon_is_visible = bool(moon_alt0 > -0.5)

        # 2. Astrometría Solar
        app_sun = obs_loc.at(t_now).observe(sun).apparent()
        s_alt, s_az, s_dist = app_sun.altaz(pressure_mbar=p_mbar, temperature_C=t_c)
        sun_az0 = float(s_az.degrees)
        sun_alt0 = float(s_alt.degrees)
        sun_dist_km = max(1000.0, float(s_dist.km))
        sun_radius_deg = float(math.degrees(math.asin(min(1.0, SUN_RADIUS_KM / sun_dist_km))))
        sun_is_visible = bool(sun_alt0 > -0.5)

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
            track = ac.get('track')
            gs = ac.get('gs', 0)
            vr_raw = ac.get('geom_rate', ac.get('baro_rate', 0))
            model_icao = str(ac.get('t', 'A320')).strip().upper()
            wingspan_m = get_wingspan(model_icao)

            alt_geom = ac.get('alt_geom')
            alt_baro = ac.get('alt_baro')
            if alt_geom is not None and alt_geom != 'ground':
                alt_ft = float(alt_geom)
                alt_type = 'GNSS'
            elif alt_baro is not None and alt_baro != 'ground':
                alt_ft = float(alt_baro)
                alt_type = 'BARO'
            else:
                continue

            if None in (raw_lat, raw_lon, track) or gs is None or float(gs) <= 15:
                continue

            try:
                alt_m = alt_ft * 0.3048
                speed_ms = float(gs) * 0.514444
                vr_ms = float(vr_raw) * 0.00508 if vr_raw else 0.0
                vr_fpm = int(float(vr_raw)) if vr_raw else 0
                track_val = float(track)
                ac_lat = float(raw_lat)
                ac_lon = float(raw_lon)
            except (ValueError, TypeError):
                continue

            # Extrapolación de latencia calibrable
            pos_age = max(0.0, now_epoch - ac.get('_pos_t', now_epoch))
            if pos_age > MAX_POS_AGE_S:
                continue
            total_lead = pos_age + user_lead_sec

            if total_lead > 0.05:
                ac_lat, ac_lon = propagate_geodetic_position(ac_lat, ac_lon, speed_ms, track_val, total_lead)
                alt_m += vr_ms * total_lead

            e0, n0, u0 = ecef_to_enu(*geodetic_to_ecef(ac_lat, ac_lon, alt_m), lat, lon, alt)
            cur_az, cur_geom_alt, cur_range = enu_to_az_alt(e0, n0, u0)
            
            refr_ac_now = compute_aircraft_refraction_deg(cur_geom_alt, cur_range, alt_m, alt, p_mbar, t_c)
            cur_app_alt = cur_geom_alt + refr_ac_now

            track_rad = math.radians(track_val)
            ve = speed_ms * math.sin(track_rad)
            vn = speed_ms * math.cos(track_rad)
            vu = vr_ms

            def compute_body_intercept(body_type, b_az0, b_alt0, b_rad, d_az, d_alt, is_vis):
                if not is_vis:
                    return {
                        'target': body_type, 'is_transit': False, 'is_close': False,
                        'min_sep': 99.0, 'current_sep': 99.0, 'tca_seconds': 0.0,
                        'tca_epoch': float(now_epoch), 'vertical_offset_deg': 0.0,
                        'vertical_offset_m': 0, 'vertical_body_diams': 0.0,
                        'vertical_dir_text': '', 'transit_duration_s': 0.0,
                        'angular_size_arcsec': 0.0, 'disk_coverage_pct': 0.0,
                        'position_descriptor': "Below Horizon",
                        'tca_lat': ac_lat, 'tca_lon': ac_lon
                    }

                cur_sep = angular_separation(cur_az, cur_app_alt, b_az0, b_alt0)

                def eval_t(t_val):
                    p_lat, p_lon = propagate_geodetic_position(ac_lat, ac_lon, speed_ms, track_val, t_val)
                    p_alt_m = alt_m + (vr_ms * t_val)
                    et, nt, ut = ecef_to_enu(*geodetic_to_ecef(p_lat, p_lon, p_alt_m), lat, lon, alt)
                    p_az, p_geom_alt, p_range = enu_to_az_alt(et, nt, ut)
                    
                    if p_geom_alt <= -0.5:
                        return 999.0, p_az, p_geom_alt, p_range, p_lat, p_lon

                    refr_ac = compute_aircraft_refraction_deg(p_geom_alt, p_range, p_alt_m, alt, p_mbar, t_c)
                    p_app_alt = p_geom_alt + refr_ac

                    b_az_t = (b_az0 + d_az * t_val) % 360.0
                    b_alt_t = b_alt0 + d_alt * t_val
                    sep = angular_separation(p_az, p_app_alt, b_az_t, b_alt_t)
                    return sep, p_az, p_app_alt, p_range, p_lat, p_lon

                b_rad_az = math.radians(b_az0)
                b_rad_alt = math.radians(b_alt0)
                bx = math.cos(b_rad_alt) * math.sin(b_rad_az)
                by = math.cos(b_rad_alt) * math.cos(b_rad_az)
                bz = math.sin(b_rad_alt)

                v_dot_b = ve * bx + vn * by + vu * bz
                r_dot_b = e0 * bx + n0 * by + u0 * bz
                r_dot_v = e0 * ve + n0 * vn + u0 * vu
                v_mag_sq = ve**2 + vn**2 + vu**2
                denom = v_mag_sq - (v_dot_b ** 2)

                t_analytical = 0.0
                if denom > 1e-4:
                    t_analytical = (r_dot_b * v_dot_b - r_dot_v) / denom

                if t_analytical < -3.0 and cur_sep > (b_rad * 4.0):
                    return {
                        'target': body_type, 'is_transit': False, 'is_close': False,
                        'min_sep': round(cur_sep, 3), 'current_sep': round(cur_sep, 2),
                        'tca_seconds': 0.0, 'tca_epoch': float(now_epoch),
                        'vertical_offset_deg': 0.0, 'vertical_offset_m': 0,
                        'vertical_body_diams': 0.0, 'vertical_dir_text': 'Departing',
                        'transit_duration_s': 0.0, 'angular_size_arcsec': 0.0,
                        'disk_coverage_pct': 0.0, 'position_descriptor': "Diverging",
                        'tca_lat': ac_lat, 'tca_lon': ac_lon
                    }

                center_t = max(0.0, min(300.0, t_analytical if t_analytical > 0 else 0.0))
                scan_min = max(0.0, center_t - 7.5)
                scan_max = min(300.0, center_t + 7.5)
                
                best_t = center_t
                min_sep, _, best_p_alt, best_p_range, best_lat, best_lon = eval_t(best_t)

                for step in range(16):
                    t_cand = scan_min + (step / 15.0) * (scan_max - scan_min)
                    sep_val, _, p_alt_val, p_range_val, cand_lat, cand_lon = eval_t(t_cand)
                    if sep_val < min_sep:
                        min_sep = sep_val
                        best_t = t_cand
                        best_p_alt = p_alt_val
                        best_p_range = p_range_val
                        best_lat, best_lon = cand_lat, cand_lon

                a = max(0.0, best_t - 1.2)
                b = min(300.0, best_t + 1.2)
                phi = (1.0 + math.sqrt(5.0)) / 2.0
                resphi = 2.0 - phi

                x1 = a + resphi * (b - a)
                x2 = b - resphi * (b - a)
                f1, _, _, _, _, _ = eval_t(x1)
                f2, _, _, _, _, _ = eval_t(x2)

                for _ in range(12):
                    if f1 < f2:
                        b = x2; x2 = x1; f2 = f1
                        x1 = a + resphi * (b - a)
                        f1, _, _, _, _, _ = eval_t(x1)
                    else:
                        a = x1; x1 = x2; f1 = f2
                        x2 = b - resphi * (b - a)
                        f2, _, _, _, _, _ = eval_t(x2)

                t_opt = (a + b) / 2.0
                sep_opt, _, opt_p_alt, opt_p_range, opt_lat, opt_lon = eval_t(t_opt)
                if sep_opt < min_sep:
                    min_sep = sep_opt
                    best_t = t_opt
                    best_p_alt = opt_p_alt
                    best_p_range = opt_p_range
                    best_lat, best_lon = opt_lat, opt_lon

                best_b_alt = b_alt0 + d_alt * best_t
                vert_offset_deg = round(best_p_alt - best_b_alt, 3)
                body_diam_deg = 2.0 * b_rad
                vert_body_diams = round(abs(vert_offset_deg) / max(0.01, body_diam_deg), 1)
                vert_offset_m = int(best_p_range * math.tan(math.radians(vert_offset_deg)))

                is_transit = (min_sep <= b_rad)
                is_close = (min_sep > b_rad and min_sep <= (b_rad * 3.5))

                transit_dur = 0.0
                if is_transit and best_t > 0:
                    chord_deg = 2.0 * math.sqrt(max(0.0, (b_rad ** 2) - (min_sep ** 2)))
                    dt_d = 0.4
                    sep_p, _, _, _, _, _ = eval_t(max(0.0, best_t - dt_d))
                    sep_n, _, _, _, _, _ = eval_t(best_t + dt_d)
                    ang_speed = max(0.08, math.hypot((sep_n - sep_p) / (2 * dt_d), (speed_ms / max(100.0, best_p_range)) * (180.0 / math.pi)))
                    transit_dur = round(chord_deg / ang_speed, 2)

                ang_size_rad = 2.0 * math.atan2(wingspan_m, 2.0 * max(100.0, best_p_range))
                ang_size_arcsec = round(math.degrees(ang_size_rad) * 3600.0, 1)
                disk_coverage_pct = round((ang_size_arcsec / max(1.0, (body_diam_deg * 3600.0))) * 100.0, 1)

                symbol = "🌕" if body_type == 'moon' else "☀️"
                name = "Lunar" if body_type == 'moon' else "Solar"

                if is_transit:
                    if abs(vert_offset_deg) <= (b_rad * 0.28):
                        pos_desc = f"{name} Center"
                    elif vert_offset_deg > 0:
                        pos_desc = "Upper Limb (Above)"
                    else:
                        pos_desc = "Lower Limb (Below)"
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
                    'tca_seconds': round(best_t, 2),
                    'tca_epoch': float(now_epoch + best_t) if best_t > 0 else float(now_epoch),
                    'vertical_offset_deg': vert_offset_deg,
                    'vertical_offset_m': vert_offset_m,
                    'vertical_body_diams': vert_body_diams,
                    'vertical_dir_text': dir_clean,
                    'transit_duration_s': transit_dur,
                    'angular_size_arcsec': ang_size_arcsec,
                    'disk_coverage_pct': disk_coverage_pct,
                    'position_descriptor': pos_desc,
                    'tca_lat': round(best_lat, 5),
                    'tca_lon': round(best_lon, 5)
                }

            moon_data = compute_body_intercept('moon', moon_az0, moon_alt0, moon_radius_deg, d_az_dt_moon, d_alt_dt_moon, moon_is_visible)
            sun_data = compute_body_intercept('sun', sun_az0, sun_alt0, sun_radius_deg, d_az_dt_sun, d_alt_dt_sun, sun_is_visible)

            primary = moon_data if moon_is_visible else (sun_data if sun_is_visible else moon_data)

            aircraft_results.append({
                'callsign': str(ac.get('flight') or ac.get('hex', 'UNKNOWN')).strip(),
                'model': model_icao,
                'wingspan_m': wingspan_m,
                'reg': str(ac.get('r', '')).strip().upper(),
                'lat': round(ac_lat, 5),
                'lon': round(ac_lon, 5),
                'alt_ft': int(alt_ft),
                'alt_type': alt_type,
                'track': float(round(track_val, 1)),
                'speed_kt': int(float(gs)),
                'speed_ms': float(round(speed_ms, 1)),
                'vr_fpm': vr_fpm,
                'azimuth': float(round(cur_az, 1)),
                'elevation': float(round(cur_app_alt, 1)),
                'distance_km': float(round(cur_range / 1000.0, 1)),
                'moon': moon_data,
                'sun': sun_data,
                'primary': primary
            })

        return jsonify({
            'source_feed': source_feed,
            'server_time': now_epoch,
            'lead_used_sec': round(user_lead_sec, 1),
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
        return jsonify({'error': str(e), 'aircraft': []})

# =========================================================================
# 4. RUTAS DE INDEXACIÓN Y PLANTILLA HTML
# =========================================================================
@app.route('/api/time')
def api_time():
    return jsonify({'t': time.time()})

@app.after_request
def _no_store(resp):
    if request.path.startswith('/api/'):
        resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/google92a4c5b46b2ec0bf.html')
def google_verification():
    return 'google-site-verification: google92a4c5b46b2ec0bf.html'

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# =========================================================================
# 5. FRONTEND: CABINA TÁCTICA, TARJETAS CELESTES Y HARVARD TERMINAL
# =========================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="google-site-verification" content="google92a4c5b46b2ec0bf" />
    <title>Lunar Transit Radar PRO</title>
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { 
            background-color: #020617; 
            color: #f8fafc; 
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-feature-settings: "tnum" 1;
        }
        
        .map-container { height: calc(100dvh - 114px); width: 100%; border-radius: 12px; }
        .leaflet-container { background: #020617 !important; }
        
        .leaflet-div-icon {
            background: transparent !important;
            border: none !important;
        }

        .obs-target { 
            position: relative;
            display: flex; 
            align-items: center; 
            justify-content: center; 
            width: 44px; 
            height: 44px; 
            pointer-events: none;
        }
        .obs-sweep {
            position: absolute;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: conic-gradient(from 0deg, rgba(6, 182, 212, 0.4) 0deg, transparent 60deg, transparent 360deg);
            animation: radar-sweep 3s linear infinite;
            will-change: transform;
        }
        .obs-ring { 
            position: absolute; 
            width: 32px; 
            height: 32px; 
            border-radius: 50%; 
            background: rgba(6, 182, 212, 0.15); 
            border: 1.5px solid #06b6d4; 
            animation: pulse-ring 2.2s infinite ease-out; 
        }
        .obs-dot { 
            width: 8px; 
            height: 8px; 
            border-radius: 50%; 
            background: #22d3ee; 
            border: 2px solid #ffffff; 
            box-shadow: 0 0 12px #06b6d4; 
            z-index: 10; 
        }
        @keyframes pulse-ring { 0% { transform: scale(0.6); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }
        @keyframes radar-sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .tca-target { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; }
        .tca-ring { position: absolute; width: 28px; height: 28px; border-radius: 50%; border: 2px dashed #ef4444; animation: tca-spin 3s linear infinite; will-change: transform; }
        .tca-crosshair-h { position: absolute; width: 28px; height: 1.5px; background: rgba(239, 68, 68, 0.8); }
        .tca-crosshair-v { position: absolute; width: 1.5px; height: 28px; background: rgba(239, 68, 68, 0.8); }
        .tca-core { width: 6px; height: 6px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 12px #ef4444; z-index: 10; }
        @keyframes tca-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .avionics-panel { 
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(3, 7, 18, 0.98) 100%);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(51, 65, 85, 0.7);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
        }

        .hud-card { 
            position: relative;
            background: rgba(11, 19, 38, 0.85);
            border: 1px solid rgba(30, 41, 59, 0.9);
            backdrop-filter: blur(8px);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            contain: layout style;
        }
        .hud-card::before {
            content: '';
            position: absolute;
            top: -1px; left: -1px;
            width: 6px; height: 6px;
            border-top: 2px solid #06b6d4;
            border-left: 2px solid #06b6d4;
            pointer-events: none;
        }
        .hud-card::after {
            content: '';
            position: absolute;
            bottom: -1px; right: -1px;
            width: 6px; height: 6px;
            border-bottom: 2px solid #06b6d4;
            border-right: 2px solid #06b6d4;
            pointer-events: none;
        }
        .hud-card:hover { 
            transform: translateY(-1px); 
            border-color: rgba(6, 182, 212, 0.75);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        }

        .glow-cyan { box-shadow: 0 0 18px rgba(6, 182, 212, 0.4); }
        .glow-amber { box-shadow: 0 0 18px rgba(245, 158, 11, 0.4); }
        .glow-red { box-shadow: 0 0 24px rgba(239, 68, 68, 0.45); }

        .terminal-screen {
            background-color: #010409;
            background-image: radial-gradient(rgba(16, 185, 129, 0.12) 1px, transparent 0);
            background-size: 20px 20px;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #020617; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }

        /* ===== TEMA PROFESIONAL v27 ===== */
        :root{--bg:#070b12;--s1:#0c121d;--s2:#111a29;--line:rgba(148,163,184,.14);--line2:rgba(148,163,184,.28);--tx:#e6edf7;--mut:#8696ae;--moon:#7dd3fc;--sun:#fbbf24;--bad:#f87171;--ok:#34d399;--acc:#38bdf8}
        html,body{background:var(--bg)}
        body{color:var(--tx);font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased}
        .font-mono,.tabular-nums{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
        [class*="text-[7.5px]"],[class*="text-[8px]"],[class*="text-[9px]"]{font-size:10px!important}
        [class*="text-[9.5px]"],[class*="text-[10px]"]{font-size:11px!important}
        .ic{width:16px;height:16px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
        .topbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;padding:8px 12px;margin-bottom:6px;background:var(--s1);border:1px solid var(--line);border-radius:12px}
        .brand{display:flex;align-items:center;gap:10px}
        .brand .brand-mark{width:28px;height:28px;color:var(--moon);stroke-width:1.5}
        .brand h1{font-size:13px;font-weight:650;letter-spacing:.06em;text-transform:uppercase;line-height:1.2}
        .chips{display:flex;gap:4px;margin-top:4px}
        .chip{font:600 10px/1 'JetBrains Mono',monospace;padding:3px 6px;border-radius:5px;border:1px solid var(--line2);color:var(--mut);background:rgba(255,255,255,.02);letter-spacing:.04em;white-space:nowrap}
        #feed-badge{color:var(--ok);border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.08)}
        #feed-badge[data-state="bad"]{color:var(--bad);border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.08)}
        .bodies{display:flex;gap:6px}
        .body-card{display:flex;align-items:center;gap:9px;padding:6px 10px;border-radius:10px;border:1px solid var(--line);background:var(--s2);cursor:pointer;opacity:.55;transition:.15s}
        .body-card:hover{opacity:1}
        .body-card.on{opacity:1}
        #moon-status-card.on{border-color:rgba(125,211,252,.45)}
        #sun-status-card.on{border-color:rgba(251,191,36,.45)}
        .body-txt{display:flex;flex-direction:column;gap:2px}
        .body-txt span{font:600 12px/1.2 'JetBrains Mono',monospace}
        .body-txt small{font-size:10px;color:var(--mut);letter-spacing:.08em;text-transform:uppercase}
        #moon-coords{color:var(--moon)} #sun-coords{color:var(--sun)}
        .controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
        .stepper,.seg{display:flex;align-items:center;gap:2px;padding:3px;border:1px solid var(--line);border-radius:9px;background:var(--s2)}
        .stepper .lbl{font-size:10px;font-weight:600;color:var(--mut);letter-spacing:.1em;padding:0 6px}
        .stepper button,.seg button{min-width:26px;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;color:var(--mut);transition:.12s}
        .stepper button:hover,.seg button:hover{color:var(--tx);background:rgba(255,255,255,.05)}
        .seg button.on{color:var(--tx);background:rgba(56,189,248,.16);box-shadow:inset 0 0 0 1px rgba(56,189,248,.4)}
        #lead-display{min-width:54px;text-align:center;font:600 12px 'JetBrains Mono',monospace;color:var(--sun)}
        .coords{font:500 11px 'JetBrains Mono',monospace;color:var(--mut)}
        .icon-btn{display:grid;place-items:center;width:32px;height:32px;border-radius:9px;border:1px solid var(--line);background:var(--s2);color:var(--mut);transition:.12s}
        .icon-btn:hover{color:var(--tx);border-color:var(--line2)}
        .icon-btn.on{color:var(--acc);border-color:rgba(56,189,248,.45);background:rgba(56,189,248,.1)}
        .map-container{flex:1 1 auto;min-height:0;height:auto;width:100%}
        .leaflet-container{background:#05080e!important;font-family:inherit}
        .leaflet-bar a{background:var(--s1)!important;color:var(--tx)!important;border-color:var(--line)!important}
        .leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--s1);color:var(--tx);border:1px solid var(--line2);box-shadow:0 8px 24px rgba(0,0,0,.5)}
        .leaflet-popup-content{margin:10px 12px;font:12px/1.55 'JetBrains Mono',monospace}
        .leaflet-control-attribution{background:rgba(7,11,18,.7)!important;color:var(--mut)!important}
        .avionics-panel{background:var(--s1);border:1px solid var(--line);box-shadow:none;backdrop-filter:none}
        .hud-card{background:var(--s2);border:1px solid var(--line);border-radius:10px;backdrop-filter:none;contain:layout style}
        .hud-card::before,.hud-card::after{display:none}
        .hud-card:hover{transform:none;box-shadow:none;border-color:var(--line2)}
        .glow-red{box-shadow:0 0 0 1px rgba(248,113,113,.55),0 0 22px rgba(248,113,113,.18)!important}
        .glow-amber{box-shadow:0 0 0 1px rgba(245,158,11,.5),0 0 18px rgba(245,158,11,.14)!important}
        .glow-cyan{box-shadow:none}
        #settings-modal>div{background:var(--s1)!important;border-color:var(--line2)!important;border-radius:14px!important}
    </style>
</head>
<body class="p-1.5 md:p-2 flex flex-col h-[100dvh] overflow-hidden select-none">
    
    <header class="topbar">
        <div class="brand">
            <svg class="ic brand-mark" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M2.5 15.5 21.5 8.5"/></svg>
            <div>
                <h1>Lunar Transit Radar</h1>
                <div class="chips"><span id="feed-badge" class="chip">ONLINE</span><span class="chip">DE421 JPL</span><span id="sync-badge" class="chip">SYNC …</span></div>
            </div>
        </div>
        <div class="bodies">
            <div id="moon-status-card" class="body-card on" onclick="setFilterMode('moon')">
                <div id="header-moon-icon" class="w-5 h-5 flex items-center justify-center"></div>
                <div class="body-txt"><span id="moon-coords">Moon: Az --° | Alt --°</span><small>Moon · optical vector</small></div>
                <span id="moon-badge" class="chip">--:--</span>
            </div>
            <div id="sun-status-card" class="body-card" onclick="setFilterMode('sun')">
                <div id="header-sun-icon" class="w-5 h-5 flex items-center justify-center"></div>
                <div class="body-txt"><span id="sun-coords">Sun: Az --° | Alt --°</span><small>Sun · solar sight line</small></div>
                <span id="sun-badge" class="chip">--:--</span>
            </div>
        </div>
        <div class="controls">
            <div class="stepper"><span class="lbl">LEAD</span><button onclick="adjustLead(-0.5)">−</button><span id="lead-display">+6.0s</span><button onclick="adjustLead(0.5)">+</button></div>
            <div class="seg">
                <button id="btn-flt-moon" class="on" onclick="setFilterMode('moon')">Moon</button>
                <button id="btn-flt-sun" onclick="setFilterMode('sun')">Sun</button>
                <button id="btn-flt-all" onclick="setFilterMode('all')">Dual</button>
            </div>
            <div class="hidden md:flex flex-col items-end leading-tight"><span id="obs-coords" class="coords">41.6079, 2.2876</span><span id="obs-alt-badge" class="coords">145 m ASL</span></div>
            <button id="voice-btn" class="icon-btn on" onclick="toggleVoice()" title="Voz"><svg class="ic" viewBox="0 0 24 24"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg></button>
            <button id="audio-btn" class="icon-btn on" onclick="toggleAudio()" title="Sonido"><svg class="ic" viewBox="0 0 24 24"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a2 2 0 0 0 3.4 0"/></svg></button>
            <button class="icon-btn" onclick="toggleSettingsModal()" title="Ajustes"><svg class="ic" viewBox="0 0 24 24"><path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/></svg></button>
            <button class="icon-btn" onclick="locateUser()" title="Mi ubicación"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/><path d="M12 1v3M12 20v3M1 12h3M20 12h3"/></svg></button>
        </div>
    </header>

    <!-- CUERPO PRINCIPAL -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-1.5 flex-grow overflow-hidden relative">
        
        <!-- VISTA 1: RADAR MAP -->
        <div id="radar-viewport" class="lg:col-span-3 rounded-xl overflow-hidden border border-slate-800 relative shadow-2xl flex flex-col">
            <div id="map" class="map-container"></div>
            
            <div class="hidden sm:flex absolute top-2.5 right-2.5 z-[1000] bg-slate-950/90 backdrop-blur p-2 rounded-xl text-[9px] border border-slate-800/80 flex-col gap-1 shadow-2xl">
                <span class="font-bold text-slate-400 uppercase text-[8px] mb-0.5 tracking-wider">Flight Level (FL)</span>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#ef4444]"></span> &lt; 3k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#f97316]"></span> 3k - 10k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#eab308]"></span> 10k - 18k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#22c55e]"></span> 18k - 28k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#06b6d4]"></span> 28k - 36k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#a855f7]"></span> &gt; 36k ft</div>
            </div>

            <div class="absolute bottom-2.5 left-2.5 z-[1000] bg-slate-950/90 backdrop-blur px-2.5 py-1.5 rounded-lg text-[11px] border border-slate-800 text-slate-300 flex items-center gap-2">
                <span id="footer-vector-indicator" class="font-bold text-cyan-400">━━━━</span>
                <span id="footer-astro-info">Optical Sight Vector Active</span>
            </div>
        </div>

        <!-- TELEMETRÍA LATERAL HUD (TARJETAS POD MODULARES) -->
        <div class="avionics-panel rounded-xl p-2.5 overflow-y-auto flex flex-col gap-2 shadow-2xl max-h-[42vh] lg:max-h-full">
            <div class="flex justify-between items-center border-b border-slate-800 pb-1.5">
                <h2 class="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <span>Sector telemetry</span>
                    <span id="plane-count" class="bg-cyan-950 text-cyan-300 px-1.5 py-0.2 rounded-full text-[9px] border border-cyan-800 font-mono tabular-nums">0</span>
                </h2>
                <span id="filter-indicator" class="text-[9px] text-cyan-400 font-mono font-bold tracking-wider">TARGET · MOON</span>
            </div>
            
            <div id="alerts-container" class="flex flex-col gap-2 overflow-y-auto">
                <div class="text-xs text-slate-500 text-center py-8 font-mono">Scanning airspace for transit intercept...</div>
            </div>
        </div>
    </div>

    <!-- MODAL AJUSTES Y CALIBRACIÓN AVANZADA DE ALTURA -->
    <div id="settings-modal" class="fixed inset-0 z-[2000] bg-black/75 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-4 max-w-sm w-full shadow-2xl flex flex-col gap-3 font-mono">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <h3 class="font-bold text-sm text-cyan-400">⚙️ Configuración & Calibración</h3>
                <button onclick="toggleSettingsModal()" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <!-- GESTIÓN ORÓGRAFICA / ELEVACIÓN DEL OBSERVADOR -->
            <div class="flex flex-col gap-2 bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-300 font-bold uppercase tracking-wider">⛰️ Cota del Terreno (MSL)</span>
                    <button onclick="detectTerrainAuto()" class="text-[9px] px-1.5 py-0.5 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 rounded border border-cyan-800">
                        Auto-Detect
                    </button>
                </div>
                <div class="flex items-center gap-2">
                    <input id="terrain-elevation-input" type="number" step="1" oninput="calculateModalTotalAlt()" class="w-full bg-slate-900 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold tabular-nums">
                    <span class="text-slate-400 text-xs">m</span>
                </div>

                <div class="flex justify-between items-center mt-1">
                    <span class="text-[10px] text-slate-300 font-bold uppercase tracking-wider">🏢 Azotea / Trípode (Offset)</span>
                </div>
                <div class="flex items-center gap-2">
                    <input id="building-offset" type="number" step="0.5" min="0" max="500" oninput="calculateModalTotalAlt()" class="w-full bg-slate-900 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold tabular-nums">
                    <span class="text-slate-400 text-xs">m</span>
                </div>

                <div class="flex justify-between items-center pt-2 border-t border-slate-800/80 text-xs">
                    <span class="text-slate-400 font-bold">Cota Total Efectiva:</span>
                    <span id="total-alt-display" class="text-emerald-400 font-black tabular-nums">145m</span>
                </div>
            </div>

            <!-- PRUEBA DE AUDIO -->
            <div class="flex justify-between items-center bg-slate-950 p-2 rounded-xl border border-slate-800">
                <span class="text-[10px] text-slate-300 font-bold">🔊 Sistema Acústico</span>
                <button onclick="testAudioSystem()" class="px-2 py-1 bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 rounded text-[10px] font-bold">
                    🔔 Test Audio
                </button>
            </div>

            <div class="flex justify-between items-center pt-2 border-t border-slate-800">
                <button onclick="toggleMapLayer()" id="layer-btn" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 font-bold transition">
                    🗺️ Toggle Sat
                </button>
                <button onclick="saveAltitudeSettings()" class="bg-cyan-600 hover:bg-cyan-500 text-xs text-white px-4 py-1.5 rounded-lg font-bold transition">
                    Guardar
                </button>
            </div>
        </div>
    </div>

    <script>
        const CARTO_KEY = 'cb1_2l65_1_3a8e83de8b889ec5e4e98278';

        let observerLat = parseFloat(localStorage.getItem('obs_lat') || 41.6079);
        let observerLon = parseFloat(localStorage.getItem('obs_lon') || 2.2876);
        let terrainElevationM = parseFloat(localStorage.getItem('obs_terrain_m') || 145.0);
        let buildingOffsetM = parseFloat(localStorage.getItem('obs_building_m') || 0.0);
        let userLeadSec = parseFloat(localStorage.getItem('user_lead_sec') || 6.0);

        let activeFilter = 'moon';
        let viewMode = 'map';
        let isLocationLocked = true;
        let serverClockDelta = 0.0;
        
        // ALERTAS DE SONIDO Y VOZ ACTIVADAS POR DEFECTO
        let audioEnabled = true;
        let voiceEnabled = true;
        let audioContext = null;
        let lastBeepedFlight = '';
        
        let detectedTransits = new Set();
        let spokenCountdowns = new Set();
        let tcaSmoothedEpochs = {};

        let sunDataGlobal = { azimuth: 0, elevation: 0, visible: false, radius_deg: 0.26 };
        let moonDataGlobal = { azimuth: 0, elevation: 0, visible: false, radius_deg: 0.26 };

        let map, obsMarker;
        let rangeCircles = [];
        let sunLine = null, sunIconMarker = null;
        let moonLine = null, moonIconMarker = null;
        let tcaMarker = null, trajectoryLine = null;
        let currentBaseTileLayer, isSatelliteMode = false;
        
        const planesState = {};
        let activeAircraftData = [];

        // Inicialización y desbloqueo transparente de AudioContext en el primer toque de pantalla
        function initAndUnlockAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        ['click', 'touchstart', 'pointerdown', 'keydown'].forEach(ev => {
            window.addEventListener(ev, initAndUnlockAudio, { passive: true });
        });

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
            className: '',
            html: '<div class="obs-target"><div class="obs-sweep"></div><div class="obs-ring"></div><div class="obs-dot"></div></div>',
            iconSize: [44, 44], iconAnchor: [22, 22]
        });

        obsMarker = L.marker([observerLat, observerLon], { draggable: false, icon: obsCustomIcon }).addTo(map);

        obsMarker.on('drag', function(e) {
            const pos = e.target.getLatLng();
            observerLat = pos.lat; observerLon = pos.lng;
            renderAstroVectors();
            drawRangeRings();
        });

        obsMarker.on('dragend', function (e) {
            const pos = e.target.getLatLng();
            saveAndSetObserverPos(pos.lat, pos.lng);
        });

        function adjustLead(delta) {
            userLeadSec = Math.max(0.0, Math.min(15.0, Math.round((userLeadSec + delta) * 10) / 10));
            localStorage.setItem('user_lead_sec', userLeadSec.toFixed(1));
            document.getElementById('lead-display').innerText = (userLeadSec >= 0 ? '+' : '') + userLeadSec.toFixed(1) + 's';
        }

        function drawRangeRings() {
            rangeCircles.forEach(c => map.removeLayer(c));
            rangeCircles = [];
            [15000, 30000, 50000].forEach((r) => {
                const circle = L.circle([observerLat, observerLon], {
                    radius: r, color: '#06b6d4', weight: 1, dashArray: '4, 8', fill: false, opacity: 0.22, interactive: false
                }).addTo(map);
                rangeCircles.push(circle);
            });
        }

        function toggleSettingsModal() {
            const m = document.getElementById('settings-modal');
            m.classList.toggle('hidden'); m.classList.toggle('flex');
            if (!m.classList.contains('hidden')) {
                updateAltitudeDisplay();
            }
        }

        function calculateModalTotalAlt() {
            const tVal = parseFloat(document.getElementById('terrain-elevation-input').value) || 0;
            const bVal = parseFloat(document.getElementById('building-offset').value) || 0;
            document.getElementById('total-alt-display').innerText = `${Math.round(tVal + bVal)}m`;
        }

        function saveAltitudeSettings() {
            const tVal = parseFloat(document.getElementById('terrain-elevation-input').value);
            const bVal = parseFloat(document.getElementById('building-offset').value);
            if (!isNaN(tVal)) terrainElevationM = Math.max(-400, Math.min(9000, tVal));
            if (!isNaN(bVal)) buildingOffsetM = Math.max(0, Math.min(1000, bVal));
            
            localStorage.setItem('obs_terrain_m', terrainElevationM.toString());
            localStorage.setItem('obs_building_m', buildingOffsetM.toString());
            
            updateAltitudeDisplay();
            fetchData();
            toggleSettingsModal();
        }

        function detectTerrainAuto() {
            fetchTerrainElevation(observerLat, observerLon, true);
        }

        function testAudioSystem() {
            initAndUnlockAudio();
            playChime();
            speak("Audio and vocal alerts verified active");
        }

        function setFilterMode(mode) {
            activeFilter = mode;
            ['moon', 'sun', 'all'].forEach(m => document.getElementById('btn-flt-' + m).classList.toggle('on', m === mode));
            document.getElementById('moon-status-card').classList.toggle('on', mode !== 'sun');
            document.getElementById('sun-status-card').classList.toggle('on', mode !== 'moon');
            document.getElementById('filter-indicator').innerText = { moon: 'TARGET · MOON', sun: 'TARGET · SUN', all: 'TARGET · DUAL' }[mode];
            const ind = document.getElementById('footer-vector-indicator');
            ind.className = 'font-bold ' + (mode === 'sun' ? 'text-amber-400' : 'text-cyan-400');
            document.getElementById('footer-astro-info').innerText = { moon: 'Optical sight line · Moon', sun: 'Optical sight line · Sun', all: 'Dual optical sight lines' }[mode];
            renderAstroVectors();
            updateHUDCountdowns();
        }

        function toggleVoice() {
            voiceEnabled = !voiceEnabled;
            document.getElementById('voice-btn').classList.toggle('on', voiceEnabled);
            if (voiceEnabled) speak("Vocal radar active");
        }

        function speak(text) {
            if (!voiceEnabled || !('speechSynthesis' in window)) return;
            initAndUnlockAudio();
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(text);
            msg.rate = 1.05;
            window.speechSynthesis.speak(msg);
        }

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            document.getElementById('audio-btn').classList.toggle('on', audioEnabled);
            if (audioEnabled) { initAndUnlockAudio(); playChime(); }
        }

        function playChime() {
            if (!audioEnabled) return;
            initAndUnlockAudio();
            try {
                const now = audioContext.currentTime;
                [523.25, 659.25, 783.99].forEach((freq, i) => {
                    const osc = audioContext.createOscillator();
                    const gain = audioContext.createGain();
                    osc.frequency.value = freq;
                    gain.gain.setValueAtTime(0.12, now + i * 0.08);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.32);
                    osc.connect(gain); gain.connect(audioContext.destination);
                    osc.start(now + i * 0.08); osc.stop(now + i * 0.08 + 0.32);
                });
            } catch (e) {}
        }

        function playTone(freq, dur) {
            if (!audioEnabled) return;
            initAndUnlockAudio();
            try {
                const osc = audioContext.createOscillator();
                const gain = audioContext.createGain();
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0.16, audioContext.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + dur);
                osc.connect(gain); gain.connect(audioContext.destination);
                osc.start(); osc.stop(audioContext.currentTime + dur);
            } catch (e) {}
        }

        function playTransitChord() {
            if (!audioEnabled) return;
            initAndUnlockAudio();
            try {
                const now = audioContext.currentTime;
                [880, 1108.73, 1318.51, 1760].forEach(f => {
                    const osc = audioContext.createOscillator();
                    const gain = audioContext.createGain();
                    osc.frequency.value = f;
                    gain.gain.setValueAtTime(0.14, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
                    osc.connect(gain); gain.connect(audioContext.destination);
                    osc.start(now); osc.stop(now + 0.65);
                });
            } catch (e) {}
        }

        async function fetchTerrainElevation(lat, lon, forceUpdate = false) {
            try {
                const res = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}`);
                const data = await res.json();
                if (data.elevation && data.elevation.length > 0) {
                    terrainElevationM = Math.round(parseFloat(data.elevation[0]));
                    localStorage.setItem('obs_terrain_m', terrainElevationM.toString());
                    updateAltitudeDisplay();
                    if (forceUpdate) fetchData();
                }
            } catch (e) {
                updateAltitudeDisplay();
            }
        }

        function updateAltitudeDisplay() {
            const totalAlt = Math.round(terrainElevationM + buildingOffsetM);
            const badge = document.getElementById('obs-alt-badge');
            if (badge) badge.innerText = `${totalAlt} m ASL`;

            const tInput = document.getElementById('terrain-elevation-input');
            const bInput = document.getElementById('building-offset');
            const totalDisp = document.getElementById('total-alt-display');
            if (tInput) tInput.value = terrainElevationM;
            if (bInput) bInput.value = buildingOffsetM;
            if (totalDisp) totalDisp.innerText = `${totalAlt}m`;
        }

        async function saveAndSetObserverPos(lat, lon) {
            observerLat = lat; observerLon = lon;
            localStorage.setItem('obs_lat', lat.toString());
            localStorage.setItem('obs_lon', lon.toString());
            document.getElementById('obs-coords').innerText = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
            await fetchTerrainElevation(lat, lon);
            drawRangeRings();
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

        function getRealisticMoonSvgHtml(size = 38) {
            return `
                <div style="position:relative; width:${size}px; height:${size}px; display:flex; align-items:center; justify-content:center; filter: drop-shadow(0 0 10px rgba(56,189,248,0.7));">
                    <svg viewBox="0 0 100 100" width="${size}" height="${size}" style="overflow:visible;">
                        <defs>
                            <radialGradient id="lunarLimbGrad" cx="44%" cy="40%" r="56%">
                                <stop offset="0%" stop-color="#f8fafc" />
                                <stop offset="55%" stop-color="#e2e8f0" />
                                <stop offset="82%" stop-color="#94a3b8" />
                                <stop offset="100%" stop-color="#475569" />
                            </radialGradient>
                            <clipPath id="lunarSphereClip">
                                <circle cx="50" cy="50" r="46" />
                            </clipPath>
                        </defs>
                        <circle cx="50" cy="50" r="46" fill="url(#lunarLimbGrad)" stroke="#38bdf8" stroke-width="1.2"/>
                        <g clip-path="url(#lunarSphereClip)">
                            <path d="M 18,36 C 14,48 18,68 30,72 C 38,72 40,62 38,50 C 36,40 28,32 18,36 Z" fill="#334155" opacity="0.68"/>
                            <circle cx="39" cy="34" r="12" fill="#334155" opacity="0.72"/>
                            <circle cx="59" cy="35" r="8.5" fill="#334155" opacity="0.7"/>
                            <ellipse cx="67" cy="48" rx="9" ry="8" fill="#334155" opacity="0.75"/>
                            <ellipse cx="80" cy="38" rx="5.5" ry="4.5" fill="#1e293b" opacity="0.85"/>
                            <ellipse cx="73" cy="60" rx="8" ry="7" fill="#334155" opacity="0.7"/>
                            <circle cx="65" cy="63" r="5" fill="#334155" opacity="0.65"/>
                            <ellipse cx="40" cy="64" rx="8" ry="7" fill="#334155" opacity="0.7"/>
                            <circle cx="27" cy="63" r="4.5" fill="#334155" opacity="0.65"/>
                            <line x1="48" y1="80" x2="30" y2="60" stroke="#ffffff" stroke-width="0.75" opacity="0.65"/>
                            <line x1="48" y1="80" x2="68" y2="65" stroke="#ffffff" stroke-width="0.75" opacity="0.65"/>
                            <line x1="48" y1="80" x2="48" y2="40" stroke="#ffffff" stroke-width="0.6" opacity="0.55"/>
                            <line x1="48" y1="80" x2="18" y2="82" stroke="#ffffff" stroke-width="0.6" opacity="0.55"/>
                            <line x1="48" y1="80" x2="76" y2="82" stroke="#ffffff" stroke-width="0.6" opacity="0.55"/>
                            <circle cx="48" cy="80" r="2.4" fill="#ffffff"/>
                            <circle cx="36" cy="44" r="2.2" fill="#ffffff" opacity="0.95"/>
                            <circle cx="26" cy="45" r="1.6" fill="#ffffff" opacity="0.9"/>
                            <circle cx="24" cy="33" r="1.8" fill="#ffffff" opacity="1.0"/>
                        </g>
                    </svg>
                </div>
            `;
        }

        function getRealisticSunSvgHtml(size = 38) {
            return `
                <div style="position:relative; width:${size}px; height:${size}px; display:flex; align-items:center; justify-content:center; filter: drop-shadow(0 0 12px rgba(245,158,11,0.9));">
                    <svg viewBox="0 0 100 100" width="${size}" height="${size}" style="overflow:visible;">
                        <defs>
                            <radialGradient id="solarDiscGrad" cx="45%" cy="45%" r="55%">
                                <stop offset="0%" stop-color="#fffbeb" />
                                <stop offset="60%" stop-color="#fbbf24" />
                                <stop offset="90%" stop-color="#f59e0b" />
                                <stop offset="100%" stop-color="#d97706" />
                            </radialGradient>
                        </defs>
                        <circle cx="50" cy="50" r="44" fill="none" stroke="#fef08a" stroke-width="1.5" stroke-dasharray="4, 4" opacity="0.8" />
                        <circle cx="50" cy="50" r="38" fill="url(#solarDiscGrad)" stroke="#fde047" stroke-width="1.8" />
                        <ellipse cx="44" cy="42" rx="2" ry="1.5" fill="#78350f" opacity="0.75"/>
                        <ellipse cx="58" cy="46" rx="2.5" ry="1.8" fill="#78350f" opacity="0.75"/>
                    </svg>
                </div>
            `;
        }

        function renderAstroVectors() {
            const distKm = 55;
            const showMoon = (activeFilter === 'all' || activeFilter === 'moon') && moonDataGlobal.visible;
            const showSun = (activeFilter === 'all' || activeFilter === 'sun') && sunDataGlobal.visible;

            if (showMoon) {
                const radAzM = (moonDataGlobal.azimuth * Math.PI) / 180;
                const endM = [
                    observerLat + (distKm * Math.cos(radAzM)) / 111.0,
                    observerLon + (distKm * Math.sin(radAzM)) / (111.0 * Math.cos(observerLat * Math.PI / 180))
                ];
                if (moonLine) {
                    moonLine.setLatLngs([[observerLat, observerLon], endM]);
                } else {
                    moonLine = L.polyline([[observerLat, observerLon], endM], { color: '#38bdf8', weight: 2.2, dashArray: '5, 8', opacity: 0.95 }).addTo(map);
                }
                if (moonIconMarker) {
                    moonIconMarker.setLatLng(endM);
                } else {
                    moonIconMarker = L.marker(endM, { 
                        icon: L.divIcon({ className: '', html: getRealisticMoonSvgHtml(38), iconSize: [38, 38], iconAnchor: [19, 19] }) 
                    }).addTo(map);
                }
            } else {
                if (moonLine) { map.removeLayer(moonLine); moonLine = null; }
                if (moonIconMarker) { map.removeLayer(moonIconMarker); moonIconMarker = null; }
            }

            if (showSun) {
                const radAzS = (sunDataGlobal.azimuth * Math.PI) / 180;
                const endS = [
                    observerLat + (distKm * Math.cos(radAzS)) / 111.0,
                    observerLon + (distKm * Math.sin(radAzS)) / (111.0 * Math.cos(observerLat * Math.PI / 180))
                ];
                if (sunLine) {
                    sunLine.setLatLngs([[observerLat, observerLon], endS]);
                } else {
                    sunLine = L.polyline([[observerLat, observerLon], endS], { color: '#f59e0b', weight: 2.2, dashArray: '5, 8', opacity: 0.95 }).addTo(map);
                }
                if (sunIconMarker) {
                    sunIconMarker.setLatLng(endS);
                } else {
                    sunIconMarker = L.marker(endS, { 
                        icon: L.divIcon({ className: '', html: getRealisticSunSvgHtml(38), iconSize: [38, 38], iconAnchor: [19, 19] }) 
                    }).addTo(map);
                }
            } else {
                if (sunLine) { map.removeLayer(sunLine); sunLine = null; }
                if (sunIconMarker) { map.removeLayer(sunIconMarker); sunIconMarker = null; }
            }
        }

        // ---------- Sincronización de reloj tipo NTP (muestra de menor RTT) ----------
        let syncRtt = null;
        async function syncClock(n = 5) {
            if (syncRtt !== null) syncRtt *= 1.2;
            for (let i = 0; i < n; i++) {
                try {
                    const t0 = Date.now() / 1000.0;
                    const r = await fetch('/api/time?i=' + i, { cache: 'no-store' });
                    const j = await r.json();
                    const t1 = Date.now() / 1000.0;
                    const rtt = t1 - t0;
                    const off = (t0 + t1) / 2.0 - j.t;
                    if (syncRtt === null || rtt <= syncRtt * 1.3 + 0.01) {
                        serverClockDelta = (syncRtt === null) ? off : serverClockDelta * 0.6 + off * 0.4;
                        syncRtt = (syncRtt === null) ? rtt : Math.min(rtt, syncRtt * 1.05);
                    }
                } catch (e) {}
                await new Promise(res => setTimeout(res, 120));
            }
            const b = document.getElementById('sync-badge');
            if (b && syncRtt !== null) b.innerText = `SYNC ±${Math.max(1, Math.round(syncRtt * 500))}ms`;
        }
        const syncedNow = () => (Date.now() / 1000.0) - serverClockDelta;

        function projectLL(lat, lon, speedMs, trackRad, dt) {
            const d = speedMs * dt;
            return [lat + (d * Math.cos(trackRad)) / 111139.0,
                    lon + (d * Math.sin(trackRad)) / (111139.0 * Math.cos(lat * Math.PI / 180.0))];
        }

        function planeIconHtml(plane) {
            const color = getAltitudeColor(plane.alt_ft);
            const anyT = (plane.moon.is_transit && moonDataGlobal.visible) || (plane.sun.is_transit && sunDataGlobal.visible);
            const anyC = (plane.moon.is_close && moonDataGlobal.visible) || (plane.sun.is_close && sunDataGlobal.visible);
            const glow = anyT ? 'filter: drop-shadow(0 0 10px #ef4444);' : (anyC ? 'filter: drop-shadow(0 0 6px #f59e0b);' : '');
            return `<div style="transform: rotate(${plane.track}deg); width: 24px; height: 24px; display:flex; align-items:center; justify-content:center; ${glow}">
                <svg viewBox="0 0 24 24" width="22" height="22" fill="${color}"><path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg></div>`;
        }

        function popupHtml(p) {
            const arrow = p.vr_fpm > 300 ? '▲' : (p.vr_fpm < -300 ? '▼' : '►');
            return `<div><b style="color:#7dd3fc">${p.callsign}</b> · ${p.model}${p.reg ? ' · ' + p.reg : ''}<br>
                ALT ${p.alt_ft.toLocaleString()} ft ${p.alt_type} ${arrow} ${p.vr_fpm} ft/m<br>
                GS ${p.speed_kt} kt · HDG ${p.track}°<br>DIST ${p.distance_km} km · SPAN ${p.wingspan_m} m</div>`;
        }

        // ---------- Petición de datos: nunca solapada, nunca fuera de orden ----------
        let fetchSeq = 0, appliedSeq = 0, fetchBusy = false, fetchQueued = false;
        async function fetchData() {
            if (fetchBusy) { fetchQueued = true; return; }
            fetchBusy = true;
            const seq = ++fetchSeq;
            const ctl = new AbortController();
            const to = setTimeout(() => ctl.abort(), 6000);
            const feed = document.getElementById('feed-badge');
            try {
                const totalObserverAlt = Math.round(terrainElevationM + buildingOffsetM);
                const res = await fetch(`/api/data?lat=${observerLat}&lon=${observerLon}&alt=${totalObserverAlt}&lead=${userLeadSec}`, { cache: 'no-store', signal: ctl.signal });
                const data = await res.json();
                if (seq < appliedSeq || data.error || !data.moon) return;
                appliedSeq = seq;
                feed.dataset.state = 'ok';
                applyData(data);
            } catch (err) {
                feed.dataset.state = 'bad'; feed.innerText = 'RECONNECTING';
            } finally {
                clearTimeout(to);
                fetchBusy = false;
                if (fetchQueued) { fetchQueued = false; setTimeout(fetchData, 0); }
            }
        }

        function applyData(data) {
            moonDataGlobal = data.moon;
            sunDataGlobal = data.sun;
            if (data.source_feed) document.getElementById('feed-badge').innerText = data.source_feed.toUpperCase();

            const mCoords = document.getElementById('moon-coords');
            const mBadge = document.getElementById('moon-badge');
            mCoords.innerText = moonDataGlobal.visible ? `Moon: Az ${moonDataGlobal.azimuth}° | Alt +${moonDataGlobal.elevation}°` : `Moon Hidden (${moonDataGlobal.elevation}°)`;
            mBadge.innerText = `${moonDataGlobal.event_type}: ${moonDataGlobal.next_event_str}`;
            const sCoords = document.getElementById('sun-coords');
            const sBadge = document.getElementById('sun-badge');
            sCoords.innerText = sunDataGlobal.visible ? `Sun: Az ${sunDataGlobal.azimuth}° | Alt +${sunDataGlobal.elevation}°` : `Sun Hidden (${sunDataGlobal.elevation}°)`;
            sBadge.innerText = `${sunDataGlobal.event_type}: ${sunDataGlobal.next_event_str}`;
            renderAstroVectors();

            activeAircraftData = data.aircraft || [];
            document.getElementById('plane-count').innerText = activeAircraftData.length;
            const tRef = data.server_time;      // instante al que corresponden lat/lon recibidos
            const nowSec = syncedNow();
            const seenKeys = new Set();

            activeAircraftData.forEach(plane => {
                const k = plane._k = plane.callsign + '|' + plane.reg;
                seenKeys.add(k);
                const target = activeFilter === 'sun' ? plane.sun : (activeFilter === 'moon' ? plane.moon : plane.primary);

                // Suavizado mínimo del TCA: solo amortigua jitter < 0.3 s, nunca introduce retardo en cambios reales
                const prev = tcaSmoothedEpochs[k], cur = target.tca_epoch;
                tcaSmoothedEpochs[k] = (prev && Math.abs(cur - prev) < 0.3) ? prev * 0.5 + cur * 0.5 : cur;

                const trackRad = (plane.track * Math.PI) / 180.0;
                const sig = Math.round(plane.track) + '|' + getAltitudeColor(plane.alt_ft) + '|' + plane.moon.is_transit + plane.sun.is_transit + plane.moon.is_close + plane.sun.is_close;
                const [pl, po] = projectLL(plane.lat, plane.lon, plane.speed_ms, trackRad, Math.max(-1, Math.min(8, nowSec - tRef)));
                let st = planesState[k];

                if (!st) {
                    const marker = L.marker([pl, po], { icon: L.divIcon({ className: '', html: planeIconHtml(plane), iconSize: [24, 24], iconAnchor: [12, 12] }) }).addTo(map);
                    marker.bindPopup(popupHtml(plane));
                    st = planesState[k] = { marker, curLat: pl, curLon: po, errN: 0, errE: 0, iconSig: sig };
                } else {
                    // Corrección suave: el error entre lo dibujado y la nueva predicción decae en ~0.6 s (sin saltos)
                    const eN = (st.curLat - pl) * 111139.0;
                    const eE = (st.curLon - po) * 111139.0 * Math.cos(pl * Math.PI / 180.0);
                    if (Math.hypot(eN, eE) < 1500) { st.errN = eN; st.errE = eE; } else { st.errN = 0; st.errE = 0; }
                    if (st.iconSig !== sig) {
                        st.marker.setIcon(L.divIcon({ className: '', html: planeIconHtml(plane), iconSize: [24, 24], iconAnchor: [12, 12] }));
                        st.iconSig = sig;
                    }
                    st.marker.setPopupContent(popupHtml(plane));
                }
                st.baseLat = plane.lat; st.baseLon = plane.lon; st.tRef = tRef;
                st.speedMs = plane.speed_ms; st.trackRad = trackRad;
                st.lastSeen = nowSec; st.data = plane;
            });

            for (const k in planesState) {
                if (!seenKeys.has(k) && nowSec - planesState[k].lastSeen > 6.0) {
                    map.removeLayer(planesState[k].marker);
                    delete planesState[k];
                    delete tcaSmoothedEpochs[k];
                }
            }
            updateHUDCountdowns();
        }

        // ---------- Animación: posición absoluta = base + v·(t_sync − t_ref), sin acumular deriva ----------
        let lastAnimTime = performance.now();
        function animateFrame(nowMs) {
            const dtF = Math.min(0.1, Math.max(0.001, (nowMs - lastAnimTime) / 1000.0));
            lastAnimTime = nowMs;
            const now = syncedNow();
            const decay = Math.exp(-dtF / 0.6);
            for (const k in planesState) {
                const p = planesState[k];
                if (p.baseLat === undefined) continue;
                const [la, lo] = projectLL(p.baseLat, p.baseLon, p.speedMs, p.trackRad, Math.max(-1.0, Math.min(8.0, now - p.tRef)));
                p.errN *= decay; p.errE *= decay;
                p.curLat = la + p.errN / 111139.0;
                p.curLon = lo + p.errE / (111139.0 * Math.cos(la * Math.PI / 180.0));
                p.marker.setLatLng([p.curLat, p.curLon]);
            }
            requestAnimationFrame(animateFrame);
        }

        function renderTransitDiscDiagram(target) {
            const isSun = target.target === 'sun';
            const bodyColor = isSun ? '#fbbf24' : '#38bdf8';
            const bodyRad = isSun ? (sunDataGlobal.radius_deg || 0.26) : (moonDataGlobal.radius_deg || 0.26);
            const normOffset = Math.max(-1.7, Math.min(1.7, target.vertical_offset_deg / Math.max(0.01, bodyRad)));
            const chordY = 24 - (normOffset * 11);

            return `
                <div class="flex items-center gap-2.5 bg-slate-950/90 p-2 rounded-lg border border-slate-800/90 mt-1 shadow-inner">
                    <svg width="48" height="48" viewBox="0 0 48 48" class="shrink-0 select-none">
                        <circle cx="24" cy="24" r="22" fill="none" stroke="#1e293b" stroke-width="1"/>
                        <circle cx="24" cy="24" r="18" fill="${isSun ? '#451a03' : '#082f49'}" stroke="${bodyColor}" stroke-width="1.4"/>
                        <line x1="4" y1="24" x2="44" y2="24" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
                        <line x1="24" y1="4" x2="24" y2="44" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
                        <line x1="24" y1="2" x2="24" y2="6" stroke="${bodyColor}" stroke-width="1.2"/>
                        <line x1="24" y1="42" x2="24" y2="46" stroke="${bodyColor}" stroke-width="1.2"/>
                        <line x1="2" y1="24" x2="6" y2="24" stroke="${bodyColor}" stroke-width="1.2"/>
                        <line x1="42" y1="24" x2="46" y2="24" stroke="${bodyColor}" stroke-width="1.2"/>
                        <line x1="2" y1="${chordY.toFixed(1)}" x2="46" y2="${chordY.toFixed(1)}" stroke="${target.is_transit ? '#ef4444' : '#f59e0b'}" stroke-width="2.2" stroke-linecap="round"/>
                    </svg>
                    <div class="flex flex-col text-[10px] leading-tight font-mono">
                        <div class="flex items-center gap-1">
                            <span class="text-slate-500 uppercase font-bold tracking-wider text-[8px]">CHORD:</span>
                            <span class="font-bold ${target.is_transit ? 'text-red-400' : 'text-amber-400'}">${target.position_descriptor}</span>
                        </div>
                        <span class="text-slate-400 mt-0.5">Offset: <b class="text-slate-200 tabular-nums">${target.vertical_offset_deg > 0 ? '+' : ''}${target.vertical_offset_deg}°</b> (${target.vertical_body_diams} diam.)</span>
                        <span class="text-slate-400">Coverage: <b class="text-indigo-300 tabular-nums">${target.disk_coverage_pct}%</b> (${target.transit_duration_s}s dur.)</span>
                    </div>
                </div>
            `;
        }

        function updateHUDCountdowns() {
            const container = document.getElementById('alerts-container');
            if (!activeAircraftData || activeAircraftData.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-500 text-center py-8 font-mono">Scanning airspace for transit intercept...</div>';
                clearTcaTrajectory();
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
            let priorityTransitFound = false;

            displayList.forEach(({ plane, target }) => {
                const isTargetVisible = (target.target === 'sun' ? sunDataGlobal.visible : moonDataGlobal.visible);
                const smoothedEpoch = tcaSmoothedEpochs[plane._k] || target.tca_epoch;
                const remaining = Math.max(0.0, smoothedEpoch - now);

                const isTransit = target.is_transit && isTargetVisible;
                const isClose = target.is_close && isTargetVisible;
                const sym = target.target === 'sun' ? '☀️' : '🌕';

                const mins = Math.floor(remaining / 60);
                const secs = (remaining % 60).toFixed(1).padStart(4, '0');
                const timerStr = remaining > 0 ? `T-${mins.toString().padStart(2, '0')}:${secs}` : `TRANSITING!`;

                let cardBorder = 'border-slate-800/80 bg-slate-950/60';
                let tagHtml = `<span class="bg-slate-900 text-slate-400 border border-slate-800 px-1.5 py-0.5 rounded text-[8px] font-bold font-mono tracking-wider">EN ROUTE</span>`;

                if (isTransit) {
                    cardBorder = 'border-red-500/90 bg-red-950/60 glow-red';
                    tagHtml = `<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[8px] font-black font-mono tracking-wider animate-pulse">TRANSIT LOCK</span>`;
                    
                    if (!priorityTransitFound && remaining > 0) {
                        priorityTransitFound = true;
                        renderTcaTrajectory(plane, target);
                    }

                    const flightKey = `${plane.callsign}-${target.target}`;

                    if (!detectedTransits.has(flightKey)) {
                        detectedTransits.add(flightKey);
                        playChime();
                        speak(`Transit detected for flight ${plane.callsign}`);
                    }

                    const wholeSec = Math.floor(remaining);
                    const alertKey = `${flightKey}-${wholeSec}`;

                    if (!spokenCountdowns.has(alertKey) && remaining > 0) {
                        if (wholeSec === 180) {
                            spokenCountdowns.add(alertKey); playChime(); speak("Transit in 3 minutes");
                        } else if (wholeSec === 120) {
                            spokenCountdowns.add(alertKey); playChime(); speak("Transit in 2 minutes, prepare gear");
                        } else if (wholeSec === 60) {
                            spokenCountdowns.add(alertKey); playChime(); speak("Warning: transit in 60 seconds");
                        } else if (wholeSec === 30) {
                            spokenCountdowns.add(alertKey); speak("30 seconds");
                        } else if (wholeSec === 10) {
                            spokenCountdowns.add(alertKey); speak("10 seconds");
                        } else if ([5, 4, 3, 2, 1].includes(wholeSec)) {
                            spokenCountdowns.add(alertKey); playTone(1050 + (5 - wholeSec) * 140, 0.1);
                        }
                    }

                    if (remaining <= 0.2 && lastBeepedFlight !== flightKey) {
                        playTransitChord();
                        lastBeepedFlight = flightKey;
                    }
                } else if (isClose) {
                    cardBorder = 'border-amber-500/80 bg-amber-950/50 glow-amber';
                    tagHtml = `<span class="bg-amber-600 text-white px-2 py-0.5 rounded text-[8px] font-bold font-mono tracking-wider">CLOSE PASS</span>`;
                    if (!priorityTransitFound && remaining > 0 && remaining < 180) {
                        priorityTransitFound = true;
                        renderTcaTrajectory(plane, target);
                    }
                }

                html += `
                    <div onclick="focusPlane('${plane._k}')" 
                         class="p-2.5 rounded-xl border ${cardBorder} text-xs flex flex-col gap-1.5 transition cursor-pointer hud-card">
                        
                        <div class="flex justify-between items-center border-b border-slate-800/80 pb-1.5 font-mono">
                            <div class="flex items-center gap-1.5">
                                <span class="font-black text-sm tracking-wide ${isTransit ? 'text-red-400' : 'text-slate-100'}">${plane.callsign}</span>
                                <span class="px-1.5 py-0.2 bg-slate-900 text-cyan-300 border border-slate-700 rounded text-[9px] font-bold">${plane.model}</span>
                                <span class="text-[8px] px-1 bg-slate-900 text-slate-400 border border-slate-800 rounded">${plane.alt_type}</span>
                            </div>
                            ${tagHtml}
                        </div>
                        
                        <div class="grid grid-cols-3 gap-1.5 items-center font-mono my-0.5">
                            <div class="col-span-1 px-2 py-1 rounded bg-slate-950 border ${isTransit ? 'border-red-500/80 text-red-400' : (isClose ? 'border-amber-500/80 text-amber-400' : 'border-slate-800 text-cyan-400')} flex flex-col items-center justify-center">
                                <span class="text-[7.5px] text-slate-500 uppercase font-bold tracking-wider">EST. TCA</span>
                                <span class="text-xs font-black tabular-nums tracking-wide">${timerStr}</span>
                            </div>
                            <div class="col-span-2 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[9.5px]">
                                <div><span class="text-slate-500 text-[8px] uppercase tracking-wider block">ALTITUDE</span><b class="tabular-nums">${plane.alt_ft.toLocaleString()} ft</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase tracking-wider block">V/S RATE</span><b class="tabular-nums">${plane.vr_fpm > 0 ? '+' : ''}${plane.vr_fpm} ft/m</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase tracking-wider block">SPEED</span><b class="tabular-nums">${plane.speed_kt} kt</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase tracking-wider block">DISTANCE</span><b class="tabular-nums">${plane.distance_km} km</b></div>
                            </div>
                        </div>

                        <div class="flex justify-between items-center text-[9px] font-mono text-slate-400 border-t border-slate-800/60 pt-1">
                            <span>HDG: <b class="text-slate-200 tabular-nums">${plane.track}°</b></span>
                            <span>SPAN: <b class="text-slate-200 tabular-nums">${plane.wingspan_m}m</b></span>
                            <span>ANG SIZE: <b class="text-indigo-300 tabular-nums">${target.angular_size_arcsec}"</b></span>
                        </div>

                        ${(isTransit || isClose) ? renderTransitDiscDiagram(target) : ''}
                    </div>
                `;
            });

            if (!priorityTransitFound) clearTcaTrajectory();
            container.innerHTML = html;
        }

        function renderTcaTrajectory(plane, target) {
            const start = [plane.lat, plane.lon];
            const tcaPos = [target.tca_lat, target.tca_lon];

            if (trajectoryLine) {
                trajectoryLine.setLatLngs([start, tcaPos]);
            } else {
                trajectoryLine = L.polyline([start, tcaPos], { color: '#ef4444', weight: 2.2, dashArray: '4, 6', opacity: 0.85 }).addTo(map);
            }

            const tcaHtml = `
                <div class="tca-target">
                    <div class="tca-ring"></div>
                    <div class="tca-crosshair-h"></div>
                    <div class="tca-crosshair-v"></div>
                    <div class="tca-core"></div>
                </div>
            `;

            if (tcaMarker) {
                tcaMarker.setLatLng(tcaPos);
            } else {
                tcaMarker = L.marker(tcaPos, {
                    icon: L.divIcon({ className: '', html: tcaHtml, iconSize: [28, 28], iconAnchor: [14, 14] })
                }).addTo(map);
            }
        }

        function clearTcaTrajectory() {
            if (trajectoryLine) { map.removeLayer(trajectoryLine); trajectoryLine = null; }
            if (tcaMarker) { map.removeLayer(tcaMarker); tcaMarker = null; }
        }

        function focusPlane(callsign) {
            if (planesState[callsign]) {
                map.setView([planesState[callsign].curLat, planesState[callsign].curLon], 11);
                planesState[callsign].marker.openPopup();
            }
        }

        document.getElementById('lead-display').innerText = (userLeadSec >= 0 ? '+' : '') + userLeadSec.toFixed(1) + 's';
        document.getElementById('obs-coords').innerText = `${observerLat.toFixed(4)}, ${observerLon.toFixed(4)}`;

        // Iconos vectoriales de la Luna y Sol en la cabecera
        document.getElementById('header-moon-icon').innerHTML = getRealisticMoonSvgHtml(20);
        document.getElementById('header-sun-icon').innerHTML = getRealisticSunSvgHtml(20);

        updateAltitudeDisplay();
        drawRangeRings();
        setFilterMode('moon');
        syncClock(4).then(fetchData);
        fetchTerrainElevation(observerLat, observerLon);
        setInterval(fetchData, 1500);
        setInterval(syncClock, 20000);
        setInterval(updateHUDCountdowns, 100);
        requestAnimationFrame(animateFrame);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print(f" [OK] LUNAR TRANSIT RADAR PRO v27 // TIMESTAMPED ADS-B INGEST")
    print(f" [OK] Audio & Voice Alerts: ACTIVE BY DEFAULT (Autoplay-Unlocked)")
    print(f" [OK] Dual Elevation Controller (Manual + Auto-Detect): ACTIVE")
    print(f" [OK] Server Online on port: {port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

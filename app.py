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

def safe_float(val, default=None):
    if val is None or val == '' or val == 'null' or val == 'ground':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

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
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(max(1e-9, 1.0 - WGS84_E2 * (sin_lat ** 2)))
    x = (n + h_m) * math.cos(lat) * math.cos(lon)
    y = (n + h_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
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
# 2. GESTIÓN DE CACHÉ ADS-B
# =========================================================================
CACHE = {
    'lat': 0.0, 'lon': 0.0, 'timestamp': 0.0, 'aircraft': [], 'source': 'airplanes.live', 'feed_now': 0.0
}
HTTP_SESSION = requests.Session()

def get_live_aircraft(cur_lat, cur_lon):
    now = time.time()
    if now - CACHE['timestamp'] < 1.8 and abs(cur_lat - CACHE['lat']) < 0.04 and abs(cur_lon - CACHE['lon']) < 0.04:
        return CACHE['aircraft'], CACHE['source'], CACHE['feed_now']

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) LunarTransitRadar/27.0'}
    
    # 1. airplanes.live
    try:
        url = f"https://api.airplanes.live/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=1.8)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            feed_now = safe_float(data.get('now'), default=now)
            if feed_now > 1e11: feed_now /= 1000.0
            if ac:
                CACHE['lat'], CACHE['lon'], CACHE['timestamp'], CACHE['aircraft'], CACHE['source'], CACHE['feed_now'] = cur_lat, cur_lon, now, ac, 'airplanes.live', feed_now
                return ac, 'airplanes.live', feed_now
    except Exception:
        pass

    # 2. adsb.lol fallback
    try:
        url = f"https://api.adsb.lol/v2/point/{cur_lat:.4f}/{cur_lon:.4f}/80"
        r = HTTP_SESSION.get(url, headers=headers, timeout=1.8)
        if r.status_code == 200:
            data = r.json()
            ac = data.get('ac', [])
            feed_now = safe_float(data.get('now'), default=now)
            if feed_now > 1e11: feed_now /= 1000.0
            if ac:
                CACHE['lat'], CACHE['lon'], CACHE['timestamp'], CACHE['aircraft'], CACHE['source'], CACHE['feed_now'] = cur_lat, cur_lon, now, ac, 'adsb.lol', feed_now
                return ac, 'adsb.lol', feed_now
    except Exception:
        pass

    return CACHE['aircraft'], CACHE['source'], CACHE['feed_now']

# =========================================================================
# 3. MOTOR ASTROMÉTRICO Y KERNEL TELEMÉTRICO HARVARD
# =========================================================================
@app.route('/api/data')
def get_data():
    try:
        lat = float(request.args.get('lat', 41.6079))
        lon = float(request.args.get('lon', 2.2876))
        alt = float(request.args.get('alt', 145.0))
        user_lead_sec = float(request.args.get('lead', 7.5)) # Compensación de retardo ADS-B
        now_epoch = time.time()

        raw_ac, source_feed, feed_now_ts = get_live_aircraft(lat, lon)
        pipeline_lag = max(0.0, now_epoch - feed_now_ts) if feed_now_ts > 0 else 2.5

        t_now = ts.now()
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
        harvard_kernel_logs = []

        # Registro inicial de cabecera en el terminal
        obs_x, obs_y, obs_z = geodetic_to_ecef(lat, lon, alt)
        harvard_kernel_logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] [INIT_GEODETIC_WGS84] OBS_LAT:{lat:.5f} OBS_LON:{lon:.5f} ALT:{alt:.1f}m")
        harvard_kernel_logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] [ECEF_ORIGIN] X:{obs_x:.2f}m Y:{obs_y:.2f}m Z:{obs_z:.2f}m | ATM_P:{p_mbar:.1f}mb T:{t_c:.1f}C")
        harvard_kernel_logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] [ASTRO_TARGET_MOON] AZ:{moon_az0:.3f}° EL:{moon_alt0:.3f}° R_ANG:{moon_radius_deg:.4f}° DIST:{moon_dist_km:.1f}km")
        harvard_kernel_logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] [SYNC_ENGINE] FEED_LAG:{pipeline_lag:.2f}s USER_LEAD_OFFSET:+{user_lead_sec:.1f}s TOTAL_ADVANCE:+{pipeline_lag+user_lead_sec:.2f}s")

        for ac in raw_ac:
            raw_lat = safe_float(ac.get('lat'))
            raw_lon = safe_float(ac.get('lon'))
            track_val = safe_float(ac.get('track'))
            gs_val = safe_float(ac.get('gs'))

            if None in (raw_lat, raw_lon, track_val, gs_val) or gs_val <= 15.0:
                continue

            alt_geom = safe_float(ac.get('alt_geom'))
            alt_baro = safe_float(ac.get('alt_baro'))

            if alt_geom is not None and alt_geom > -1000.0:
                alt_ft = alt_geom
                alt_type = 'GNSS'
            elif alt_baro is not None and alt_baro > -1000.0:
                alt_ft = alt_baro
                alt_type = 'BARO'
            else:
                continue

            vr_raw = safe_float(ac.get('geom_rate', ac.get('baro_rate', 0.0)), default=0.0)
            vr_ms = vr_raw * 0.00508
            vr_fpm = int(vr_raw)
            callsign_str = str(ac.get('flight') or ac.get('hex', 'UNKNOWN')).strip()
            model_icao = str(ac.get('t', 'A320')).strip().upper()
            wingspan_m = get_wingspan(model_icao)

            alt_m = alt_ft * 0.3048
            speed_ms = gs_val * 0.514444

            # Extrapolación cinemática predictiva contra el retraso de Flightradar24
            seen_pos = safe_float(ac.get('seen_pos', ac.get('seen', 0.0)), default=0.0)
            total_lead_dt = max(0.0, pipeline_lag + seen_pos + user_lead_sec)
            
            ac_lat, ac_lon = propagate_geodetic_position(raw_lat, raw_lon, speed_ms, track_val, total_lead_dt)
            alt_m += vr_ms * total_lead_dt

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

                # Estimación analítica inicial
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

                if t_analytical < -5.0 and cur_sep > (b_rad * 4.0):
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

                # Ventana adaptativa con barrido seguro
                center_t = max(0.0, min(300.0, t_analytical if t_analytical > 0 else 0.0))
                scan_min = max(0.0, center_t - 25.0)
                scan_max = min(300.0, center_t + 25.0)
                
                best_t = 0.0
                min_sep, _, best_p_alt, best_p_range, best_lat, best_lon = eval_t(0.0)

                num_steps = 18
                for step in range(num_steps + 1):
                    t_cand = scan_min + (step / float(num_steps)) * (scan_max - scan_min)
                    sep_val, _, p_alt_val, p_range_val, cand_lat, cand_lon = eval_t(t_cand)
                    if sep_val < min_sep:
                        min_sep = sep_val
                        best_t = t_cand
                        best_p_alt = p_alt_val
                        best_p_range = p_range_val
                        best_lat, best_lon = cand_lat, cand_lon

                # Refinado de extrema precisión (Golden Section)
                a = max(0.0, best_t - 2.5)
                b = min(300.0, best_t + 2.5)
                phi = (1.0 + math.sqrt(5.0)) / 2.0
                resphi = 2.0 - phi

                x1 = a + resphi * (b - a)
                x2 = b - resphi * (b - a)
                f1, _, _, _, _, _ = eval_t(x1)
                f2, _, _, _, _, _ = eval_t(x2)

                for _ in range(14):
                    if f1 < f2:
                        b = x2
                        x2 = x1
                        f2 = f1
                        x1 = a + resphi * (b - a)
                        f1, _, _, _, _, _ = eval_t(x1)
                    else:
                        a = x1
                        x1 = x2
                        f1 = f2
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

            # Volcado de telemetría Harvard para aeronaves en aproximación
            if primary['min_sep'] < 4.0:
                harvard_kernel_logs.append(
                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}Z] [TCA_SOLVER_HIT] AC:{callsign_str} ({model_icao}) "
                    f"V_3D:[{ve:.1f},{vn:.1f},{vu:.1f}]m/s | SEP_MIN:{primary['min_sep']:.3f}° TCA_T:-{primary['tca_seconds']:.2f}s "
                    f"CHORD:{primary['position_descriptor']} SPAN:{wingspan_m}m ANG_SIZE:{primary['angular_size_arcsec']}\""
                )

            aircraft_results.append({
                'callsign': callsign_str,
                'model': model_icao,
                'wingspan_m': wingspan_m,
                'reg': str(ac.get('r', '')).strip().upper(),
                'lat': round(ac_lat, 5),
                'lon': round(ac_lon, 5),
                'alt_ft': int(alt_ft),
                'alt_type': alt_type,
                'track': float(round(track_val, 1)),
                'speed_kt': int(round(gs_val)),
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
            'pipeline_lag_sec': round(pipeline_lag, 2),
            'lead_used_sec': round(user_lead_sec, 2),
            'observer_altitude_used_m': alt,
            'kernel_logs': harvard_kernel_logs,
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
        return jsonify({'error': str(e), 'aircraft': [], 'kernel_logs': [f"[CRITICAL_ERROR] {str(e)}"]})

# =========================================================================
# 4. RUTAS DE INDEXACIÓN Y PLANTILLA HTML
# =========================================================================
@app.route('/google92a4c5b46b2ec0bf.html')
def google_verification():
    return 'google-site-verification: google92a4c5b46b2ec0bf.html'

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# =========================================================================
# 5. FRONTEND: CABINA TÁCTICA Y HARVARD ASTRO-TERMINAL
# =========================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="google-site-verification" content="google92a4c5b46b2ec0bf" />
    <title>Lunar Transit Radar PRO // Avionics & Harvard Astro-Terminal</title>
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #030712; color: #f1f5f9; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .map-container { height: calc(100dvh - 114px); width: 100%; border-radius: 12px; }
        .leaflet-container { background: #030712 !important; }
        
        .leaflet-div-icon {
            background: transparent !important;
            border: none !important;
        }

        .obs-target { display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; }
        .obs-ring { position: absolute; width: 34px; height: 34px; border-radius: 50%; background: rgba(6, 182, 212, 0.18); border: 2px solid #06b6d4; animation: pulse-ring 2.2s infinite ease-out; }
        .obs-dot { width: 10px; height: 10px; border-radius: 50%; background: #22d3ee; border: 2px solid #ffffff; box-shadow: 0 0 14px #06b6d4; z-index: 10; }
        @keyframes pulse-ring { 0% { transform: scale(0.5); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }

        .tca-target { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; }
        .tca-ring { position: absolute; width: 28px; height: 28px; border-radius: 50%; border: 2px dashed #ef4444; animation: tca-spin 3s linear infinite; }
        .tca-crosshair-h { position: absolute; width: 28px; height: 1.5px; background: rgba(239, 68, 68, 0.8); }
        .tca-crosshair-v { position: absolute; width: 1.5px; height: 28px; background: rgba(239, 68, 68, 0.8); }
        .tca-core { width: 6px; height: 6px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 12px #ef4444; z-index: 10; }
        @keyframes tca-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .avionics-panel { 
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94) 0%, rgba(6, 11, 25, 0.98) 100%);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.65);
        }

        .hud-card { 
            position: relative;
            background: rgba(10, 16, 31, 0.82);
            border: 1px solid rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(8px);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .hud-card:hover { 
            transform: translateY(-1px); 
            border-color: rgba(6, 182, 212, 0.7);
            box-shadow: 0 0 18px rgba(6, 182, 212, 0.15);
        }

        /* Terminal CRT Hacker / Harvard */
        .terminal-screen {
            background-color: #020617;
            background-image: radial-gradient(rgba(16, 185, 129, 0.1) 1px, transparent 0);
            background-size: 24px 24px;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #030712; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    </style>
</head>
<body class="p-1.5 md:p-2 flex flex-col h-[100dvh] overflow-hidden select-none">
    
    <!-- BARRA SUPERIOR DE CABINA -->
    <header class="avionics-panel px-3 py-1.5 rounded-xl mb-1.5 flex flex-wrap justify-between items-center gap-2 shadow-2xl">
        <div class="flex items-center gap-2">
            <span class="text-2xl animate-pulse">🌔</span>
            <div>
                <h1 class="text-xs font-black text-amber-400 tracking-wider">LUNAR RADAR PRO</h1>
                <div class="flex items-center gap-1 font-mono text-[8px]">
                    <span id="feed-badge" class="px-1 py-0.2 bg-emerald-950 text-emerald-300 border border-emerald-700/80 rounded font-bold">ONLINE</span>
                    <span class="px-1 py-0.2 bg-indigo-950 text-indigo-300 border border-indigo-700/80 rounded font-mono">DE421 JPL</span>
                </div>
            </div>
        </div>
        
        <!-- CONTROL DE SINCRONIZACIÓN DE LATENCIA (EXTRAPOLADOR CONTRA FLIGHTRADAR24) -->
        <div class="flex items-center gap-1 bg-slate-950 px-2 py-1 rounded-lg border border-slate-800">
            <span class="text-[10px] text-slate-400 font-bold">⚡ LEAD SYNC:</span>
            <button onclick="adjustLead(-0.5)" class="px-1.5 py-0.2 bg-slate-800 hover:bg-slate-700 text-cyan-300 font-bold rounded text-xs">-</button>
            <span id="lead-display" class="font-mono text-xs font-black text-amber-400 min-w-[50px] text-center">+7.5s</span>
            <button onclick="adjustLead(0.5)" class="px-1.5 py-0.2 bg-slate-800 hover:bg-slate-700 text-cyan-300 font-bold rounded text-xs">+</button>
        </div>

        <!-- CONMUTADOR: MAPA TÁCTICO VS TERMINAL HARVARD -->
        <div class="flex items-center gap-1 bg-slate-950 p-0.5 rounded-lg border border-slate-800">
            <button id="view-map-btn" onclick="setViewMode('map')" class="text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
                🗺️ Radar Map
            </button>
            <button id="view-terminal-btn" onclick="setViewMode('terminal')" class="text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-emerald-400 flex items-center gap-1">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping"></span>
                💻 Harvard Astro-Terminal
            </button>
        </div>

        <!-- HERRAMIENTAS Y UBICACIÓN -->
        <div class="flex items-center gap-1.5">
            <div class="bg-slate-950 p-0.5 rounded-lg border border-slate-800 flex">
                <button id="btn-flt-moon" onclick="setFilterMode('moon')" class="text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">🌔 Moon</button>
                <button id="btn-flt-sun" onclick="setFilterMode('sun')" class="text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-amber-300">☀️ Sun</button>
                <button id="btn-flt-all" onclick="setFilterMode('all')" class="text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-white">Dual</button>
            </div>

            <button id="voice-btn" onclick="toggleVoice()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition">🗣️</button>
            <button id="audio-btn" onclick="toggleAudio()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition">🔇</button>
            <button onclick="toggleSettingsModal()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold transition">⚙️</button>
            <button onclick="locateUser()" class="bg-cyan-600 hover:bg-cyan-500 text-white text-xs px-2.5 py-1.5 rounded-lg font-bold transition shadow-lg shadow-cyan-600/30">📍</button>
        </div>
    </header>

    <!-- CUERPO PRINCIPAL (VISTA MAPA O TERMINAL HARVARD) -->
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
                <span id="footer-vector-indicator" class="font-bold text-cyan-400">🌕──────</span>
                <span id="footer-astro-info">Optical Sight Vector Active</span>
            </div>
        </div>

        <!-- VISTA 2: HARVARD ASTRO-TERMINAL (CONSOLA HACKER EN VIVO) -->
        <div id="terminal-viewport" class="lg:col-span-3 rounded-xl overflow-hidden border border-emerald-900/60 relative shadow-2xl flex flex-col hidden terminal-screen">
            <div class="bg-slate-950 px-3 py-2 border-b border-emerald-900/50 flex justify-between items-center text-xs">
                <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span class="font-bold text-emerald-400 font-mono tracking-wider">HARVARD CENTER FOR ASTROPHYSICS // KINEMATICS ENGINE DE421</span>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="clearTerminal()" class="px-2 py-0.5 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white rounded text-[10px]">Clear</button>
                    <button onclick="copyTerminal()" class="px-2 py-0.5 bg-emerald-950 text-emerald-300 border border-emerald-800 rounded text-[10px]">Copy Buffer</button>
                </div>
            </div>
            
            <div id="terminal-output" class="p-3 font-mono text-[11px] text-emerald-400/90 overflow-y-auto flex-grow flex flex-col gap-1 leading-relaxed selection:bg-emerald-900 selection:text-white">
                <div class="text-slate-500">// HARVARD ASTRO-ENGINE INITIALIZED. STREAMING ORBITAL TELEMETRY...</div>
            </div>

            <div class="bg-slate-950/90 border-t border-emerald-900/40 p-2 flex justify-between items-center text-[10px] text-slate-400 font-mono">
                <span>MATRICES: WGS84 ECEF &bull; ENU &bull; GSS RES: &plusmn;3ms</span>
                <span id="terminal-clock" class="text-emerald-400 font-bold">2026-09-25 00:00:00 UTC</span>
            </div>
        </div>

        <!-- TELEMETRÍA LATERAL HUD -->
        <div class="avionics-panel rounded-xl p-2.5 overflow-y-auto flex flex-col gap-2 shadow-2xl max-h-[42vh] lg:max-h-full">
            <div class="flex justify-between items-center border-b border-slate-800 pb-1.5">
                <h2 class="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <span>📡 SECTOR TELEMETRY</span>
                    <span id="plane-count" class="bg-cyan-950 text-cyan-300 px-1.5 py-0.2 rounded-full text-[9px] border border-cyan-800 font-mono">0</span>
                </h2>
                <span id="filter-indicator" class="text-[9px] text-cyan-400 font-mono font-bold">TARGET: 🌕 MOON</span>
            </div>
            
            <div id="alerts-container" class="flex flex-col gap-1.5 overflow-y-auto">
                <div class="text-xs text-slate-500 text-center py-8">Scanning airspace for transit intercept...</div>
            </div>
        </div>
    </div>

    <!-- MODAL AJUSTES Y CALIBRACIÓN -->
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

            <div class="flex justify-between items-center pt-2 border-t border-slate-800">
                <button onclick="toggleMapLayer()" id="layer-btn" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 font-bold">
                    🗺️ Toggle Sat Map
                </button>
                <button onclick="toggleSettingsModal()" class="bg-cyan-600 hover:bg-cyan-500 text-xs text-white px-4 py-1.5 rounded-lg font-bold">
                    Guardar
                </button>
            </div>
        </div>
    </div>

    <script>
        const CARTO_KEY = 'cb1_2l65_1_3a8e83de8b889ec5e4e98278';

        let observerLat = parseFloat(localStorage.getItem('obs_lat') || 41.6079);
        let observerLon = parseFloat(localStorage.getItem('obs_lon') || 2.2876);
        let terrainElevationM = 145.0;
        let buildingOffsetM = parseFloat(localStorage.getItem('obs_building_m') || 0.0);
        let userLeadSec = parseFloat(localStorage.getItem('user_lead_sec') || 7.5);

        let activeFilter = 'moon';
        let viewMode = 'map';
        let isLocationLocked = true;
        let serverClockDelta = 0.0;
        let audioEnabled = false;
        let voiceEnabled = false;
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
            html: '<div class="obs-target"><div class="obs-ring"></div><div class="obs-dot"></div></div>',
            iconSize: [34, 34], iconAnchor: [17, 17]
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
            userLeadSec = Math.max(-5.0, Math.min(25.0, userLeadSec + delta));
            localStorage.setItem('user_lead_sec', userLeadSec.toFixed(1));
            document.getElementById('lead-display').innerText = (userLeadSec >= 0 ? '+' : '') + userLeadSec.toFixed(1) + 's';
            fetchData();
        }

        function setViewMode(mode) {
            viewMode = mode;
            const mapV = document.getElementById('radar-viewport');
            const termV = document.getElementById('terminal-viewport');
            const mapB = document.getElementById('view-map-btn');
            const termB = document.getElementById('view-terminal-btn');

            if (mode === 'map') {
                mapV.classList.remove('hidden');
                termV.classList.add('hidden');
                mapB.className = "text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800";
                termB.className = "text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-emerald-400 flex items-center gap-1";
                map.invalidateSize();
            } else {
                mapV.classList.add('hidden');
                termV.classList.remove('hidden');
                termB.className = "text-[10px] px-2.5 py-1 rounded font-bold bg-emerald-950 text-emerald-300 border border-emerald-800 flex items-center gap-1";
                mapB.className = "text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-cyan-300";
            }
        }

        function clearTerminal() {
            document.getElementById('terminal-output').innerHTML = '<div class="text-slate-500">// TERMINAL BUFFER CLEARED. LISTENING FOR HARVARD KERNEL PIPELINE...</div>';
        }

        function copyTerminal() {
            const txt = document.getElementById('terminal-output').innerText;
            navigator.clipboard.writeText(txt);
        }

        function drawRangeRings() {
            rangeCircles.forEach(c => map.removeLayer(c));
            rangeCircles = [];
            [15000, 30000, 50000].forEach((r) => {
                const circle = L.circle([observerLat, observerLon], {
                    radius: r, color: '#06b6d4', weight: 1, dashArray: '3, 7', fill: false, opacity: 0.22, interactive: false
                }).addTo(map);
                rangeCircles.push(circle);
            });
        }

        function toggleSettingsModal() {
            const m = document.getElementById('settings-modal');
            m.classList.toggle('hidden'); m.classList.toggle('flex');
        }

        function setFilterMode(mode) {
            activeFilter = mode;
            ['btn-flt-moon', 'btn-flt-sun', 'btn-flt-all'].forEach(id => {
                document.getElementById(id).className = "text-[10px] px-2.5 py-1 rounded font-bold text-slate-400 hover:text-white";
            });

            if (mode === 'moon') {
                document.getElementById('btn-flt-moon').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-cyan-950 text-cyan-300 border border-cyan-800";
                document.getElementById('filter-indicator').innerText = "TARGET: 🌕 MOON";
            } else if (mode === 'sun') {
                document.getElementById('btn-flt-sun').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-amber-950 text-amber-300 border border-amber-800";
                document.getElementById('filter-indicator').innerText = "TARGET: ☀️ SUN";
            } else {
                document.getElementById('btn-flt-all').className = "text-[10px] px-2.5 py-1 rounded font-bold bg-slate-800 text-cyan-300";
                document.getElementById('filter-indicator').innerText = "TARGET: DUAL (SUN/MOON)";
            }
            renderAstroVectors();
            updateHUDCountdowns();
        }

        function toggleVoice() {
            voiceEnabled = !voiceEnabled;
            const btn = document.getElementById('voice-btn');
            btn.className = voiceEnabled ? "bg-purple-600 text-white text-xs px-2 py-1.5 rounded-lg font-bold" : "bg-slate-800 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold";
            if (voiceEnabled) speak("Vocal radar active");
        }

        function speak(text) {
            if (!voiceEnabled || !('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(text);
            msg.rate = 1.05;
            window.speechSynthesis.speak(msg);
        }

        function toggleAudio() {
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('audio-btn');
            if (audioEnabled) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                if (audioContext.state === 'suspended') audioContext.resume();
                btn.innerText = "🔔"; btn.className = "bg-emerald-600 text-white text-xs px-2 py-1.5 rounded-lg font-bold";
                playChime();
            } else {
                btn.innerText = "🔇"; btn.className = "bg-slate-800 text-slate-300 text-xs px-2 py-1.5 rounded-lg border border-slate-700 font-bold";
            }
        }

        function playChime() {
            if (!audioEnabled || !audioContext) return;
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
            if (!audioEnabled || !audioContext) return;
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
            if (!audioEnabled || !audioContext) return;
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

        async function fetchTerrainElevation(lat, lon) {
            try {
                const res = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}`);
                const data = await res.json();
                if (data.elevation) terrainElevationM = parseFloat(data.elevation[0]);
            } catch (e) {}
        }

        function updateBuildingOffset(val) {
            buildingOffsetM = Math.max(0, parseFloat(val) || 0);
            localStorage.setItem('obs_building_m', buildingOffsetM.toString());
            fetchData();
        }

        async function saveAndSetObserverPos(lat, lon) {
            observerLat = lat; observerLon = lon;
            localStorage.setItem('obs_lat', lat.toString());
            localStorage.setItem('obs_lon', lon.toString());
            fetchTerrainElevation(lat, lon);
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

        async function fetchData() {
            try {
                const totalObserverAlt = terrainElevationM + buildingOffsetM;
                const res = await fetch(`/api/data?lat=${observerLat}&lon=${observerLon}&alt=${totalObserverAlt}&lead=${userLeadSec}`);
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
                document.getElementById('sun-coords').innerText = sunDataGlobal.visible ? `Sun: Az ${sunDataGlobal.azimuth}° | Alt +${sunDataGlobal.elevation}°` : `Sun Hidden (${sunDataGlobal.elevation}°)`;

                renderAstroVectors();

                // Actualizar buffer de la consola Harvard
                if (data.kernel_logs && data.kernel_logs.length > 0) {
                    const term = document.getElementById('terminal-output');
                    data.kernel_logs.forEach(l => {
                        const div = document.createElement('div');
                        div.innerText = l;
                        term.appendChild(div);
                    });
                    while (term.children.length > 300) term.removeChild(term.firstChild);
                    term.scrollTop = term.scrollHeight;
                }
                document.getElementById('terminal-clock').innerText = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

                activeAircraftData = data.aircraft || [];
                document.getElementById('plane-count').innerText = activeAircraftData.length;
                const currentCallsigns = new Set();
                const nowSec = (Date.now() / 1000.0) - serverClockDelta;

                activeAircraftData.forEach(plane => {
                    const cs = plane.callsign;
                    currentCallsigns.add(cs);

                    const target = activeFilter === 'sun' ? plane.sun : (activeFilter === 'moon' ? plane.moon : plane.primary);

                    if (tcaSmoothedEpochs[cs]) {
                        const delta = target.tca_epoch - tcaSmoothedEpochs[cs];
                        if (Math.abs(delta) < 2.0) {
                            tcaSmoothedEpochs[cs] = tcaSmoothedEpochs[cs] * 0.75 + target.tca_epoch * 0.25;
                        } else {
                            tcaSmoothedEpochs[cs] = target.tca_epoch;
                        }
                    } else {
                        tcaSmoothedEpochs[cs] = target.tca_epoch;
                    }

                    const color = getAltitudeColor(plane.alt_ft);
                    const isAnyTransit = (plane.moon.is_transit && moonDataGlobal.visible) || (plane.sun.is_transit && sunDataGlobal.visible);
                    const isAnyClose = (plane.moon.is_close && moonDataGlobal.visible) || (plane.sun.is_close && sunDataGlobal.visible);

                    const glow = isAnyTransit ? 'filter: drop-shadow(0 0 10px #ef4444);' : (isAnyClose ? 'filter: drop-shadow(0 0 6px #f59e0b);' : '');

                    const planeHtml = `
                        <div style="transform: rotate(${plane.track}deg); width: 24px; height: 24px; display:flex; align-items:center; justify-content:center; ${glow}">
                            <svg viewBox="0 0 24 24" width="22" height="22" fill="${color}">
                                <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
                            </svg>
                        </div>
                    `;

                    if (!planesState[cs]) {
                        const marker = L.marker([plane.lat, plane.lon], {
                            icon: L.divIcon({ className: '', html: planeHtml, iconSize: [24, 24], iconAnchor: [12, 12] })
                        }).addTo(map);

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
                        st.curLat = plane.lat;
                        st.curLon = plane.lon;
                        st.speedMs = plane.speed_ms;
                        st.trackRad = (plane.track * Math.PI) / 180.0;
                        st.lastSeenTime = nowSec;
                        st.data = plane;
                        st.marker.setIcon(L.divIcon({ className: '', html: planeHtml, iconSize: [24, 24], iconAnchor: [12, 12] }));
                    }
                });

                for (let cs in planesState) {
                    if (nowSec - planesState[cs].lastSeenTime > 12.0) {
                        map.removeLayer(planesState[cs].marker);
                        delete planesState[cs];
                        delete tcaSmoothedEpochs[cs];
                    }
                }

                updateHUDCountdowns();

            } catch (err) {}
        }

        let lastAnimTime = performance.now();
        function animateFrame(nowMs) {
            const dt = Math.min(0.08, Math.max(0.001, (nowMs - lastAnimTime) / 1000.0));
            lastAnimTime = nowMs;

            for (const cs in planesState) {
                const p = planesState[cs];
                const dist = p.speedMs * dt;
                p.curLat += (dist * Math.cos(p.trackRad)) / 111139.0;
                p.curLon += (dist * Math.sin(p.trackRad)) / (111139.0 * Math.cos(p.curLat * Math.PI / 180.0));
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
                <div class="flex items-center gap-2.5 bg-slate-950/85 p-2 rounded-lg border border-slate-800/90 mt-1">
                    <svg width="48" height="48" viewBox="0 0 48 48" class="shrink-0">
                        <circle cx="24" cy="24" r="19" fill="#030712" stroke="${bodyColor}" stroke-width="1.2" stroke-dasharray="3, 2"/>
                        <circle cx="24" cy="24" r="16" fill="${isSun ? '#451a03' : '#082f49'}" stroke="${bodyColor}" stroke-width="1.5"/>
                        <line x1="5" y1="24" x2="43" y2="24" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
                        <line x1="24" y1="5" x2="24" y2="43" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
                        <line x1="3" y1="${chordY.toFixed(1)}" x2="45" y2="${chordY.toFixed(1)}" stroke="${target.is_transit ? '#ef4444' : '#f59e0b'}" stroke-width="2.2" stroke-linecap="round"/>
                    </svg>
                    <div class="flex flex-col text-[10px] leading-tight">
                        <div class="flex items-center gap-1 font-mono">
                            <span class="text-slate-400 font-bold">CHORD:</span>
                            <span class="font-bold ${target.is_transit ? 'text-red-300' : 'text-amber-300'}">${target.position_descriptor}</span>
                        </div>
                        <span class="text-slate-400 font-mono mt-0.5">Offset: <b class="text-slate-200">${target.vertical_offset_deg > 0 ? '+' : ''}${target.vertical_offset_deg}°</b> (${target.vertical_body_diams} diam.)</span>
                        <span class="text-slate-400 font-mono">Disk Area: <b class="text-indigo-300">${target.disk_coverage_pct}%</b> (${target.transit_duration_s}s dur.)</span>
                    </div>
                </div>
            `;
        }

        function updateHUDCountdowns() {
            const container = document.getElementById('alerts-container');
            if (!activeAircraftData || activeAircraftData.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-500 text-center py-8">Scanning airspace for transit intercept...</div>';
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
                const smoothedEpoch = tcaSmoothedEpochs[plane.callsign] || target.tca_epoch;
                const remaining = Math.max(0.0, smoothedEpoch - now);

                const isTransit = target.is_transit && isTargetVisible;
                const isClose = target.is_close && isTargetVisible;
                const sym = target.target === 'sun' ? '☀️' : '🌕';

                const mins = Math.floor(remaining / 60);
                const secs = (remaining % 60).toFixed(1).padStart(4, '0');
                const timerStr = remaining > 0 ? `T-${mins.toString().padStart(2, '0')}:${secs}` : `TRANSITING!`;

                let cardBorder = 'border-slate-800/80 bg-slate-950/60';
                let tagHtml = `<span class="bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded text-[8px] font-bold font-mono">${sym} EN ROUTE</span>`;

                if (isTransit) {
                    cardBorder = 'border-red-500/90 bg-red-950/60';
                    tagHtml = `<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[9px] font-black font-mono animate-pulse">🎯 ${sym} TRANSIT LOCK</span>`;
                    
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
                    cardBorder = 'border-amber-500/80 bg-amber-950/50';
                    tagHtml = `<span class="bg-amber-600 text-white px-2 py-0.5 rounded text-[8px] font-bold font-mono">⚠️ ${sym} CLOSE PASS</span>`;
                    if (!priorityTransitFound && remaining > 0 && remaining < 180) {
                        priorityTransitFound = true;
                        renderTcaTrajectory(plane, target);
                    }
                }

                html += `
                    <div onclick="focusPlane('${plane.callsign}')" 
                         class="p-2.5 rounded-xl border ${cardBorder} text-xs flex flex-col gap-1 transition cursor-pointer hud-card">
                        
                        <div class="flex justify-between items-center border-b border-slate-800/80 pb-1.5">
                            <div class="flex items-center gap-1.5 font-mono">
                                <span class="font-black text-sm tracking-wide ${isTransit ? 'text-red-400' : 'text-slate-100'}">${plane.callsign}</span>
                                <span class="px-1.5 py-0.2 bg-slate-900 text-cyan-300 border border-slate-700 rounded text-[9px] font-bold">${plane.model}</span>
                                <span class="text-[8px] px-1 bg-slate-900 text-slate-400 border border-slate-700 rounded">${plane.alt_type}</span>
                            </div>
                            ${tagHtml}
                        </div>
                        
                        <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] font-mono text-slate-300 mt-0.5">
                            <div><span class="text-slate-500">ALT:</span> <b>${plane.alt_ft.toLocaleString()} ft</b></div>
                            <div><span class="text-slate-500">V/S:</span> <b>${plane.vr_fpm > 0 ? '+' : ''}${plane.vr_fpm} ft/m</b></div>
                            <div><span class="text-slate-500">SPD:</span> <b>${plane.speed_kt} kt</b></div>
                            <div><span class="text-slate-500">RNG:</span> <b>${plane.distance_km} km</b></div>
                            <div><span class="text-slate-500">HDG:</span> <b>${plane.track}°</b></div>
                            <div><span class="text-slate-500">TCA:</span> <b class="${isTransit ? 'text-red-400 font-black' : (isClose ? 'text-amber-300 font-bold' : 'text-cyan-300')}">${timerStr}</b></div>
                            <div class="col-span-2"><span class="text-slate-500">SPAN / SIZE:</span> <b class="text-indigo-300">${plane.wingspan_m}m &bull; ${target.angular_size_arcsec}"</b></div>
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
        document.getElementById('building-offset').value = buildingOffsetM.toString();

        drawRangeRings();
        fetchData();
        fetchTerrainElevation(observerLat, observerLon);
        setInterval(fetchData, 1800);
        setInterval(updateHUDCountdowns, 100);
        requestAnimationFrame(animateFrame);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print(f" [OK] LUNAR TRANSIT RADAR PRO // HARVARD ENGINE")
    print(f" [OK] Predictive Lead Extrapolator: ACTIVE")
    print(f" [OK] Harvard Terminal Stream: ACTIVE")
    print(f" [OK] Server Online on port: {port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

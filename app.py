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
    return WINGSPANS.get(model_icao, 35.8)

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
# 2. INGESTA MULTI-FUENTE BLINDADA CON BUFFER DE MEMORIA DE VUELO
# =========================================================================
HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*'
}

PROVIDERS = [
    {'name': 'adsb.fi', 'type': 'readsb', 'url': 'https://opendata.adsb.fi/api/v2/lat/{lat:.4f}/lon/{lon:.4f}/dist/80', 'next_ok': 0.0},
    {'name': 'adsb.lol', 'type': 'readsb', 'url': 'https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/80', 'next_ok': 0.0},
    {'name': 'adsb.one', 'type': 'readsb', 'url': 'https://api.adsb.one/v2/point/{lat:.4f}/{lon:.4f}/80', 'next_ok': 0.0},
    {'name': 'opensky', 'type': 'opensky', 'url': 'https://opensky-network.org/api/states/all?lamin={lamin:.4f}&lomin={lomin:.4f}&lamax={lamax:.4f}&lomax={lomax:.4f}', 'next_ok': 0.0}
]

TRACK_MEMORY = {}
_TRACK_LOCK = threading.Lock()
HTTP_SESSION = requests.Session()
_EVENT_CACHE = {}
LAST_PROVIDER_USED = 'online'

def parse_opensky_response(data, ref_time):
    states = data.get('states') or []
    normalized = []
    for s in states:
        if not isinstance(s, list) or len(s) < 11 or s[8]:  # on_ground == True
            continue
        lon_val, lat_val = s[5], s[6]
        if lat_val is None or lon_val is None:
            continue
        
        baro_alt_m = s[7]
        geom_alt_m = s[13] if len(s) > 13 else None
        alt_baro = round(baro_alt_m * 3.28084) if baro_alt_m is not None else None
        alt_geom = round(geom_alt_m * 3.28084) if geom_alt_m is not None else None
        if alt_baro is None and alt_geom is None:
            continue
            
        vel_ms = s[9] or 0.0
        gs_kt = round(vel_ms * 1.94384, 1)
        track_deg = float(s[10]) if s[10] is not None else 0.0
        vr_ms = s[11] if s[11] is not None else 0.0
        vr_fpm = round(vr_ms * 196.85)

        hex_code = str(s[0]).lower().strip()
        callsign = str(s[1]).strip() if s[1] else hex_code.upper()
        pos_t = float(s[3] or s[4] or ref_time)

        normalized.append({
            'hex': hex_code,
            'flight': callsign,
            'lat': float(lat_val),
            'lon': float(lon_val),
            'alt_baro': alt_baro,
            'alt_geom': alt_geom,
            'track': track_deg,
            'gs': gs_kt,
            'baro_rate': vr_fpm,
            'geom_rate': vr_fpm,
            't': 'A320',
            '_pos_t': pos_t
        })
    return normalized

def _query_provider(provider, lat, lon):
    now = time.time()
    try:
        if provider['type'] == 'readsb':
            req_url = provider['url'].format(lat=lat, lon=lon)
        else:
            d_lat = 1.35
            d_lon = 1.35 / max(0.2, math.cos(math.radians(lat)))
            req_url = provider['url'].format(
                lamin=lat - d_lat, lamax=lat + d_lat,
                lomin=lon - d_lon, lomax=lon + d_lon
            )

        resp = HTTP_SESSION.get(req_url, headers=HTTP_HEADERS, timeout=(2.0, 3.8))
        if resp.status_code != 200:
            provider['next_ok'] = now + (25.0 if resp.status_code in (403, 429) else 8.0)
            return None

        json_data = resp.json()
        if provider['type'] == 'readsb':
            ac_list = json_data.get('ac') or json_data.get('aircraft') or []
            if not isinstance(ac_list, list):
                return None
            for ac in ac_list:
                sp = float(ac.get('seen_pos', ac.get('seen', 0.0)) or 0.0)
                ac['_pos_t'] = now - max(0.0, sp)
            return ac_list
        else:
            return parse_opensky_response(json_data, now)

    except Exception:
        provider['next_ok'] = now + 10.0
        return None

def get_live_aircraft(cur_lat, cur_lon):
    global TRACK_MEMORY, LAST_PROVIDER_USED
    now = time.time()
    sorted_providers = sorted(PROVIDERS, key=lambda p: p['next_ok'])
    fresh_batch = None

    for p in sorted_providers:
        if p['next_ok'] > now:
            continue
        res = _query_provider(p, cur_lat, cur_lon)
        if res is not None and len(res) > 0:
            fresh_batch = res
            LAST_PROVIDER_USED = p['name']
            p['next_ok'] = now + 1.5
            print(f"[ADS-B] {len(res)} aeronaves cargadas desde {LAST_PROVIDER_USED}")
            break

    with _TRACK_LOCK:
        if fresh_batch:
            for ac in fresh_batch:
                h = ac.get('hex') or ac.get('flight')
                if h:
                    TRACK_MEMORY[h] = ac

        # Purgado de aeronaves únicamente si no se reciben datos tras 60 segundos
        purged = {}
        for h, ac in TRACK_MEMORY.items():
            if (now - ac.get('_pos_t', now)) < 60.0:
                purged[h] = ac
        TRACK_MEMORY = purged

        active_list = list(TRACK_MEMORY.values())
        return active_list, LAST_PROVIDER_USED, 0.0

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
        
        raw_ac, source_feed, feed_age = get_live_aircraft(lat, lon)
        now_epoch = time.time()
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
            ck = (body_obj is moon, is_vis, round(lat, 2), round(lon, 2))
            c = _EVENT_CACHE.get(ck)
            if c and c[2] > now_epoch + 1.0 and now_epoch - c[3] < 120.0:
                return c[0], c[1], max(0, int(c[2] - now_epoch))
            try:
                t_end = ts.from_datetime(datetime.now(timezone.utc) + timedelta(hours=36))
                f_rs = almanac.risings_and_settings(eph, body_obj, topos_loc)
                times_rs, events_rs = almanac.find_discrete(t_now, t_end, f_rs)
                now_utc = datetime.now(timezone.utc)
                for t_e, ev in zip(times_rs, events_rs):
                    dt_e = t_e.utc_datetime()
                    if is_vis and ev == 0:
                        res = ("SET", dt_e.strftime('%H:%M UTC'), max(0, int((dt_e - now_utc).total_seconds())))
                        _EVENT_CACHE[ck] = (res[0], res[1], now_epoch + res[2], now_epoch)
                        return res
                    elif not is_vis and ev == 1:
                        res = ("RISE", dt_e.strftime('%H:%M UTC'), max(0, int((dt_e - now_utc).total_seconds())))
                        _EVENT_CACHE[ck] = (res[0], res[1], now_epoch + res[2], now_epoch)
                        return res
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

            pos_age = max(0.0, now_epoch - ac.get('_pos_t', now_epoch))
            if pos_age > 60.0:
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
                    'vertical_dir_text': "Above" if vert_offset_deg > 0 else "Below",
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
            'feed_age_s': round(feed_age, 2),
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
# 4. RUTAS ESTÁTICAS Y CONTROL DE CABECERAS
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
# 5. FRONTEND TÁCTICO CON SILUETAS FR24 Y DIAGRAMAS COMPLETOS
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
        :root {
            --bg: #070b12; --s1: #0c121d; --s2: #111a29;
            --line: rgba(148,163,184,.14); --line2: rgba(148,163,184,.28);
            --tx: #e6edf7; --mut: #8696ae; --moon: #7dd3fc; --sun: #fbbf24;
            --bad: #f87171; --ok: #34d399; --acc: #38bdf8;
        }
        body { 
            background-color: var(--bg); 
            color: var(--tx); 
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .font-mono, .tabular-nums { font-family: 'JetBrains Mono', monospace; }
        .map-container { flex: 1 1 auto; height: calc(100dvh - 90px); width: 100%; border-radius: 12px; }
        .leaflet-container { background: #05080e !important; }
        .leaflet-div-icon { background: transparent !important; border: none !important; }

        .obs-target { position: relative; display: flex; align-items: center; justify-content: center; width: 44px; height: 44px; pointer-events: none; }
        .obs-sweep { position: absolute; width: 44px; height: 44px; border-radius: 50%; background: conic-gradient(from 0deg, rgba(6, 182, 212, 0.4) 0deg, transparent 60deg, transparent 360deg); animation: radar-sweep 3s linear infinite; }
        .obs-ring { position: absolute; width: 32px; height: 32px; border-radius: 50%; background: rgba(6, 182, 212, 0.15); border: 1.5px solid #06b6d4; animation: pulse-ring 2.2s infinite ease-out; }
        .obs-dot { width: 8px; height: 8px; border-radius: 50%; background: #22d3ee; border: 2px solid #ffffff; box-shadow: 0 0 12px #06b6d4; z-index: 10; }
        @keyframes pulse-ring { 0% { transform: scale(0.6); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }
        @keyframes radar-sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .tca-target { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; }
        .tca-ring { position: absolute; width: 28px; height: 28px; border-radius: 50%; border: 2px dashed #ef4444; animation: tca-spin 3s linear infinite; }
        .tca-crosshair-h { position: absolute; width: 28px; height: 1.5px; background: rgba(239, 68, 68, 0.8); }
        .tca-crosshair-v { position: absolute; width: 1.5px; height: 28px; background: rgba(239, 68, 68, 0.8); }
        .tca-core { width: 6px; height: 6px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 12px #ef4444; z-index: 10; }
        @keyframes tca-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .topbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 12px; margin-bottom: 6px; background: var(--s1); border: 1px solid var(--line); border-radius: 12px; }
        .chips { display: flex; gap: 4px; }
        .chip { font: 600 10px/1 'JetBrains Mono', monospace; padding: 3px 6px; border-radius: 5px; border: 1px solid var(--line2); color: var(--mut); background: rgba(255,255,255,.02); }
        #feed-badge { color: var(--ok); border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
        #feed-badge[data-state="bad"] { color: var(--bad); border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.08); }
        
        .body-card { display: flex; align-items: center; gap: 8px; padding: 5px 9px; border-radius: 9px; border: 1px solid var(--line); background: var(--s2); cursor: pointer; opacity: .6; transition: .15s; }
        .body-card.on { opacity: 1; }
        #moon-status-card.on { border-color: rgba(125,211,252,.45); }
        #sun-status-card.on { border-color: rgba(251,191,36,.45); }
        
        .stepper, .seg { display: flex; align-items: center; gap: 2px; padding: 3px; border: 1px solid var(--line); border-radius: 9px; background: var(--s2); }
        .stepper button, .seg button { padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; color: var(--mut); }
        .seg button.on { color: var(--tx); background: rgba(56,189,248,.16); box-shadow: inset 0 0 0 1px rgba(56,189,248,.4); }
        .icon-btn { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 8px; border: 1px solid var(--line); background: var(--s2); color: var(--mut); }
        .icon-btn.on { color: var(--acc); border-color: rgba(56,189,248,.4); }
        .hud-card { background: var(--s2); border: 1px solid var(--line); border-radius: 10px; }
        .glow-red { box-shadow: 0 0 0 1px rgba(248,113,113,.6), 0 0 18px rgba(248,113,113,.2) !important; }
        .glow-amber { box-shadow: 0 0 0 1px rgba(245,158,11,.5), 0 0 14px rgba(245,158,11,.15) !important; }
    </style>
</head>
<body class="p-1.5 md:p-2 flex flex-col h-[100dvh] overflow-hidden select-none">
    
    <header class="topbar">
        <div class="flex items-center gap-2">
            <svg class="w-6 h-6 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="m2 15 19-7"/></svg>
            <div>
                <h1 class="text-xs font-bold uppercase tracking-wider text-slate-100">Lunar Transit Radar PRO</h1>
                <div class="chips"><span id="feed-badge" class="chip">ONLINE</span><span class="chip">DE421 JPL</span><span id="sync-badge" class="chip">SYNC …</span></div>
            </div>
        </div>
        <div class="flex items-center gap-2">
            <div id="moon-status-card" class="body-card on" onclick="setFilterMode('moon')">
                <div id="header-moon-icon" class="w-5 h-5 flex items-center justify-center"></div>
                <div class="flex flex-col text-[11px] font-mono leading-tight"><span id="moon-coords" class="text-cyan-300">Moon: Az --° | Alt --°</span><small class="text-[9px] text-slate-400">LUNAR VECTOR</small></div>
                <span id="moon-badge" class="chip">--:--</span>
            </div>
            <div id="sun-status-card" class="body-card" onclick="setFilterMode('sun')">
                <div id="header-sun-icon" class="w-5 h-5 flex items-center justify-center"></div>
                <div class="flex flex-col text-[11px] font-mono leading-tight"><span id="sun-coords" class="text-amber-300">Sun: Az --° | Alt --°</span><small class="text-[9px] text-slate-400">SOLAR SIGHT LINE</small></div>
                <span id="sun-badge" class="chip">--:--</span>
            </div>
        </div>
        <div class="flex items-center gap-2">
            <div class="stepper"><span class="text-[10px] font-mono text-slate-400 px-1 font-bold">LEAD</span><button onclick="adjustLead(-0.5)">−</button><span id="lead-display" class="font-mono text-xs text-amber-400 font-bold px-1">+6.0s</span><button onclick="adjustLead(0.5)">+</button></div>
            <div class="seg">
                <button id="btn-flt-moon" class="on" onclick="setFilterMode('moon')">Moon</button>
                <button id="btn-flt-sun" onclick="setFilterMode('sun')">Sun</button>
                <button id="btn-flt-all" onclick="setFilterMode('all')">Dual</button>
            </div>
            <div class="hidden md:flex flex-col items-end text-[11px] font-mono text-slate-400 leading-tight"><span id="obs-coords">41.6079, 2.2876</span><span id="obs-alt-badge">145 m ASL</span></div>
            <button id="voice-btn" class="icon-btn on" onclick="toggleVoice()" title="Voz"><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg></button>
            <button id="audio-btn" class="icon-btn on" onclick="toggleAudio()" title="Sonido"><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a2 2 0 0 0 3.4 0"/></svg></button>
            <button class="icon-btn" onclick="toggleSettingsModal()" title="Ajustes"><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button>
            <button class="icon-btn" onclick="locateUser()" title="Mi ubicación"><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg></button>
        </div>
    </header>

    <!-- CUERPO PRINCIPAL -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-1.5 flex-grow overflow-hidden relative">
        <div id="radar-viewport" class="lg:col-span-3 rounded-xl overflow-hidden border border-slate-800 relative flex flex-col">
            <div id="map" class="map-container"></div>
            
            <div class="hidden sm:flex absolute top-2.5 right-2.5 z-[1000] bg-slate-950/90 backdrop-blur p-2 rounded-xl text-[10px] font-mono border border-slate-800 flex-col gap-1 shadow-2xl">
                <span class="font-bold text-slate-400 uppercase text-[9px] mb-0.5">FLIGHT LEVEL</span>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#ef4444]"></span> &lt; 3k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#f97316]"></span> 3k - 10k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#eab308]"></span> 10k - 18k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#22c55e]"></span> 18k - 28k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#06b6d4]"></span> 28k - 36k ft</div>
                <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-[#a855f7]"></span> &gt; 36k ft</div>
            </div>

            <div class="absolute bottom-2.5 left-2.5 z-[1000] bg-slate-950/90 backdrop-blur px-2.5 py-1.5 rounded-lg text-[11px] font-mono border border-slate-800 text-slate-300 flex items-center gap-2">
                <span id="footer-vector-indicator" class="font-bold text-cyan-400">━━━━</span>
                <span id="footer-astro-info">Optical Sight Vector Active</span>
            </div>
        </div>

        <div class="bg-slate-900/60 rounded-xl p-2.5 overflow-y-auto flex flex-col gap-2 border border-slate-800 shadow-2xl max-h-[42vh] lg:max-h-full">
            <div class="flex justify-between items-center border-b border-slate-800 pb-1.5">
                <h2 class="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <span>Sector Telemetry</span>
                    <span id="plane-count" class="bg-cyan-950 text-cyan-300 px-1.5 py-0.2 rounded-full text-[10px] border border-cyan-800 font-mono tabular-nums">0</span>
                </h2>
                <span id="filter-indicator" class="text-[9px] text-cyan-400 font-mono font-bold tracking-wider">TARGET · MOON</span>
            </div>
            
            <div id="alerts-container" class="flex flex-col gap-2 overflow-y-auto">
                <div class="text-xs text-slate-500 text-center py-8 font-mono">Scanning airspace for transit intercept...</div>
            </div>
        </div>
    </div>

    <!-- MODAL CONFIGURACIÓN -->
    <div id="settings-modal" class="fixed inset-0 z-[2000] bg-black/75 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-4 max-w-sm w-full shadow-2xl flex flex-col gap-3 font-mono">
            <div class="flex justify-between items-center border-b border-slate-800 pb-2">
                <h3 class="font-bold text-sm text-cyan-400">⚙️ Calibración de Observación</h3>
                <button onclick="toggleSettingsModal()" class="text-slate-400 hover:text-white font-bold">✕</button>
            </div>
            
            <div class="flex flex-col gap-2 bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div class="flex justify-between items-center">
                    <span class="text-[10px] text-slate-300 font-bold uppercase">⛰️ Terreno (MSL)</span>
                    <button onclick="detectTerrainAuto()" class="text-[9px] px-2 py-0.5 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 rounded border border-cyan-800">Auto-Detect</button>
                </div>
                <div class="flex items-center gap-2">
                    <input id="terrain-elevation-input" type="number" step="1" oninput="calculateModalTotalAlt()" class="w-full bg-slate-900 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold tabular-nums">
                    <span class="text-slate-400 text-xs">m</span>
                </div>

                <div class="flex justify-between items-center mt-1">
                    <span class="text-[10px] text-slate-300 font-bold uppercase">🏢 Trípode / Azotea (Offset)</span>
                </div>
                <div class="flex items-center gap-2">
                    <input id="building-offset" type="number" step="0.5" min="0" max="500" oninput="calculateModalTotalAlt()" class="w-full bg-slate-900 text-cyan-300 text-xs px-2 py-1.5 rounded border border-slate-700 font-bold tabular-nums">
                    <span class="text-slate-400 text-xs">m</span>
                </div>

                <div class="flex justify-between items-center pt-2 border-t border-slate-800 text-xs">
                    <span class="text-slate-400 font-bold">Cota Total:</span>
                    <span id="total-alt-display" class="text-emerald-400 font-black tabular-nums">145m</span>
                </div>
            </div>

            <div class="flex justify-between items-center bg-slate-950 p-2 rounded-xl border border-slate-800">
                <span class="text-[10px] text-slate-300 font-bold">🔊 Audio Radar</span>
                <button onclick="testAudioSystem()" class="px-2 py-1 bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 rounded text-[10px] font-bold">🔔 Probar</button>
            </div>

            <div class="flex justify-between items-center pt-2 border-t border-slate-800">
                <button onclick="toggleMapLayer()" id="layer-btn" class="bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 font-bold">🗺️ Toggle Sat</button>
                <button onclick="saveAltitudeSettings()" class="bg-cyan-600 hover:bg-cyan-500 text-xs text-white px-4 py-1.5 rounded-lg font-bold">Guardar</button>
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
        let serverClockDelta = 0.0;
        let audioEnabled = true, voiceEnabled = true;
        let audioContext = null, lastBeepedFlight = '';
        
        let detectedTransits = new Set(), spokenCountdowns = new Set();
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

        function initAndUnlockAudio() {
            if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext && audioContext.state === 'suspended') audioContext.resume();
        }
        ['click', 'touchstart', 'pointerdown', 'keydown'].forEach(ev => {
            window.addEventListener(ev, initAndUnlockAudio, { passive: true });
        });

        map = L.map('map', { preferCanvas: true, zoomControl: false }).setView([observerLat, observerLon], 10);
        L.control.zoom({ position: 'bottomright' }).addTo(map);

        const cartoDarkLayer = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png?key=${CARTO_KEY}`, {
            subdomains: 'abcd', maxZoom: 20, attribution: '&copy; CARTO &bull; ADS-B OpenData'
        });
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 18, attribution: 'Esri &bull; ADS-B OpenData'
        });

        currentBaseTileLayer = cartoDarkLayer;
        currentBaseTileLayer.addTo(map);

        function toggleMapLayer() {
            map.removeLayer(currentBaseTileLayer);
            isSatelliteMode = !isSatelliteMode;
            currentBaseTileLayer = isSatelliteMode ? satelliteLayer : cartoDarkLayer;
            document.getElementById('layer-btn').innerText = isSatelliteMode ? "🗺️ Basemap: Satellite" : "🗺️ Basemap: Dark";
            currentBaseTileLayer.addTo(map);
        }

        const obsCustomIcon = L.divIcon({
            className: '',
            html: '<div class="obs-target"><div class="obs-sweep"></div><div class="obs-ring"></div><div class="obs-dot"></div></div>',
            iconSize: [44, 44], iconAnchor: [22, 22]
        });

        obsMarker = L.marker([observerLat, observerLon], { draggable: true, icon: obsCustomIcon }).addTo(map);

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
                    radius: r, color: '#06b6d4', weight: 1, dashArray: '4, 8', fill: false, opacity: 0.25, interactive: false
                }).addTo(map);
                rangeCircles.push(circle);
            });
        }

        function toggleSettingsModal() {
            const m = document.getElementById('settings-modal');
            m.classList.toggle('hidden'); m.classList.toggle('flex');
            if (!m.classList.contains('hidden')) updateAltitudeDisplay();
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
            speak("Radar audio and vocal alerts active");
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
                    gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.3);
                    osc.connect(gain); gain.connect(audioContext.destination);
                    osc.start(now + i * 0.08); osc.stop(now + i * 0.08 + 0.3);
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
                gain.gain.setValueAtTime(0.15, audioContext.currentTime);
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

        // =====================================================================
        // SILUETAS AERONÁUTICAS PROFESIONALES ESTILO FLIGHTRADAR24 (FR24)
        // =====================================================================
        function planeIconHtml(plane) {
            const color = getAltitudeColor(plane.alt_ft);
            const anyT = (plane.moon.is_transit && moonDataGlobal.visible) || (plane.sun.is_transit && sunDataGlobal.visible);
            const anyC = (plane.moon.is_close && moonDataGlobal.visible) || (plane.sun.is_close && sunDataGlobal.visible);
            
            let filterStyle = 'filter: drop-shadow(0 2px 4px rgba(0,0,0,0.85));';
            let alertRing = '';
            let labelBadge = '';

            if (anyT) {
                filterStyle = 'filter: drop-shadow(0 0 8px #ef4444) drop-shadow(0 2px 4px rgba(0,0,0,0.9));';
                alertRing = '<div class="absolute inset-0 rounded-full border-2 border-red-500 animate-ping opacity-75 pointer-events-none"></div>';
                labelBadge = `<div style="position:absolute;left:34px;top:8px;white-space:nowrap;pointer-events:none;font:700 10px 'JetBrains Mono',monospace;padding:1px 6px;border-radius:4px;background:rgba(15,23,42,0.95);border:1px solid #ef4444;color:#fca5a5;box-shadow:0 0 10px rgba(239,68,68,0.5)">${plane.callsign}</div>`;
            } else if (anyC) {
                filterStyle = 'filter: drop-shadow(0 0 6px #f59e0b) drop-shadow(0 2px 4px rgba(0,0,0,0.9));';
                labelBadge = `<div style="position:absolute;left:34px;top:8px;white-space:nowrap;pointer-events:none;font:700 10px 'JetBrains Mono',monospace;padding:1px 6px;border-radius:4px;background:rgba(15,23,42,0.95);border:1px solid #f59e0b;color:#fcd34d">${plane.callsign}</div>`;
            }

            return `
                <div style="position:relative;width:40px;height:40px;display:flex;align-items:center;justify-content:center;">
                    ${alertRing}
                    <svg viewBox="0 0 48 48" width="34" height="34" style="transform: rotate(${plane.track}deg); transform-origin: 50% 50%; ${filterStyle}; transition: transform 0.15s linear;">
                        <!-- Silueta exacta comercial: morro, fuselaje, motores gemelos, alas en flecha con winglets y cola -->
                        <path fill="${color}" stroke="#030712" stroke-width="1.3" stroke-linejoin="round" d="
                            M 24 3.5
                            C 25.2 3.5, 26.2 5.5, 26.2 11
                            L 26.2 15
                            L 28.6 16.5 L 28.6 20.8 L 31.4 20.8 L 31.4 17.8
                            L 44.5 24.2 L 45.2 22.8 L 45.8 23.4 L 44.6 26.2
                            L 26.5 22.4
                            L 26.2 34.5
                            L 34.8 39.8 L 34.8 41.5 L 25.6 40.8
                            L 24 44.5
                            L 22.4 40.8 L 13.2 41.5 L 13.2 39.8
                            L 21.8 34.5
                            L 21.5 22.4
                            L 3.4 26.2 L 2.2 23.4 L 2.8 22.8 L 3.5 24.2
                            L 16.6 17.8 L 16.6 20.8 L 19.4 20.8 L 19.4 16.5
                            L 21.8 15
                            L 21.8 11
                            C 21.8 5.5, 22.8 3.5, 24 3.5 Z
                        "/>
                        <path d="M 24 6 L 24 13" stroke="#030712" stroke-width="0.8" opacity="0.6"/>
                    </svg>
                    ${labelBadge}
                </div>
            `;
        }

        function getRealisticMoonSvgHtml(size = 38) {
            return `
                <div style="position:relative; width:${size}px; height:${size}px; display:flex; align-items:center; justify-content:center; filter: drop-shadow(0 0 8px rgba(56,189,248,0.7));">
                    <svg viewBox="0 0 100 100" width="${size}" height="${size}">
                        <circle cx="50" cy="50" r="46" fill="#cbd5e1" stroke="#38bdf8" stroke-width="1.5"/>
                        <circle cx="38" cy="35" r="10" fill="#64748b" opacity="0.6"/>
                        <circle cx="62" cy="38" r="8" fill="#64748b" opacity="0.6"/>
                        <ellipse cx="60" cy="62" rx="9" ry="8" fill="#64748b" opacity="0.6"/>
                        <circle cx="36" cy="64" r="6" fill="#64748b" opacity="0.6"/>
                    </svg>
                </div>
            `;
        }

        function getRealisticSunSvgHtml(size = 38) {
            return `
                <div style="position:relative; width:${size}px; height:${size}px; display:flex; align-items:center; justify-content:center; filter: drop-shadow(0 0 10px rgba(245,158,11,0.85));">
                    <svg viewBox="0 0 100 100" width="${size}" height="${size}">
                        <circle cx="50" cy="50" r="44" fill="none" stroke="#fef08a" stroke-width="1.5" stroke-dasharray="4, 4" opacity="0.8" />
                        <circle cx="50" cy="50" r="38" fill="#f59e0b" stroke="#fde047" stroke-width="2" />
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
                if (moonLine) moonLine.setLatLngs([[observerLat, observerLon], endM]);
                else moonLine = L.polyline([[observerLat, observerLon], endM], { color: '#38bdf8', weight: 2.2, dashArray: '5, 8', opacity: 0.95 }).addTo(map);
                
                if (moonIconMarker) moonIconMarker.setLatLng(endM);
                else moonIconMarker = L.marker(endM, { icon: L.divIcon({ html: getRealisticMoonSvgHtml(36), iconSize: [36, 36], iconAnchor: [18, 18] }) }).addTo(map);
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
                if (sunLine) sunLine.setLatLngs([[observerLat, observerLon], endS]);
                else sunLine = L.polyline([[observerLat, observerLon], endS], { color: '#f59e0b', weight: 2.2, dashArray: '5, 8', opacity: 0.95 }).addTo(map);

                if (sunIconMarker) sunIconMarker.setLatLng(endS);
                else sunIconMarker = L.marker(endS, { icon: L.divIcon({ html: getRealisticSunSvgHtml(36), iconSize: [36, 36], iconAnchor: [18, 18] }) }).addTo(map);
            } else {
                if (sunLine) { map.removeLayer(sunLine); sunLine = null; }
                if (sunIconMarker) { map.removeLayer(sunIconMarker); sunIconMarker = null; }
            }
        }

        async function syncClock() {
            try {
                const t0 = Date.now() / 1000.0;
                const r = await fetch('/api/time', { cache: 'no-store' });
                const j = await r.json();
                const t1 = Date.now() / 1000.0;
                serverClockDelta = (t0 + t1) / 2.0 - j.t;
                document.getElementById('sync-badge').innerText = `SYNC ±${Math.max(1, Math.round((t1 - t0) * 500))}ms`;
            } catch (e) {}
        }
        const syncedNow = () => (Date.now() / 1000.0) - serverClockDelta;

        function projectLL(lat, lon, speedMs, trackRad, dt) {
            const d = speedMs * dt;
            return [
                lat + (d * Math.cos(trackRad)) / 111139.0,
                lon + (d * Math.sin(trackRad)) / (111139.0 * Math.cos(lat * Math.PI / 180.0))
            ];
        }

        function popupHtml(p) {
            const arrow = p.vr_fpm > 300 ? '▲' : (p.vr_fpm < -300 ? '▼' : '►');
            return `<div class="font-mono text-xs">
                <b class="text-cyan-400 font-bold">${p.callsign}</b> · ${p.model}<br>
                ALT ${p.alt_ft.toLocaleString()} ft ${p.alt_type} ${arrow} ${p.vr_fpm} ft/m<br>
                GS ${p.speed_kt} kt · HDG ${p.track}°<br>DIST ${p.distance_km} km
            </div>`;
        }

        // =====================================================================
        // DIAGRAMA DE CORTE DE CUERDA (TRÁNSITO LUNAR Y SOLAR)
        // =====================================================================
        function renderTransitDiscDiagram(target) {
            const isSun = target.target === 'sun';
            const bodyColor = isSun ? '#fbbf24' : '#38bdf8';
            const bodyRad = isSun ? (sunDataGlobal.radius_deg || 0.26) : (moonDataGlobal.radius_deg || 0.26);
            const normOffset = Math.max(-1.7, Math.min(1.7, target.vertical_offset_deg / Math.max(0.01, bodyRad)));
            const chordY = 24 - (normOffset * 11);

            return `
                <div class="flex items-center gap-2.5 bg-slate-950/90 p-2 rounded-lg border border-slate-800/90 mt-1 shadow-inner font-mono">
                    <svg width="48" height="48" viewBox="0 0 48 48" class="shrink-0 select-none">
                        <circle cx="24" cy="24" r="22" fill="none" stroke="#1e293b" stroke-width="1"/>
                        <circle cx="24" cy="24" r="18" fill="${isSun ? '#451a03' : '#082f49'}" stroke="${bodyColor}" stroke-width="1.4"/>
                        <line x1="4" y1="24" x2="44" y2="24" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
                        <line x1="24" y1="4" x2="24" y2="44" stroke="#475569" stroke-width="0.75" stroke-dasharray="2,2"/>
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

        let fetchBusy = false;
        async function fetchData() {
            if (fetchBusy) return;
            fetchBusy = true;
            const feedBadge = document.getElementById('feed-badge');
            try {
                const totalAlt = Math.round(terrainElevationM + buildingOffsetM);
                const res = await fetch(`/api/data?lat=${observerLat}&lon=${observerLon}&alt=${totalAlt}&lead=${userLeadSec}`, { cache: 'no-store' });
                const data = await res.json();
                if (data.error) {
                    feedBadge.dataset.state = 'bad';
                    feedBadge.innerText = 'FEED ERROR';
                    return;
                }
                feedBadge.dataset.state = 'ok';
                feedBadge.innerText = (data.source_feed || 'ONLINE').toUpperCase();
                applyData(data);
            } catch (err) {
                feedBadge.dataset.state = 'bad';
                feedBadge.innerText = 'OFFLINE';
            } finally {
                fetchBusy = false;
            }
        }

        function applyData(data) {
            moonDataGlobal = data.moon;
            sunDataGlobal = data.sun;

            document.getElementById('moon-coords').innerText = moonDataGlobal.visible ? `Moon: Az ${moonDataGlobal.azimuth}° | Alt +${moonDataGlobal.elevation}°` : `Moon Hidden (${moonDataGlobal.elevation}°)`;
            document.getElementById('moon-badge').innerText = `${moonDataGlobal.event_type}: ${moonDataGlobal.next_event_str}`;
            document.getElementById('sun-coords').innerText = sunDataGlobal.visible ? `Sun: Az ${sunDataGlobal.azimuth}° | Alt +${sunDataGlobal.elevation}°` : `Sun Hidden (${sunDataGlobal.elevation}°)`;
            document.getElementById('sun-badge').innerText = `${sunDataGlobal.event_type}: ${sunDataGlobal.next_event_str}`;
            renderAstroVectors();

            activeAircraftData = data.aircraft || [];
            document.getElementById('plane-count').innerText = activeAircraftData.length;
            const tRef = data.server_time;
            const nowSec = syncedNow();
            const seenKeys = new Set();

            activeAircraftData.forEach(plane => {
                const k = plane._k = plane.callsign;
                seenKeys.add(k);
                const trackRad = (plane.track * Math.PI) / 180.0;
                const [pl, po] = projectLL(plane.lat, plane.lon, plane.speed_ms, trackRad, Math.max(-1, Math.min(8, nowSec - tRef)));
                let st = planesState[k];

                if (!st) {
                    const marker = L.marker([pl, po], { icon: L.divIcon({ className: '', html: planeIconHtml(plane), iconSize: [40, 40], iconAnchor: [20, 20] }) }).addTo(map);
                    marker.bindPopup(popupHtml(plane));
                    st = planesState[k] = { marker, curLat: pl, curLon: po };
                } else {
                    st.marker.setIcon(L.divIcon({ className: '', html: planeIconHtml(plane), iconSize: [40, 40], iconAnchor: [20, 20] }));
                    st.marker.setPopupContent(popupHtml(plane));
                }
                st.baseLat = plane.lat; st.baseLon = plane.lon; st.tRef = tRef;
                st.speedMs = plane.speed_ms; st.trackRad = trackRad;
                st.lastSeen = nowSec;
            });

            for (const k in planesState) {
                if (!seenKeys.has(k) && nowSec - planesState[k].lastSeen > 15.0) {
                    map.removeLayer(planesState[k].marker);
                    delete planesState[k];
                    delete tcaSmoothedEpochs[k];
                }
            }
            updateHUDCountdowns();
        }

        let lastAnim = performance.now();
        function animateFrame(nowMs) {
            lastAnim = nowMs;
            const now = syncedNow();
            for (const k in planesState) {
                const p = planesState[k];
                if (p.baseLat === undefined) continue;
                const [la, lo] = projectLL(p.baseLat, p.baseLon, p.speedMs, p.trackRad, Math.max(-1.0, Math.min(8.0, now - p.tRef)));
                p.curLat = la; p.curLon = lo;
                p.marker.setLatLng([la, lo]);
            }
            requestAnimationFrame(animateFrame);
        }

        function updateHUDCountdowns() {
            const container = document.getElementById('alerts-container');
            if (!activeAircraftData || activeAircraftData.length === 0) {
                container.innerHTML = `<div class="text-xs text-slate-500 text-center py-8 font-mono">Scanning airspace for transit intercept...</div>`;
                clearTcaTrajectory();
                return;
            }

            const now = syncedNow();
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

                const mins = Math.floor(remaining / 60);
                const secs = (remaining % 60).toFixed(1).padStart(4, '0');
                const timerStr = remaining > 0 ? `T-${mins.toString().padStart(2, '0')}:${secs}` : `TRANSITING!`;

                let cardBorder = 'border-slate-800/80 bg-slate-950/60';
                let tagHtml = `<span class="bg-slate-900 text-slate-400 border border-slate-800 px-1.5 py-0.5 rounded text-[8px] font-bold font-mono">EN ROUTE</span>`;

                if (isTransit) {
                    cardBorder = 'border-red-500/90 bg-red-950/60 glow-red';
                    tagHtml = `<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[8px] font-black font-mono animate-pulse">TRANSIT LOCK</span>`;
                    
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
                        if (wholeSec === 180) { spokenCountdowns.add(alertKey); playChime(); speak("Transit in 3 minutes"); }
                        else if (wholeSec === 60) { spokenCountdowns.add(alertKey); playChime(); speak("Warning, transit in 60 seconds"); }
                        else if (wholeSec === 10) { spokenCountdowns.add(alertKey); speak("10 seconds"); }
                        else if ([5, 4, 3, 2, 1].includes(wholeSec)) { spokenCountdowns.add(alertKey); playTone(1100 + (5 - wholeSec) * 120, 0.08); }
                    }

                    if (remaining <= 0.2 && lastBeepedFlight !== flightKey) {
                        playTransitChord();
                        lastBeepedFlight = flightKey;
                    }
                } else if (isClose) {
                    cardBorder = 'border-amber-500/80 bg-amber-950/50 glow-amber';
                    tagHtml = `<span class="bg-amber-600 text-white px-2 py-0.5 rounded text-[8px] font-bold font-mono">CLOSE PASS</span>`;
                    if (!priorityTransitFound && remaining > 0 && remaining < 180) {
                        priorityTransitFound = true;
                        renderTcaTrajectory(plane, target);
                    }
                }

                html += `
                    <div onclick="focusPlane('${plane._k}')" class="p-2.5 rounded-xl border ${cardBorder} text-xs flex flex-col gap-1.5 cursor-pointer hud-card">
                        <div class="flex justify-between items-center border-b border-slate-800 pb-1.5 font-mono">
                            <div class="flex items-center gap-1.5">
                                <span class="font-black text-sm ${isTransit ? 'text-red-400' : 'text-slate-100'}">${plane.callsign}</span>
                                <span class="px-1.5 py-0.2 bg-slate-900 text-cyan-300 border border-slate-700 rounded text-[9px] font-bold">${plane.model}</span>
                            </div>
                            ${tagHtml}
                        </div>
                        <div class="grid grid-cols-3 gap-1.5 items-center font-mono my-0.5">
                            <div class="col-span-1 px-2 py-1 rounded bg-slate-950 border ${isTransit ? 'border-red-500 text-red-400' : (isClose ? 'border-amber-500 text-amber-400' : 'border-slate-800 text-cyan-400')} flex flex-col items-center justify-center">
                                <span class="text-[8px] text-slate-500 font-bold uppercase">EST. TCA</span>
                                <span class="text-xs font-black tabular-nums">${timerStr}</span>
                            </div>
                            <div class="col-span-2 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px]">
                                <div><span class="text-slate-500 text-[8px] uppercase block">ALT</span><b class="tabular-nums">${plane.alt_ft.toLocaleString()} ft</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase block">V/S</span><b class="tabular-nums">${plane.vr_fpm > 0 ? '+' : ''}${plane.vr_fpm}</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase block">SPEED</span><b class="tabular-nums">${plane.speed_kt} kt</b></div>
                                <div><span class="text-slate-500 text-[8px] uppercase block">DIST</span><b class="tabular-nums">${plane.distance_km} km</b></div>
                            </div>
                        </div>
                        <div class="flex justify-between items-center text-[10px] font-mono text-slate-400 border-t border-slate-800/80 pt-1">
                            <span>HDG: <b class="text-slate-200 tabular-nums">${plane.track}°</b></span>
                            <span>SEP: <b class="text-indigo-300 tabular-nums">${target.min_sep}°</b></span>
                            <span>OFF: <b class="text-slate-200 tabular-nums">${target.position_descriptor}</b></span>
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

            if (trajectoryLine) trajectoryLine.setLatLngs([start, tcaPos]);
            else trajectoryLine = L.polyline([start, tcaPos], { color: '#ef4444', weight: 2.2, dashArray: '4, 6', opacity: 0.85 }).addTo(map);

            const tcaHtml = `<div class="tca-target"><div class="tca-ring"></div><div class="tca-crosshair-h"></div><div class="tca-crosshair-v"></div><div class="tca-core"></div></div>`;
            if (tcaMarker) tcaMarker.setLatLng(tcaPos);
            else tcaMarker = L.marker(tcaPos, { icon: L.divIcon({ html: tcaHtml, iconSize: [28, 28], iconAnchor: [14, 14] }) }).addTo(map);
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
        document.getElementById('header-moon-icon').innerHTML = getRealisticMoonSvgHtml(20);
        document.getElementById('header-sun-icon').innerHTML = getRealisticSunSvgHtml(20);

        updateAltitudeDisplay();
        drawRangeRings();
        setFilterMode('moon');
        syncClock().then(fetchData);
        fetchTerrainElevation(observerLat, observerLon);
        setInterval(fetchData, 1800);
        setInterval(syncClock, 30000);
        setInterval(updateHUDCountdowns, 120);
        requestAnimationFrame(animateFrame);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print(f" [OK] LUNAR TRANSIT RADAR PRO // BLINDADO CON FR24 ICONS")
    print(f" [OK] Buffer de Memoria (60s hold) & Fallback: ACTIVO")
    print(f" [OK] Servidor escuchando en puerto: {port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)

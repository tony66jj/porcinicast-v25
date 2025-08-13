# main.py — Trova Porcini API v2.5.0 - Render Free Optimized
# Mantiene funzionalità scientifiche avanzate con algoritmi Python puri
# Build veloce per rispettare i limiti di tempo di Render Free

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, time, sqlite3, logging, json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Trova Porcini API v2.5.0 - Scientific Advanced", version="2.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HEADERS = {"User-Agent": "TrovaPorcini/2.5.0", "Accept-Language": "it"}
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY")
CDS_API_KEY = os.environ.get("CDS_API_KEY")

# Database
DB_PATH = "porcini_validations.db"

def init_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL, lon REAL NOT NULL, date TEXT NOT NULL,
                species TEXT NOT NULL, quantity INTEGER DEFAULT 1,
                size_cm_avg REAL, confidence REAL DEFAULT 0.8,
                notes TEXT, habitat_observed TEXT,
                predicted_score INTEGER, model_version TEXT DEFAULT '2.5.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS no_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL, lon REAL NOT NULL, date TEXT NOT NULL,
                searched_hours REAL DEFAULT 2.0, habitat_searched TEXT,
                notes TEXT, predicted_score INTEGER,
                model_version TEXT DEFAULT '2.5.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL, lon REAL NOT NULL, date TEXT NOT NULL,
                predicted_score INTEGER NOT NULL, species TEXT NOT NULL,
                habitat TEXT, confidence_data TEXT,
                model_version TEXT DEFAULT '2.5.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database inizializzato")
    except Exception as e:
        logger.error(f"Errore database: {e}")

init_database()

# ===== ALGORITMI SCIENTIFICI AVANZATI (Python puro) =====

def clamp(v, a, b): 
    return max(a, min(b, v))

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std_dev(values):
    if len(values) < 2: return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5

def percentile(values, p):
    if not values: return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] * (1 - c) + sorted_vals[f + 1] * c
    return sorted_vals[f]

# ===== MODELLI AVANZATI =====

def api_index(precip_list, half_life=8.0):
    """Antecedent Precipitation Index - scientificamente accurato"""
    k = 1.0 - 0.5 ** (1.0 / max(1.0, half_life))
    api = 0.0
    for p in precip_list:
        api = (1 - k) * api + k * (p or 0.0)
    return api

def smi_advanced_pure_python(precip, et0):
    """Soil Moisture Index avanzato senza numpy"""
    alpha = 0.25
    s = 0.0
    values = []
    
    for i, (p, et) in enumerate(zip(precip, et0)):
        forcing = (p or 0.0) - (et or 0.0)
        
        # Correzione stagionale alpha
        month = ((i + datetime.now().month - len(precip) - 1) % 12) + 1
        alpha_adj = alpha * 1.2 if month in [6,7,8] else alpha
        
        s = (1 - alpha_adj) * s + alpha_adj * forcing
        values.append(s)
    
    if len(values) >= 10:
        p10 = percentile(values, 10)
        p90 = percentile(values, 90)
        if p90 - p10 < 1e-6: p10, p90 = p10-1, p90+1
        return [clamp((v - p10) / (p90 - p10), 0.0, 1.0) for v in values]
    
    return [0.5] * len(values)

def vpd_hpa(temp_c, rh_pct):
    """Deficit pressione vapore - formula Magnus"""
    sat_vp = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    return sat_vp * (1.0 - clamp(rh_pct, 0, 100) / 100.0)

def vpd_penalty_advanced(vpd_hpa_val, species_sens, elevation):
    """Penalità VPD con correzione altimetrica"""
    alt_factor = 1.0 - (elevation - 500.0) / 2000.0
    alt_factor = clamp(alt_factor, 0.7, 1.2)
    vpd_corrected = vpd_hpa_val * alt_factor
    
    if vpd_corrected <= 5.0: base = 1.0
    elif vpd_corrected >= 15.0: base = 0.3
    else: base = 1.0 - 0.7 * (vpd_corrected - 5.0) / 10.0
    
    penalty = 1.0 - (1.0 - base) * species_sens
    return clamp(penalty, 0.25, 1.0)

def thermal_shock_index(tmin_series, window_days=3):
    """Indice shock termico basato su Büntgen et al. 2012"""
    if len(tmin_series) < 2 * window_days: return 0.0
    
    recent = mean(tmin_series[-window_days:])
    previous = mean(tmin_series[-2*window_days:-window_days])
    drop = previous - recent
    
    if drop <= 0.5: return 0.0
    if drop >= 6.0: return 1.0
    
    # Funzione sigmoide
    return 1.0 / (1.0 + math.exp(-2.0 * (drop - 3.0)))

def microclimate_energy_advanced(aspect_oct, slope_deg, month, latitude, elevation):
    """Indice energetico microclimático"""
    if not aspect_oct or slope_deg < 0.5: return 0.5
    
    aspect_energy = {"N": 0.3, "NE": 0.4, "E": 0.6, "SE": 0.8, "S": 1.0, "SW": 0.9, "W": 0.7, "NW": 0.4}
    base_energy = aspect_energy.get(aspect_oct, 0.5)
    
    # Fattori stagionali
    if month in [6,7,8]: seasonal_factor = 1.0
    elif month in [9,10]: seasonal_factor = 0.8
    elif month in [4,5]: seasonal_factor = 0.7
    else: seasonal_factor = 0.5
    
    # Correzione latitudinale
    lat_factor = clamp(1.0 - (latitude - 42.0) / 50.0, 0.7, 1.2)
    
    # Correzione altimetrica
    if elevation > 1500: alt_factor = 0.85
    elif elevation > 1000: alt_factor = 0.95
    else: alt_factor = 1.0
    
    slope_factor = 1.0 + min(0.3, slope_deg / 60.0)
    
    final_energy = base_energy * seasonal_factor * lat_factor * alt_factor * slope_factor
    return clamp(final_energy, 0.2, 1.2)

# ===== SOGLIE DINAMICHE AVANZATE =====

def dynamic_rain_threshold_v25(smi, month, elevation, lat, temp_trend=0.0):
    """Soglie pioggia dinamiche super avanzate"""
    base_threshold = 7.5
    
    # SMI effect non-lineare
    if smi > 0.8: smi_factor = 0.6
    elif smi > 0.6: smi_factor = 0.8  
    elif smi < 0.3: smi_factor = 1.4
    else: smi_factor = 1.0
    
    # Evapotraspirazione stagionale
    seasonal_et = {1:0.5, 2:0.6, 3:0.8, 4:1.0, 5:1.3, 6:1.5, 7:1.6, 8:1.5, 9:1.2, 10:0.9, 11:0.6, 12:0.5}
    et_factor = seasonal_et.get(month, 1.0)
    
    # Correzioni geografiche
    if elevation > 1500: alt_factor = 0.75
    elif elevation > 1200: alt_factor = 0.85
    elif elevation > 800: alt_factor = 0.92
    else: alt_factor = 1.0
    
    if lat > 46.0: lat_factor = 0.9
    elif lat < 41.0: lat_factor = 1.1
    else: lat_factor = 1.0
    
    # Feedback trend termico
    if temp_trend > 1.0: temp_factor = 1.15
    elif temp_trend < -1.0: temp_factor = 0.9
    else: temp_factor = 1.0
    
    final_threshold = base_threshold * smi_factor * et_factor * alt_factor * lat_factor * temp_factor
    return clamp(final_threshold, 4.0, 18.0)

# ===== SMOOTHING AVANZATO (senza scipy) =====

def advanced_gaussian_smoothing(forecast):
    """Smoothing avanzato che preserva i picchi"""
    if len(forecast) < 3: return forecast[:]
    
    smoothed = []
    for i in range(len(forecast)):
        weights, values = [], []
        
        # Kernel gaussiano adattivo
        for j in range(max(0, i-2), min(len(forecast), i+3)):
            dist = abs(i - j)
            weight = math.exp(-dist**2 / 2.0)
            weights.append(weight)
            values.append(forecast[j])
        
        smoothed_val = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        
        # Preserva picchi importanti (>70)
        if forecast[i] > 70 and smoothed_val < forecast[i] * 0.85:
            smoothed_val = forecast[i] * 0.92
            
        smoothed.append(smoothed_val)
    
    return smoothed

# ===== CONFIDENCE 5D =====

def confidence_5d_advanced(weather_agree, habitat_conf, smi_reliable, vpd_valid, has_validations):
    """Sistema confidence 5-dimensionale"""
    met = clamp(weather_agree, 0.15, 0.98)
    eco = clamp(habitat_conf * (1.15 if has_validations else 1.0), 0.1, 0.95)
    hydro = clamp(smi_reliable * 0.8, 0.2, 0.9)  # Penalità per P-ET0 vs ERA5
    atmo = 0.85 if vpd_valid else 0.35
    emp = 0.75 if has_validations else 0.35
    
    # Media pesata con penalità per componenti basse
    weights = {"met": 0.28, "eco": 0.24, "hydro": 0.22, "atmo": 0.16, "emp": 0.10}
    components = [met, eco, hydro, atmo, emp]
    min_component = min(components)
    
    penalty = 1.0 if min_component > 0.4 else (0.8 + 0.2 * min_component / 0.4)
    
    overall = (weights["met"] * met + weights["eco"] * eco + weights["hydro"] * hydro + 
               weights["atmo"] * atmo + weights["emp"] * emp) * penalty
    
    return {
        "meteorological": round(met, 3), "ecological": round(eco, 3),
        "hydrological": round(hydro, 3), "atmospheric": round(atmo, 3),
        "empirical": round(emp, 3), "overall": round(clamp(overall, 0.15, 0.95), 3)
    }

# ===== PROFILI SPECIE AVANZATI =====

SPECIES_PROFILES_V25 = {
    "aereus": {
        "hosts": ["quercia", "castagno", "misto"], "season": [6,7,8,9,10], "peak": [7,8],
        "temp_opt": (18.0, 24.0), "lag_base": 9.2, "vpd_sens": 1.15, "elev_opt": (200, 1000)
    },
    "reticulatus": {
        "hosts": ["quercia", "castagno", "faggio", "misto"], "season": [5,6,7,8,9], "peak": [6,7],
        "temp_opt": (16.0, 22.0), "lag_base": 8.8, "vpd_sens": 1.0, "elev_opt": (100, 1200)
    },
    "edulis": {
        "hosts": ["faggio", "conifere", "misto"], "season": [8,9,10,11], "peak": [9,10],
        "temp_opt": (12.0, 18.0), "lag_base": 10.5, "vpd_sens": 1.2, "elev_opt": (600, 2000)
    },
    "pinophilus": {
        "hosts": ["conifere", "misto"], "season": [6,7,8,9,10], "peak": [8,9],
        "temp_opt": (14.0, 20.0), "lag_base": 9.8, "vpd_sens": 0.9, "elev_opt": (400, 1800)
    }
}

def infer_species_advanced(habitat, month, elevation, aspect, lat):
    """Inferenza specie multi-fattoriale"""
    h = habitat.lower() if habitat else "misto"
    candidates = []
    
    for species, profile in SPECIES_PROFILES_V25.items():
        if h not in profile["hosts"]: continue
        
        score = 1.0
        
        # Stagione
        if month in profile["peak"]: score *= 1.5
        elif month in profile["season"]: score *= 1.0
        else: score *= 0.3
        
        # Altitudine
        elev_min, elev_max = profile["elev_opt"]
        if elev_min <= elevation <= elev_max: score *= 1.2
        elif elevation < elev_min: score *= max(0.4, 1.0 - (elev_min - elevation) / 500.0)
        else: score *= max(0.4, 1.0 - (elevation - elev_max) / 800.0)
        
        # Geografia
        if species == "aereus" and lat < 42.0: score *= 1.2
        elif species == "edulis" and lat > 45.0: score *= 1.15
        
        candidates.append((species, score))
    
    if not candidates: return "reticulatus"
    return max(candidates, key=lambda x: x[1])[0]

# ===== LAG BIOLOGICO DINAMICO AVANZATO =====

def dynamic_lag_v25(smi, thermal_shock, tmean_7d, species, vpd_stress=0.0):
    """Lag biologico dinamico basato su Boddy et al. 2014"""
    profile = SPECIES_PROFILES_V25[species]
    base_lag = profile["lag_base"]
    
    # Effetti scientifici
    smi_effect = -4.5 * (smi ** 1.5)  # Non-lineare
    shock_effect = -2.0 * thermal_shock
    
    # Temperatura ottimale
    temp_min, temp_max = profile["temp_opt"]
    if temp_min <= tmean_7d <= temp_max: temp_effect = -1.5
    elif tmean_7d < temp_min: temp_effect = 2.0 * (temp_min - tmean_7d) / (temp_min - 5.0)
    else: temp_effect = 1.5 * (tmean_7d - temp_max) / (30.0 - temp_max)
    
    # VPD stress
    vpd_effect = 1.5 * vpd_stress * profile["vpd_sens"]
    
    final_lag = base_lag + smi_effect + shock_effect + temp_effect + vpd_effect
    return int(round(clamp(final_lag, 5, 15)))

def gaussian_kernel_advanced(x, mu, sigma, skewness=0.0):
    """Kernel gaussiano con asimmetria"""
    base_gauss = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    if skewness != 0.0:
        skew_factor = 1.0 + skewness * ((x - mu) / sigma)
        return base_gauss * max(0.1, skew_factor)
    
    return base_gauss

def event_strength_advanced(mm, duration_hours=24.0, antecedent_smi=0.5):
    """Forza evento con correzioni"""
    base_strength = 1.0 - math.exp(-mm / 15.0)
    duration_factor = min(1.2, 1.0 + (duration_hours - 12.0) / 48.0)
    smi_factor = 0.7 + 0.6 * antecedent_smi
    return clamp(base_strength * duration_factor * smi_factor, 0.0, 1.5)

# ===== EVENT DETECTION AVANZATA =====

def detect_rain_events_v25(rains, smi_series, month, elevation, lat):
    """Event detection con soglie dinamiche"""
    events = []
    i = 0
    
    while i < len(rains):
        smi_local = smi_series[i] if i < len(smi_series) else 0.5
        threshold_1d = dynamic_rain_threshold_v25(smi_local, month, elevation, lat)
        threshold_2d = threshold_1d * 1.4
        threshold_3d = threshold_1d * 1.8
        
        if rains[i] >= threshold_1d:
            strength = event_strength_advanced(rains[i], antecedent_smi=smi_local)
            events.append((i, rains[i], strength))
            i += 1
        elif i + 1 < len(rains) and (rains[i] + rains[i+1]) >= threshold_2d:
            avg_smi = (smi_local + (smi_series[i+1] if i+1 < len(smi_series) else 0.5)) / 2
            strength = event_strength_advanced(rains[i] + rains[i+1], duration_hours=36.0, antecedent_smi=avg_smi)
            events.append((i + 1, rains[i] + rains[i+1], strength))
            i += 2
        elif i + 2 < len(rains) and (rains[i] + rains[i+1] + rains[i+2]) >= threshold_3d:
            avg_smi = mean(smi_series[i:i+3]) if i+2 < len(smi_series) else 0.5
            strength = event_strength_advanced(rains[i] + rains[i+1] + rains[i+2], duration_hours=60.0, antecedent_smi=avg_smi)
            events.append((i + 2, rains[i] + rains[i+1] + rains[i+2], strength))
            i += 3
        else:
            i += 1
    
    return events

# ===== FETCHING METEO AVANZATO =====

async def fetch_open_meteo_advanced(lat, lon, past=15, future=10):
    """Open-Meteo con parametri avanzati"""
    url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "precipitation_sum", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
        "et0_fao_evapotranspiration", "relative_humidity_2m_mean", "shortwave_radiation_sum"
    ]
    
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join(daily_vars), "past_days": past, "forecast_days": future,
        "models": "best_match"
    }
    
    async with httpx.AsyncClient(timeout=35, headers=HEADERS) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

async def fetch_openweather_advanced(lat, lon):
    """OpenWeather per blend meteo"""
    if not OWM_KEY: return {}
    
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon, "exclude": "minutely,alerts",
        "units": "metric", "lang": "it", "appid": OWM_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=35, headers=HEADERS) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"OpenWeather failed: {e}")
        return {}

async def fetch_elevation_simple(lat, lon):
    """Elevazione con fallback robusto"""
    try:
        coords = [{"latitude": lat, "longitude": lon}]
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
            response = await client.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
            response.raise_for_status()
            data = response.json()
            return float(data["results"][0]["elevation"]), 8.0, 180.0, "S"  # elevation, slope, aspect, octant
    except Exception:
        return 800.0, 8.0, 180.0, "S"

async def infer_habitat_advanced(lat, lon, elevation):
    """Inferenza habitat euristica avanzata"""
    if elevation > 1200: return "faggio", 0.7
    elif elevation > 800: return "misto", 0.6  
    elif lat > 43.0: return "castagno", 0.6
    else: return "quercia", 0.6

def check_validations_advanced(lat, lon, days=30, radius_km=15.0):
    """Check validazioni con statistiche"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*), AVG(confidence) FROM sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta, cutoff))
        
        pos_result = cursor.fetchone()
        pos_count = pos_result[0] or 0
        
        cursor.execute('''
            SELECT COUNT(*) FROM no_sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta, cutoff))
        
        neg_count = cursor.fetchone()[0] or 0
        conn.close()
        
        return pos_count > 0, pos_count + neg_count, (pos_result[1] or 0.0)
        
    except Exception:
        return False, 0, 0.0

# ===== ANALISI TESTUALE AVANZATA =====

def build_analysis_v25_advanced(payload):
    """Genera analisi scientifica dettagliata"""
    idx = payload["index"]
    species = payload["species"]
    habitat = payload["habitat_used"]
    elevation = payload["elevation_m"]
    confidence = payload["confidence_detailed"]["overall"]
    
    lines = [
        f"<h4>🧬 Analisi Biologica Avanzata v2.5.0</h4>",
        f"<p><em>Modello fenologico basato su letteratura scientifica: Boddy et al. (2014), Büntgen et al. (2012)</em></p>",
        f"<p><strong>Specie predetta</strong>: <em>Boletus {species}</em> in habitat <strong>{habitat}</strong> a {elevation}m</p>",
        f"<p><strong>Indice corrente</strong>: <strong style='font-size:1.2em'>{idx}/100</strong> - "
    ]
    
    if idx >= 75: lines.append("<span style='color:#66e28a;font-weight:bold'>ECCELLENTE</span> - Condizioni ottimali")
    elif idx >= 60: lines.append("<span style='color:#8bb7ff;font-weight:bold'>MOLTO BUONE</span> - Fruttificazione abbondante")
    elif idx >= 45: lines.append("<span style='color:#ffc857;font-weight:bold'>BUONE</span> - Fruttificazione moderata")
    elif idx >= 30: lines.append("<span style='color:#ff9966;font-weight:bold'>MODERATE</span> - Fruttificazione limitata")
    else: lines.append("<span
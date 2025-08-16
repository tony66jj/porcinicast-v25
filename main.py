from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time, sqlite3, logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone, timedelta
import json

# Import avanzati con fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import geohash2
    GEOHASH_AVAILABLE = True
except ImportError:
    GEOHASH_AVAILABLE = False

try:
    import cdsapi
    from netCDF4 import Dataset, num2date
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False

# Setup logging professionale
log_dir = os.getenv('LOG_DIR', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'porcini.log'))
    ]
)
logger = logging.getLogger(__name__)

# ======= ESPOSIZIONE: HELPERS =======
ASPECT_LABELS = {"N":"N","NE":"NE","E":"E","SE":"SE","S":"S","SW":"SW","W":"W","NW":"NW",
                 "NORD":"N","NORD-EST":"NE","EST":"E","SUD-EST":"SE","SUD":"S","SUD-OVEST":"SW","OVEST":"W","NORD-OVEST":"NW"}

def normalize_octant(label: str) -> str:
    if not label: return ""
    return ASPECT_LABELS.get(label.strip().upper(), "")

def blend_to_neutral(value: float, neutral: float = 1.0, weight: float = 0.35) -> float:
    try:
        return neutral + (float(value) - neutral) * float(weight)
    except Exception:
        return value

app = FastAPI(
    title="BoletusLab® v3.0.0 - Sistema Multi-Specie con ERA5-Land",
    version="3.0.0",
    description="Sistema meteorologico ibrido con curve multiple sovrapposte per coesistenza specie"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

HEADERS = {"User-Agent":"BoletusLab/3.0.0 (+scientific)", "Accept-Language":"it"}
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

# CHIAVI API
VISUAL_CROSSING_KEY = os.environ.get("VISUAL_CROSSING_KEY", "")

# Database avanzato
data_dir = os.getenv('DATA_DIR', 'data')
os.makedirs(data_dir, exist_ok=True)
DB_PATH = os.path.join(data_dir, "porcini_validations.db")

def init_database():
    """Inizializza database SQLite per species tracking"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Aggiorna tabella segnalazioni per multi-specie
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                secondary_species TEXT,
                quantity INTEGER DEFAULT 1,
                size_cm_avg REAL,
                confidence REAL DEFAULT 0.8,
                notes TEXT,
                habitat_observed TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '3.0.0',
                coexistence_predicted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS no_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                searched_hours REAL DEFAULT 2.0,
                search_method TEXT DEFAULT 'visual',
                habitat_searched TEXT,
                notes TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '3.0.0',
                search_thoroughness INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                predicted_score INTEGER NOT NULL,
                species TEXT NOT NULL,
                secondary_species TEXT,
                coexistence_probability REAL,
                habitat TEXT,
                confidence_data TEXT,
                weather_data TEXT,
                model_version TEXT DEFAULT '3.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database multi-specie inizializzato")
    except Exception as e:
        logger.error(f"Errore database: {e}")

init_database()

# ===== UTILITIES MATEMATICHE =====
def clamp(v, a, b): 
    return a if v < a else b if v > b else v

def half_life_coeff(days: float) -> float:
    return 1.0 - 0.5**(1.0/max(1.0, days))

def api_index(precip: List[float], half_life: float = 8.0) -> float:
    k = half_life_coeff(half_life)
    api = 0.0
    for p in precip: 
        api = (1-k) * api + k * (p or 0.0)
    return api

def stddev_advanced(xs: List[float]) -> float:
    if not xs: return 0.0
    if NUMPY_AVAILABLE:
        return float(np.std(xs, ddof=1))
    else:
        m = sum(xs) / len(xs)
        return (sum((x-m)**2 for x in xs) / max(1, len(xs)-1))**0.5

def percentile_advanced(xs: List[float], p: float) -> float:
    if NUMPY_AVAILABLE:
        return float(np.percentile(xs, p))
    else:
        if not xs: return 0.0
        sorted_xs = sorted(xs)
        k = (len(sorted_xs) - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_xs):
            return sorted_xs[f] * (1 - c) + sorted_xs[f + 1] * c
        return sorted_xs[f]

def geohash_encode_advanced(lat: float, lon: float, precision: int = 8) -> str:
    if GEOHASH_AVAILABLE:
        return geohash2.encode(lat, lon, precision)
    else:
        return f"{lat:.4f},{lon:.4f}"

# ===== VPD AVANZATO =====
def saturation_vapor_pressure_hpa(Tc: float) -> float:
    return 6.112 * math.exp((17.67 * Tc) / (Tc + 243.5))

def vpd_hpa(Tc: float, RH: float) -> float:
    RHc = clamp(RH, 0.0, 100.0)
    return saturation_vapor_pressure_hpa(Tc) * (1.0 - RHc/100.0)

def vpd_penalty_advanced(vpd_max_hpa: float, species_vpd_sens: float = 1.0, 
                        elevation_m: float = 800.0) -> float:
    alt_factor = 1.0 - (elevation_m - 500.0) / 2000.0
    alt_factor = clamp(alt_factor, 0.7, 1.2)
    vpd_corrected = vpd_max_hpa * alt_factor
    
    if vpd_corrected <= 5.0: base = 1.0
    elif vpd_corrected >= 15.0: base = 0.3
    else: base = 1.0 - 0.7 * (vpd_corrected - 5.0) / 10.0
    
    penalty = 1.0 - (1.0-base) * species_vpd_sens
    return clamp(penalty, 0.25, 1.0)

def thermal_shock_index_advanced(tmin_series: List[float], window_days: int = 3) -> float:
    if len(tmin_series) < 2 * window_days: return 0.0
    
    recent = sum(tmin_series[-window_days:]) / window_days
    previous = sum(tmin_series[-2*window_days:-window_days]) / window_days
    drop = previous - recent
    
    if drop <= 0.5: return 0.0
    if drop >= 6.0: return 1.0
    
    return 1.0 / (1.0 + math.exp(-2.0 * (drop - 3.0)))

# ===== TOPOGRAFIA AVANZATA =====
def twi_advanced_proxy(slope_deg: float, concavity: float, 
                      drainage_area_proxy: float = 1.0) -> float:
    beta = max(0.1, math.radians(max(0.1, slope_deg)))
    tanb = max(0.05, math.tan(beta))
    area_proxy = max(0.1, 1.0 + 10.0 * max(0.0, concavity))
    twi = math.log(area_proxy) - math.log(tanb)
    return clamp((twi + 3.0) / 6.0, 0.0, 1.0)

def microclimate_energy_advanced(aspect_oct: Optional[str], slope_deg: float, 
                                month: int, latitude: float, elevation_m: float) -> float:
    if (not aspect_oct) or aspect_oct in {'FLAT','MULTI','MULTI_FLAT'}:
        return 0.5
    if slope_deg < 5.0:
        slope_damp = 0.2
    elif slope_deg < 10.0:
        slope_damp = 0.5
    else:
        slope_damp = 1.0
    
    aspect_energy = { 'N':0.6, 'NE':0.65, 'E':0.7, 'SE':0.75, 'S':0.7, 'SW':0.72, 'W':0.7, 'NW':0.62 }
    base_energy = aspect_energy.get(aspect_oct, 0.5)
    base_energy = 1.0 - (1.0 - base_energy) * slope_damp
    
    if month in [6,7,8]: seasonal_factor = 1.0
    elif month in [9,10]: seasonal_factor = 0.8
    elif month in [4,5]: seasonal_factor = 0.7
    else: seasonal_factor = 0.5
    
    lat_factor = 1.0 - (latitude - 42.0) / 50.0
    lat_factor = clamp(lat_factor, 0.7, 1.2)
    
    if elevation_m > 1500: alt_factor = 0.85
    elif elevation_m > 1000: alt_factor = 0.95
    else: alt_factor = 1.0
    
    slope_factor = 1.0 + min(0.3, slope_deg / 60.0)
    
    final_energy = base_energy * seasonal_factor * lat_factor * alt_factor * slope_factor
    return clamp(final_energy, 0.2, 1.2)

# ===== UMIDITÀ CUMULATIVA =====
def cumulative_moisture_index(P_series: List[float], days_window: int = 14) -> List[float]:
    cmi_values = []
    
    for i in range(len(P_series)):
        start_idx = max(0, i - days_window + 1)
        window_precip = P_series[start_idx:i+1]
        
        weights = []
        for j, p in enumerate(window_precip):
            days_ago = len(window_precip) - 1 - j
            weight = math.exp(-days_ago / 7.0)
            weights.append(weight)
        
        if weights:
            weighted_sum = sum(w * p for w, p in zip(weights, window_precip))
            weight_sum = sum(weights)
            cmi = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            cmi = 0.0
        
        cmi_values.append(cmi)
    
    return cmi_values

# ===== SOGLIE DINAMICHE SPECIE-SPECIFICHE =====
def dynamic_rain_threshold_v30(base_threshold_species: float, smi: float, month: int, elevation: float, 
                              lat: float, recent_temp_trend: float, 
                              cumulative_moisture: float) -> float:
    base_threshold = base_threshold_species # USA LA SOGLIA DELLA SPECIE
    
    if smi > 0.8: smi_factor = 0.6
    elif smi > 0.6: smi_factor = 0.8
    elif smi < 0.3: smi_factor = 1.25 # PENALITÀ PER SUOLO SECCO LEGGERMENTE RIDOTTA
    else: smi_factor = 1.0
    
    seasonal_et = {
        1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.3, 6: 1.5,
        7: 1.6, 8: 1.5, 9: 1.2, 10: 0.9, 11: 0.6, 12: 0.5
    }
    et_factor = seasonal_et.get(month, 1.0)
    
    if elevation > 1500: alt_factor = 0.75
    elif elevation > 1200: alt_factor = 0.85
    elif elevation > 800: alt_factor = 0.92
    else: alt_factor = 1.0
    
    if lat > 46.0: lat_factor = 0.9
    elif lat < 41.0: lat_factor = 1.1
    else: lat_factor = 1.0
    
    if recent_temp_trend > 1.0: temp_factor = 1.15
    elif recent_temp_trend < -1.0: temp_factor = 0.9
    else: temp_factor = 1.0
    
    if cumulative_moisture > 15.0: moisture_factor = 0.7
    elif cumulative_moisture > 10.0: moisture_factor = 0.85
    elif cumulative_moisture > 5.0: moisture_factor = 0.95
    else: moisture_factor = 1.1
    
    final_threshold = (base_threshold * smi_factor * et_factor * alt_factor * lat_factor * temp_factor * moisture_factor)
    return clamp(final_threshold, 3.0, 20.0)

# ===== SMOOTHING SAVITZKY-GOLAY =====
def savitzky_golay_advanced(forecast: List[float], window_length: int = 5, 
                           polyorder: int = 2) -> List[float]:
    if len(forecast) < 5:
        return simple_smoothing_fallback(forecast)
    
    if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
        try:
            arr = np.array(forecast, dtype=float)
            wl = min(window_length, len(arr))
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
            po = min(polyorder, wl - 1)
            
            smoothed = savgol_filter(arr, window_length=wl, polyorder=po, mode='nearest')
            
            for i, (orig, smooth) in enumerate(zip(forecast, smoothed)):
                if orig > 75 and smooth < orig * 0.8:
                    smoothed[i] = orig * 0.9
            
            return np.clip(smoothed, 0, 100).tolist()
            
        except Exception as e:
            logger.warning(f"Savgol failed: {e}")
    
    return advanced_custom_smoothing(forecast)

def advanced_custom_smoothing(forecast: List[float]) -> List[float]:
    if len(forecast) < 3:
        return forecast[:]
    
    smoothed = []
    for i in range(len(forecast)):
        weights = []
        values = []
        
        for j in range(max(0, i-2), min(len(forecast), i+3)):
            dist = abs(i - j)
            weight = math.exp(-dist**2 / 2.0)
            weights.append(weight)
            values.append(forecast[j])
        
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        weight_sum = sum(weights)
        smoothed_val = weighted_sum / weight_sum
        
        if forecast[i] > 70 and smoothed_val < forecast[i] * 0.85:
            smoothed_val = forecast[i] * 0.92
        
        smoothed.append(smoothed_val)
    
    return smoothed

def simple_smoothing_fallback(forecast: List[float]) -> List[float]:
    if len(forecast) <= 2:
        return forecast[:]
    
    smoothed = [forecast[0]]
    for i in range(1, len(forecast)-1):
        smoothed.append((forecast[i-1] + 2*forecast[i] + forecast[i+1]) / 4.0)
    smoothed.append(forecast[-1])
    return smoothed

# ===== SMI CON ERA5-LAND OPZIONALE =====
SM_CACHE: Dict[str, Dict[str, Any]] = {}

async def _prefetch_era5l_sm_advanced(lat: float, lon: float, days: int = 40, 
                                    enable_era5: bool = False) -> None:
    if not enable_era5 or not CDS_API_KEY or not CDS_AVAILABLE: 
        logger.info("ERA5-Land skipped (disabled or no API key)")
        return
    
    key = f"{round(lat,3)},{round(lon,3)}"
    if key in SM_CACHE and (time.time() - SM_CACHE[key].get("ts", 0)) < 12*3600:
        return
    
    def _blocking_download():
        try:
            logger.info("ERA5-Land: Downloading soil moisture data...")
            c = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY, quiet=True, verify=1)
            end = datetime.utcnow().date()
            start = end - timedelta(days=days-1)
            years = sorted({start.year, end.year})
            months = [f"{m:02d}" for m in range(1,13)] if len(years)>1 else [f"{m:02d}" for m in range(start.month, end.month+1)]
            days_list = [f"{d:02d}" for d in range(1,31)]
            bbox = [lat+0.05, lon-0.05, lat-0.05, lon+0.05]
            
            req = {
                "product_type": "reanalysis",
                "variable": ["volumetric_soil_water_layer_1"],
                "year": [str(y) for y in years],
                "month": months,
                "day": days_list,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,
                "format": "netcdf",
            }
            
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                target = tmp.name
            c.retrieve("reanalysis-era5-land", req, target)
            
            ds = Dataset(target)
            t = ds.variables["time"]
            times = num2date(t[:], t.units)
            var = ds.variables.get("swvl1") or ds.variables["volumetric_soil_water_layer_1"]
            data = var[:]
            
            daily: Dict[str, float] = {}
            if NUMPY_AVAILABLE:
                for i in range(data.shape[0]):
                    v = float(np.nanmean(np.array(data[i]).astype("float64")))
                    day = times[i].date().isoformat()
                    if day not in daily: daily[day] = v
                    else: daily[day] = (daily[day] + v) / 2.0
            
            ds.close()
            os.remove(target)
            logger.info(f"ERA5-Land: Successfully downloaded {len(daily)} days")
            return {"daily": daily, "ts": time.time()}
            
        except Exception as e:
            logger.warning(f"ERA5-Land download failed: {e}")
            return None
    
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _blocking_download)
        if data and "daily" in data:
            SM_CACHE[key] = data
    except Exception as e:
        logger.warning(f"ERA5-Land processing failed: {e}")

def smi_from_p_et0_advanced(P: List[float], ET0: List[float], 
                           era5_data: Optional[Dict] = None) -> List[float]:
    if era5_data and "daily" in era5_data:
        logger.info("Using ERA5-Land soil moisture data")
        # Usa dati ERA5-Land se disponibili
        era5_values = list(era5_data["daily"].values())
        if len(era5_values) >= len(P):
            return era5_values[-len(P):]  # Ultimi N giorni
    
    # Fallback a calcolo P-ET0
    if NUMPY_AVAILABLE:
        alpha = 0.25
        S = 0.0
        xs = []
        
        for i, (p, et) in enumerate(zip(P, ET0)):
            forcing = (p or 0.0) - (et or 0.0)
            month = ((i + datetime.now().month - len(P) - 1) % 12) + 1
            if month in [6,7,8]: alpha_adj = alpha * 1.2
            else: alpha_adj = alpha
            
            S = (1 - alpha_adj) * S + alpha_adj * forcing
            xs.append(S)
        
        arr = np.array(xs, dtype=float)
        valid = arr[np.isfinite(arr)]
        
        if valid.size >= 10:
            p10, p90 = np.percentile(valid, [10, 90])
        else:
            p10, p90 = (float(arr.min()), float(arr.max())) if valid.size > 0 else (-1, 1)
            if p90 - p10 < 1e-6: p10, p90 = p10-1, p90+1
        
        normalized = (arr - p10) / max(1e-6, (p90 - p10))
        return np.clip(normalized, 0.0, 1.0).tolist()
    else:
        return smi_fallback_pure_python(P, ET0)

def smi_fallback_pure_python(P: List[float], ET0: List[float]) -> List[float]:
    alpha = 0.25
    S = 0.0
    xs = []
    
    for p, et in zip(P, ET0):
        forcing = (p or 0.0) - (et or 0.0)
        S = (1 - alpha) * S + alpha * forcing
        xs.append(S)
    
    if len(xs) >= 5:
        sorted_xs = sorted(xs)
        p10_idx = max(0, int(0.1 * len(sorted_xs)))
        p90_idx = min(len(sorted_xs)-1, int(0.9 * len(sorted_xs)))
        p10, p90 = sorted_xs[p10_idx], sorted_xs[p90_idx]
    else:
        p10, p90 = (min(xs), max(xs)) if xs else (-1.0, 1.0)
        if p90-p10 < 1e-6: p10, p90 = p10-1, p90+1
    
    return [clamp((x-p10)/(p90-p10), 0.0, 1.0) for x in xs]

# ===== SISTEMA MULTI-SPECIE: PROFILI CON SOGLIE CALIBRATE =====
SPECIES_PROFILES_V30 = {
    "aereus": {
        "hosts": ["quercia", "castagno", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [7, 8]},
        "tm7_opt": (18.0, 24.0), "tm7_critical": (12.0, 28.0),
        "lag_base": 8.5, "lag_range": (6, 11),
        "vpd_sens": 1.15, "drought_tolerance": 0.8,
        "elevation_opt": (200, 1200),
        "min_precip_flush": 10.5, # Soglia alta, specie termofila
        "humidity_requirement": 85.0,
        "geographic_preference": "mediterraneo",
        "description": "Porcino Nero, termofilo, preferisce querce e zone calde e secche"
    },
    "reticulatus": {
        "hosts": ["quercia", "castagno", "faggio", "misto"],
        "season": {"start_m": 5, "end_m": 9, "peak_m": [6, 7]},
        "tm7_opt": (16.0, 22.0), "tm7_critical": (10.0, 26.0),
        "lag_base": 7.8, "lag_range": (5, 10),
        "vpd_sens": 1.0, "drought_tolerance": 0.9,
        "elevation_opt": (100, 1300),
        "min_precip_flush": 9.0, # Soglia intermedia
        "humidity_requirement": 80.0,
        "geographic_preference": "temperato_caldo",
        "description": "Porcino Estivo, comune in faggete estive e querceti"
    },
    "edulis": {
        "hosts": ["faggio", "conifere", "misto"],
        "season": {"start_m": 8, "end_m": 11, "peak_m": [9, 10]},
        "tm7_opt": (12.0, 18.0), "tm7_critical": (6.0, 22.0),
        "lag_base": 10.2, "lag_range": (8, 14),
        "vpd_sens": 1.2, "drought_tolerance": 0.6,
        "elevation_opt": (500, 2000),
        "min_precip_flush": 7.5, # Soglia bassa, ama l'umidità
        "humidity_requirement": 90.0,
        "geographic_preference": "temperato_fresco",
        "description": "Porcino Classico autunnale, ama il fresco e l'umido di faggi e abeti"
    },
    "pinophilus": {
        "hosts": ["conifere", "pino", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [8, 9]},
        "tm7_opt": (14.0, 20.0), "tm7_critical": (8.0, 24.0),
        "lag_base": 9.3, "lag_range": (7, 12),
        "vpd_sens": 0.9, "drought_tolerance": 1.1,
        "elevation_opt": (400, 1800),
        "min_precip_flush": 8.5, # Soglia medio-bassa
        "humidity_requirement": 85.0,
        "geographic_preference": "coniferato",
        "description": "Porcino dei pini, legato a terreni acidi e ben drenati"
    }
}

# ===== SISTEMA DI COESISTENZA CON REGOLE ECOLOGICHE RIGIDE =====
def calculate_species_probabilities(habitat_used: str, month: int, elev_m: float, 
                                   aspect_oct: Optional[str], lat: float) -> Dict[str, float]:
    """
    Calcola probabilità di coesistenza con regole di habitat più rigide come richiesto.
    - Faggio: solo edulis o reticulatus.
    - Quercia: solo aereus.
    - Conifere: solo pinophilus.
    - Castagno/Misto: logica più flessibile.
    """
    scores = { "aereus": 0.0, "reticulatus": 0.0, "edulis": 0.0, "pinophilus": 0.0 }
    h = (habitat_used or "misto").lower()

    if h == "faggio":
        # Nelle faggete, solo edulis o reticulatus.
        # Li distinguiamo in base a stagione e altitudine per una stima più accurata.
        score_edulis = 1.0
        score_reticulatus = 1.0
        
        # B. edulis è tipicamente più tardivo (autunnale) e di alta quota.
        if month >= 9: score_edulis *= 1.5
        if elev_m > 1100: score_edulis *= 1.5
        
        # B. reticulatus (aestivalis) è più precoce (estivo) e comune a quote inferiori nella faggeta.
        if month < 9: score_reticulatus *= 1.5
        if elev_m <= 1100: score_reticulatus *= 1.5
        
        scores["edulis"] = score_edulis
        scores["reticulatus"] = score_reticulatus

    elif h == "quercia":
        # Nelle querce, l'unica specie ecologicamente corretta è B. aereus.
        scores["aereus"] = 1.0
        
    elif h in ["conifere", "pino"]:
        # Nelle conifere, la specie simbionte è B. pinophilus.
        scores["pinophilus"] = 1.0
        
    elif h == "castagno":
        # Nei castagneti sono comuni specie termofile come reticulatus e aereus.
        scores["reticulatus"] = 1.0
        scores["aereus"] = 0.7  # Spesso presente ma con B. reticulatus come dominante.
        
    else:  # 'misto' o altri habitat non specificati
        # Per i boschi misti, usiamo una logica flessibile che considera più fattori.
        # B. aereus (termofilo)
        if "quercia" in h or "castagno" in h or h == "misto":
            scores["aereus"] = 1.0 if 6 <= month <= 9 and elev_m < 1000 else 0.1
        # B. reticulatus (estivo)
        if "faggio" in h or "quercia" in h or "castagno" in h or h == "misto":
            scores["reticulatus"] = 1.0 if 5 <= month <= 9 else 0.2
        # B. edulis (autunnale, montano)
        if "faggio" in h or "conifere" in h or h == "misto":
            scores["edulis"] = 1.0 if 8 <= month <= 11 and elev_m > 800 else 0.2
        # B. pinophilus (legato alle conifere)
        if "conifere" in h or "pino" in h or h == "misto":
            scores["pinophilus"] = 0.9 if elev_m > 700 else 0.1

    # Normalizza i punteggi per ottenere le probabilità
    total = sum(scores.values())
    if total > 0:
        probabilities = {species: score / total for species, score in scores.items()}
    else:
        # Fallback nel caso nessun punteggio sia > 0
        return {"reticulatus": 1.0}
    
    # Filtra le specie con probabilità significativa per evitare rumore nel grafico
    significant_species = {k: v for k, v in probabilities.items() if v > 0.05}
    
    # Se il filtro rimuove tutte le specie, ripristina la più probabile come fallback
    if not significant_species:
        if probabilities:
            most_probable = max(probabilities, key=probabilities.get)
            return {most_probable: 1.0}
        else:
            return {"reticulatus": 1.0} # Fallback definitivo
        
    return significant_species


def determine_coexistence_scenario(species_probabilities: Dict[str, float]) -> str:
    """
    Determina lo scenario di coesistenza basato su letteratura
    """
    sorted_species = sorted(species_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_species:
        return "dominanza_netta"
    if len(sorted_species) == 1:
        return "dominanza_netta"
    elif len(sorted_species) == 2 and sorted_species[1][1] > 0.25:
        return "codominanza"
    elif len(sorted_species) >= 3:
        return "comunita_mista"
    else:
        return "dominanza_moderata"

# ===== LAG BIOLOGICO MULTI-SPECIE =====
def calculate_weighted_lag(species_probabilities: Dict[str, float], 
                          smi: float, thermal_shock: float, tmean7: float, 
                          vpd_stress: float, cumulative_moisture: float) -> Dict[str, int]:
    """
    Calcola lag per ogni specie significativa
    """
    species_lags = {}
    
    for species, probability in species_probabilities.items():
        if probability > 0.05:  # Solo specie significative
            profile = SPECIES_PROFILES_V30[species]
            
            # Calcolo lag specie-specifico
            base_lag = profile["lag_base"]
            smi_effect = -3.8 * (smi ** 1.3)
            shock_effect = -1.8 * thermal_shock
            
            tm_opt_min, tm_opt_max = profile["tm7_opt"]
            if tm_opt_min <= tmean7 <= tm_opt_max:
                temp_effect = -1.2
            else:
                temp_effect = 1.8
            
            vpd_effect = 1.3 * vpd_stress * profile["vpd_sens"]
            moisture_effect = -0.8 * min(1.0, cumulative_moisture / 20.0)
            
            final_lag = (base_lag + smi_effect + shock_effect + temp_effect + 
                        vpd_effect + moisture_effect)
            
            lag_min, lag_max = profile["lag_range"]
            species_lags[species] = int(round(clamp(final_lag, lag_min, lag_max)))
    
    return species_lags

# ===== CONFIDENCE SYSTEM 5D AGGIORNATO =====
def confidence_5d_multi_species(
    weather_agreement: float,
    habitat_confidence: float,
    smi_reliability: float,
    vpd_validity: bool,
    has_recent_validation: bool,
    coexistence_stability: float = 0.8,
    era5_quality: float = 0.0
) -> Dict[str, float]:
    
    met_conf = clamp(weather_agreement, 0.15, 0.98)
    
    # Bonus ERA5-Land
    if era5_quality > 0:
        met_conf *= (1.0 + era5_quality * 0.1)
    
    eco_base = clamp(habitat_confidence, 0.1, 0.9)
    if has_recent_validation: eco_base *= 1.15
    eco_conf = clamp(eco_base, 0.1, 0.95)
    
    hydro_base = clamp(smi_reliability, 0.2, 0.9)
    hydro_conf = hydro_base * clamp(coexistence_stability, 0.5, 1.0)
    
    atmo_base = 0.85 if vpd_validity else 0.35
    atmo_conf = atmo_base
    
    emp_base = 0.75 if has_recent_validation else 0.35
    emp_conf = emp_base
    
    weights = {"met": 0.28, "eco": 0.24, "hydro": 0.22, "atmo": 0.16, "emp": 0.10}
    
    components = [met_conf, eco_conf, hydro_conf, atmo_conf, emp_conf]
    min_component = min(components)
    
    penalty = 1.0 if min_component > 0.4 else (0.8 + 0.2 * min_component / 0.4)
    
    overall = (weights["met"] * met_conf + 
               weights["eco"] * eco_conf + 
               weights["hydro"] * hydro_conf + 
               weights["atmo"] * atmo_conf + 
               weights["emp"] * emp_conf) * penalty
    
    return {
        "meteorological": round(met_conf, 3),
        "ecological": round(eco_conf, 3),
        "hydrological": round(hydro_conf, 3),
        "atmospheric": round(atmo_conf, 3),
        "empirical": round(emp_conf, 3),
        "overall": round(clamp(overall, 0.15, 0.95), 3)
    }

# ===== METEO IBRIDO AGGIORNATO =====
async def fetch_visual_crossing_historical(lat: float, lon: float, days_back: int = 8) -> List[Dict[str, Any]]:
    if not VISUAL_CROSSING_KEY:
        logger.warning("Visual Crossing API key not configured")
        return []
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back-1)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_str}/{end_str}"
        params = {
            "key": VISUAL_CROSSING_KEY,
            "include": "days",
            "elements": "datetime,temp,tempmin,tempmax,precip,humidity",
            "unitGroup": "metric"
        }
        
        async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        
        results = []
        for day in data.get("days", []):
            results.append({
                "date": day["datetime"],
                "precipitation_mm": float(day.get("precip", 0.0)),
                "temp_min": float(day.get("tempmin", 0.0)),
                "temp_max": float(day.get("tempmax", 0.0)),
                "temp_mean": float(day.get("temp", 0.0)),
                "humidity": float(day.get("humidity", 65.0)),
                "source": "visual_crossing"
            })
        
        logger.info(f"Visual Crossing fetched {len(results)} days")
        return results
        
    except Exception as e:
        logger.error(f"Visual Crossing API failed: {e}")
        return []

async def fetch_open_meteo_recent_and_forecast(lat: float, lon: float, past: int = 7, future: int = 10) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "precipitation_sum", "precipitation_hours",
        "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
        "et0_fao_evapotranspiration", "relative_humidity_2m_mean"
    ]
    
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join(daily_vars),
        "past_days": past, "forecast_days": future,
    }
    
    async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    
    daily = data["daily"]
    results = []
    for i, date_str in enumerate(daily["time"]):
        results.append({
            "date": date_str,
            "precipitation_mm": float(daily["precipitation_sum"][i] or 0.0),
            "temp_min": float(daily["temperature_2m_min"][i] or 0.0),
            "temp_max": float(daily["temperature_2m_max"][i] or 0.0), 
            "temp_mean": float(daily.get("temperature_2m_mean", [0.0] * len(daily["time"]))[i] or 
                             (daily["temperature_2m_min"][i] + daily["temperature_2m_max"][i]) / 2.0),
            "humidity": float(daily.get("relative_humidity_2m_mean", [65.0] * len(daily["time"]))[i] or 65.0),
            "et0": float(daily.get("et0_fao_evapotranspiration", [2.0] * len(daily["time"]))[i] or 2.0),
            "source": "open_meteo"
        })
    
    return {"data": results, "original": data}

async def fetch_hybrid_weather_data(lat: float, lon: float, total_past_days: int = 20, 
                                   future_days: int = 10, enable_era5: bool = False) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching weather: Open-Meteo + ERA5-Land({enable_era5})")

        # Prefetch ERA5-Land se abilitato
        era5_task = None
        if enable_era5:
            era5_task = asyncio.create_task(_prefetch_era5l_sm_advanced(lat, lon, total_past_days, enable_era5))

        om_result = await fetch_open_meteo_recent_and_forecast(lat, lon, past=total_past_days, future=future_days)
        om_data_list = om_result.get("data", [])
        
        if not om_data_list or len(om_data_list) < total_past_days:
            raise RuntimeError(f"Open-Meteo insufficient data: {len(om_data_list)} rows")

        # Attendi ERA5-Land se attivo
        era5_data = None
        era5_quality = 0.0
        if era5_task:
            try:
                await era5_task
                cache_key = f"{round(lat,3)},{round(lon,3)}"
                if cache_key in SM_CACHE:
                    era5_data = SM_CACHE[cache_key]
                    era5_quality = 0.9  # Alta qualità per ERA5-Land
                    logger.info("ERA5-Land data integrated successfully")
            except Exception as e:
                logger.warning(f"ERA5-Land integration failed: {e}")

        time_all = [d["date"] for d in om_data_list]
        P_all = [float(d.get("precipitation_mm", 0.0)) for d in om_data_list]
        Tmin_all = [float(d.get("temp_min", 0.0)) for d in om_data_list]
        Tmax_all = [float(d.get("temp_max", 0.0)) for d in om_data_list]
        Tmean_all = [float(d.get("temp_mean", (d.get("temp_min",0.0)+d.get("temp_max",0.0))/2.0)) for d in om_data_list]
        RH_all = [float(d.get("humidity", 65.0)) for d in om_data_list]
        ET0_all = [float(d.get("et0", 2.0)) for d in om_data_list]

        daily_payload = {
            "time": time_all,
            "precipitation_sum": P_all,
            "temperature_2m_min": Tmin_all,
            "temperature_2m_max": Tmax_all,
            "temperature_2m_mean": Tmean_all,
            "relative_humidity_2m_mean": RH_all,
            "et0_fao_evapotranspiration": ET0_all
        }

        total_past_received = min(total_past_days, len(P_all))
        completeness = min(1.0, float(len(P_all)) / float(total_past_days + future_days))
        
        base_quality = min(0.97, 0.9 + 0.07 * completeness)
        enhanced_quality = base_quality + (era5_quality * 0.1)  # Bonus ERA5-Land

        return {
            "daily": daily_payload,
            "era5_data": era5_data,
            "metadata": {
                "sources": {
                    "visual_crossing_days": 0,
                    "open_meteo_past_days": total_past_received,
                    "open_meteo_forecast_days": max(0, len(P_all) - total_past_days),
                    "era5_land_enabled": enable_era5,
                    "era5_quality": era5_quality,
                    "total_past_days": total_past_received,
                    "completeness": completeness,
                    "backup_used": False
                },
                "quality_score": min(0.99, enhanced_quality)
            }
        }

    except Exception as e:
        logger.error(f"Hybrid weather system failed: {e}")
        # Fallback a Visual Crossing
        vc_data = await fetch_visual_crossing_historical(lat, lon, days_back=total_past_days)
        if not vc_data:
            raise HTTPException(500, "Errore sistema meteorologico: nessun dato disponibile")

        vc_sorted = sorted(vc_data, key=lambda x: x.get("date"))
        time_series = [d.get("date") for d in vc_sorted]
        P_series = [float(d.get("precipitation_mm", 0.0)) for d in vc_sorted]
        Tmin_series = [float(d.get("temp_min", 0.0)) for d in vc_sorted]
        Tmax_series = [float(d.get("temp_max", 0.0)) for d in vc_sorted]
        Tmean_series = [float(d.get("temp_mean", (d.get("temp_min",0.0)+d.get("temp_max",0.0))/2.0)) for d in vc_sorted]
        RH_series = [float(d.get("humidity", 65.0)) for d in vc_sorted]
        ET0_series = [2.0] * len(P_series)  # Stima per ET0

        daily_payload = {
            "time": time_series,
            "precipitation_sum": P_series,
            "temperature_2m_min": Tmin_series,
            "temperature_2m_max": Tmax_series,
            "temperature_2m_mean": Tmean_series,
            "relative_humidity_2m_mean": RH_series,
            "et0_fao_evapotranspiration": ET0_series
        }

        completeness = min(1.0, float(len(P_series)) / float(total_past_days))
        return {
            "daily": daily_payload,
            "era5_data": None,
            "metadata": {
                "sources": {
                    "visual_crossing_days": len(P_series),
                    "open_meteo_past_days": 0,
                    "era5_land_enabled": False,
                    "era5_quality": 0.0,
                    "completeness": completeness,
                    "backup_used": True,
                    "backup_provider": "visual_crossing"
                },
                "quality_score": min(0.9, 0.7 + 0.2 * completeness)
            }
        }

# ===== ELEVAZIONE AVANZATA =====
_elev_cache: Dict[str, Any] = {}

async def fetch_elevation_grid_super_advanced(lat: float, lon: float) -> Tuple[float, float, float, Optional[str], float, float]:
    best_result = None
    
    for step_m in [30.0, 90.0, 180.0]:
        try:
            grid = await _fetch_elevation_block_super_advanced(lat, lon, step_m)
            if not grid: continue
            
            slope, aspect, octant = slope_aspect_from_grid_super_advanced(grid, step_m)
            concavity = concavity_from_grid_super_advanced(grid)
            drainage = drainage_proxy_from_grid_advanced(grid)
            elevation = grid[1][1]
            
            relief = max(max(row) for row in grid) - min(min(row) for row in grid)
            quality = min(1.0, relief / 50.0)
            
            result = {
                "elevation": elevation, "slope": slope, "aspect": aspect,
                "octant": octant, "concavity": concavity, "drainage": drainage,
                "quality": quality, "scale": step_m
            }
            
            if best_result is None or quality > best_result["quality"]:
                best_result = result
                
        except Exception as e:
            logger.warning(f"Elevation error at scale {step_m}m: {e}")
            continue
    
    if not best_result:
        return 800.0, 8.0, 180.0, "S", 0.0, 1.0
    
    r = best_result
    return (float(r["elevation"]), r["slope"], r["aspect"], 
            r["octant"], r["concavity"], r["drainage"])

async def _fetch_elevation_block_super_advanced(lat: float, lon: float, step_m: float) -> Optional[List[List[float]]]:
    cache_key = f"{round(lat,5)},{round(lon,5)}@{int(step_m)}"
    
    if cache_key in _elev_cache:
        cache_age = time.time() - _elev_cache[cache_key].get("timestamp", 0)
        if cache_age < 3600:
            return _elev_cache[cache_key]["grid"]
    
    try:
        deg_lat = step_m / 111320.0
        deg_lon = step_m / (111320.0 * max(0.2, math.cos(math.radians(lat))))
        
        coords = []
        for dy in [-deg_lat, 0, deg_lat]:
            for dx in [-deg_lon, 0, deg_lon]:
                coords.append({
                    "latitude": lat + dy,
                    "longitude": lon + dx
                })
        
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
            r = await c.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": coords}
            )
            r.raise_for_status()
            j = r.json()
        
        elevations = [p["elevation"] for p in j["results"]]
        grid = [elevations[0:3], elevations[3:6], elevations[6:9]]
        
        _elev_cache[cache_key] = {
            "grid": grid,
            "timestamp": time.time()
        }
        
        if len(_elev_cache) > 1000:
            oldest_keys = sorted(_elev_cache.keys(), 
                               key=lambda k: _elev_cache[k]["timestamp"])[:200]
            for k in oldest_keys:
                _elev_cache.pop(k, None)
        
        return grid
        
    except Exception as e:
        logger.warning(f"Elevation fetch error: {e}")
        return None

def slope_aspect_from_grid_super_advanced(grid: List[List[float]], cell_size_m: float = 30.0) -> Tuple[float, float, Optional[str]]:
    z = grid
    
    dzdx = ((z[0][2] + 2*z[1][2] + z[2][2]) - (z[0][0] + 2*z[1][0] + z[2][0])) / (8 * cell_size_m)
    dzdy = ((z[2][0] + 2*z[2][1] + z[2][2]) - (z[0][0] + 2*z[0][1] + z[0][2])) / (8 * cell_size_m)
    
    slope_rad = math.atan(math.hypot(dzdx, dzdy))
    slope_deg = math.degrees(slope_rad)
    
    if dzdx == 0 and dzdy == 0:
        aspect_deg = 0.0
        octant = None
    else:
        aspect_rad = math.atan2(-dzdx, dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
        
        octants = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        idx = int((aspect_deg + 22.5) // 45)
        octant = octants[idx] if slope_deg > 2.0 else None
    
    return round(slope_deg, 2), round(aspect_deg, 1), octant

def concavity_from_grid_super_advanced(grid: List[List[float]]) -> float:
    z = grid
    center = z[1][1]
    
    curvatures = []
    
    if len(z) >= 3:
        ns_curv = z[0][1] + z[2][1] - 2*center
        curvatures.append(ns_curv)
    
    if len(z[0]) >= 3:
        ew_curv = z[1][0] + z[1][2] - 2*center
        curvatures.append(ew_curv)
    
    nw_se_curv = z[0][0] + z[2][2] - 2*center
    ne_sw_curv = z[0][2] + z[2][0] - 2*center
    curvatures.extend([nw_se_curv, ne_sw_curv])
    
    mean_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
    return clamp(mean_curvature / 10.0, -0.5, 0.5)

def drainage_proxy_from_grid_advanced(grid: List[List[float]]) -> float:
    z = grid
    center = z[1][1]
    
    draining_cells = 0
    total_cells = 0
    
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1: continue
            if z[i][j] > center: draining_cells += 1
            total_cells += 1
    
    drainage_ratio = draining_cells / total_cells if total_cells > 0 else 0.0
    return clamp(drainage_ratio, 0.1, 1.0)

# ===== OSM HABITAT =====
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter", 
    "https://overpass.openstreetmap.ru/api/interpreter"
]

async def fetch_osm_habitat_super_advanced(lat: float, lon: float, radius_m: int = 500) -> Tuple[str, float, Dict[str, float]]:
    
    query = f"""
    [out:json][timeout:30];
    (
      way(around:{radius_m},{lat},{lon})["landuse"="forest"];
      way(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];
      relation(around:{radius_m},{lat},{lon})["landuse"="forest"];
      relation(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];
      node(around:{radius_m},{lat},{lon})["natural"="tree"];
      node(around:{radius_m},{lat},{lon})["tree"];
    );
    out tags qt;
    """
    
    for url_idx, url in enumerate(OVERPASS_URLS):
        try:
            async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
                r = await c.post(url, data={"data": query})
                r.raise_for_status()
                j = r.json()
            
            scores = score_osm_elements_super_advanced(j.get("elements", []))
            habitat, confidence = choose_dominant_habitat_advanced(scores)
            
            logger.info(f"OSM habitat via URL {url_idx}: {habitat} (conf: {confidence:.2f})")
            return habitat, confidence, scores
            
        except Exception as e:
            logger.warning(f"OSM URL {url_idx} failed: {e}")
            continue
    
    logger.info("OSM failed, using heuristic")
    return habitat_heuristic_super_advanced(lat, lon)

def score_osm_elements_super_advanced(elements: List[Dict]) -> Dict[str, float]:
    scores = {"castagno": 0.0, "faggio": 0.0, "quercia": 0.0, "conifere": 0.0, "misto": 0.0}
    
    for element in elements:
        tags = {k.lower(): str(v).lower() for k, v in (element.get("tags", {})).items()}
        
        genus = tags.get("genus", "")
        species = tags.get("species", "")
        
        if "castanea" in genus or "castagna" in species:
            scores["castagno"] += 4.0
        elif "quercus" in genus or "querce" in species:
            scores["quercia"] += 4.0
        elif "fagus" in genus or "faggio" in species:
            scores["faggio"] += 4.0
        elif any(g in genus for g in ["pinus", "picea", "abies", "larix"]):
            scores["conifere"] += 3.5
        
        leaf_type = tags.get("leaf_type", "")
        if "needleleaved" in leaf_type:
            scores["conifere"] += 2.0
        elif "broadleaved" in leaf_type:
            scores["misto"] += 1.0
        
        wood = tags.get("wood", "")
        wood_scores = {
            "conifer": ("conifere", 2.5), "pine": ("conifere", 2.0),
            "spruce": ("conifere", 2.0), "fir": ("conifere", 2.0),
            "beech": ("faggio", 3.0), "oak": ("quercia", 3.0),
            "chestnut": ("castagno", 3.0), "broadleaved": ("misto", 1.5),
            "deciduous": ("misto", 1.0), "mixed": ("misto", 2.0)
        }
        
        for keyword, (habitat, score) in wood_scores.items():
            if keyword in wood:
                scores[habitat] += score
        
        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")
        
        if landuse == "forest" or natural in ["wood", "forest"]:
            for habitat in scores:
                scores[habitat] += 0.2
    
    return scores

def choose_dominant_habitat_advanced(scores: Dict[str, float]) -> Tuple[str, float]:
    total_score = sum(scores.values())
    
    if total_score < 0.5:
        return "misto", 0.15
    
    dominant = max(scores.items(), key=lambda x: x[1])
    habitat, max_score = dominant
    
    dominance_ratio = max_score / total_score
    confidence = min(0.95, dominance_ratio ** 0.7 * 0.9)
    
    if habitat in ["faggio", "castagno"] and max_score > 3.0:
        confidence *= 1.1
    
    return habitat, clamp(confidence, 0.1, 0.95)

def habitat_heuristic_super_advanced(lat: float, lon: float) -> Tuple[str, float, Dict[str, float]]:
    elevation_estimate = 800.0
    
    if lat > 46.5:
        habitat, conf = "conifere", 0.65
    elif lat > 45.0:
        habitat, conf = ("faggio" if elevation_estimate > 1000 else "misto"), 0.6
    elif lat > 43.5:
        if lon < 11.0:
            habitat, conf = "castagno", 0.55
        else:
            habitat, conf = "misto", 0.5
    elif lat > 41.5:
        if elevation_estimate > 1200:
            habitat, conf = "faggio", 0.6
        else:
            habitat, conf = "quercia", 0.55
    else:
        habitat, conf = "quercia", 0.6
    
    scores = {h: (0.8 if h == habitat else 0.1) for h in ["castagno", "faggio", "quercia", "conifere", "misto"]}
    
    return habitat, conf, scores

# ===== LOGICA EVENTI PIOVOSI SPECIE-SPECIFICA (NUOVA) =====
def detect_rain_events_for_species(
    species_profile: dict,
    rains: List[float], 
    smi_series: List[float], 
    month: int, 
    elevation: float, 
    lat: float,
    cumulative_moisture_series: List[float]
) -> List[Tuple[int, float, float]]:
    events = []
    n = len(rains)
    i = 0
    
    base_threshold_species = species_profile.get("min_precip_flush", 9.0)

    while i < n:
        smi_local = smi_series[i] if i < len(smi_series) else 0.5
        cum_moisture = cumulative_moisture_series[i] if i < len(cumulative_moisture_series) else 0.0
        temp_trend = 0.0
        
        # Calcola la soglia dinamica per la specie corrente
        threshold_1d = dynamic_rain_threshold_v30(
            base_threshold_species, smi_local, month, elevation, lat, temp_trend, cum_moisture
        )
        threshold_2d = threshold_1d * 1.3  # Moltiplicatore ridotto per maggiore sensibilità
        threshold_3d = threshold_1d * 1.6  # Moltiplicatore ridotto

        # Controlla prima gli eventi multi-giorno per catturare piogge prolungate
        if i + 2 < n:
            rain_3d = rains[i] + rains[i + 1] + rains[i + 2]
            if rain_3d >= threshold_3d:
                avg_smi = sum(smi_series[i:i+3]) / 3 if i+2 < len(smi_series) else 0.5
                strength = event_strength_advanced(rain_3d, duration_hours=60.0, antecedent_smi=avg_smi)
                events.append((i + 2, rain_3d, strength))
                i += 3
                continue
        
        if i + 1 < n:
            rain_2d = rains[i] + rains[i + 1]
            if rain_2d >= threshold_2d:
                avg_smi = (smi_local + (smi_series[i+1] if i+1 < len(smi_series) else 0.5)) / 2
                strength = event_strength_advanced(rain_2d, duration_hours=36.0, antecedent_smi=avg_smi)
                events.append((i + 1, rain_2d, strength))
                i += 2
                continue
        
        if rains[i] >= threshold_1d:
            strength = event_strength_advanced(rains[i], antecedent_smi=smi_local)
            events.append((i, rains[i], strength))
            i += 1
            continue
        
        i += 1
    
    return events


def event_strength_advanced(mm: float, duration_hours: float = 24.0, 
                          antecedent_smi: float = 0.5) -> float:
    base_strength = 1.0 - math.exp(-mm / 15.0)
    duration_factor = min(1.2, 1.0 + (duration_hours - 12.0) / 48.0)
    smi_factor = 0.7 + 0.6 * antecedent_smi
    return clamp(base_strength * duration_factor * smi_factor, 0.0, 1.5)

def gaussian_kernel_advanced(x: float, mu: float, sigma: float, skewness: float = 0.0) -> float:
    base_gauss = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    if skewness != 0.0:
        skew_factor = 1.0 + skewness * ((x - mu) / sigma)
        return base_gauss * max(0.1, skew_factor)
    
    return base_gauss

# ===== DATABASE UTILS MULTI-SPECIE =====
def save_prediction_multi_species(lat: float, lon: float, date: str, 
                                 primary_species: str, secondary_species: str,
                                 coexistence_prob: float, primary_score: int,
                                 confidence_data: dict, weather_data: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        geohash = geohash_encode_advanced(lat, lon, precision=8)
        
        cursor.execute('''
            INSERT INTO predictions 
            (lat, lon, date, predicted_score, species, secondary_species, 
             coexistence_probability, confidence_data, weather_data, model_version, geohash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, primary_score, primary_species, secondary_species,
              coexistence_prob, json.dumps(confidence_data), json.dumps(weather_data),
              "3.0.0", geohash))
        
        conn.commit()
        conn.close()
        logger.info(f"Multi-species prediction saved: {primary_species} + {secondary_species}")
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")

def check_recent_validations_super_advanced(lat: float, lon: float, days: int = 30, 
                                           radius_km: float = 15.0) -> Tuple[bool, int, float]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*), AVG(confidence) FROM sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, 
              lon - lon_delta, lon + lon_delta, cutoff_date))
        
        pos_result = cursor.fetchone()
        pos_count = pos_result[0] or 0
        pos_conf = pos_result[1] or 0.0
        
        cursor.execute('''
            SELECT COUNT(*), AVG(search_thoroughness) FROM no_sightings 
            WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
            AND date >= ?
        ''', (lat - lat_delta, lat + lat_delta, 
              lon - lon_delta, lon + lon_delta, cutoff_date))
        
        neg_result = cursor.fetchone()
        neg_count = neg_result[0] or 0
        
        total_count = pos_count + neg_count
        avg_accuracy = (pos_conf + (neg_result[1] or 0.0)) / 2.0 if total_count > 0 else 0.0
        
        conn.close()
        
        has_validations = total_count >= 3
        return has_validations, total_count, avg_accuracy
        
    except Exception as e:
        logger.error(f"Error checking validations: {e}")
        return False, 0, 0.0

# ===== ANALISI TESTUALE MULTI-SPECIE =====
def build_analysis_multi_species_v30(payload: Dict[str, Any]) -> str:
    """Genera analisi scientifica dettagliata per sistema multi-specie"""
    
    species_data = payload.get("species_analysis", {})
    primary_species = species_data.get("primary_species", "reticulatus")
    secondary_species = species_data.get("secondary_species", "")
    coexistence_scenario = species_data.get("coexistence_scenario", "dominanza_netta")
    
    weather_sources = payload.get("weather_sources", {})
    weather_quality = payload.get("weather_quality_score", 0.5)
    era5_enabled = weather_sources.get("era5_land_enabled", False)
    era5_quality = weather_sources.get("era5_quality", 0.0)
    
    habitat_used = payload.get("habitat_used", "misto")
    elevation = payload.get("elevation_m", 800)
    aspect = payload.get("aspect_octant", "N/A")
    
    lines = []
    
    # Header
    lines.append("<h4>🧬 Analisi Biologica Multi-Specie v3.0.0</h4>")
    lines.append("<p><em>Sistema a curve multiple basato su letteratura scientifica di coesistenza (Borgotaro model, Van der Linde 2004, Leonardi et al. 2005)</em></p>")
    
    # Sistema meteorologico avanzato
    lines.append("<h4>🌦️ Sistema Meteorologico Ibrido Avanzato</h4>")
    om_days = weather_sources.get("open_meteo_past_days", 0)
    era5_text = f" + ERA5-Land ({era5_quality:.2f})" if era5_enabled and era5_quality > 0 else ""
    quality_color = "#66e28a" if weather_quality >= 0.8 else "#ffc857" if weather_quality >= 0.6 else "#ff6b6b"
    
    lines.append(f"<p><strong>Fonti integrate</strong>: Open-Meteo ({om_days} giorni){era5_text}</p>")
    lines.append(f"<p><strong>Qualità dati</strong>: <span style='color:{quality_color};font-weight:bold'>{weather_quality:.2f}</span>/1.00")
    if era5_enabled:
        lines.append(" (Enhanced con dati di umidità suolo ERA5-Land)")
    lines.append("</p>")
    
    # Coesistenza specie
    lines.append("<h4>🍄 Analisi Coesistenza Specie (Borgotaro Model)</h4>")
    
    if coexistence_scenario == "dominanza_netta":
        primary_profile = SPECIES_PROFILES_V30.get(primary_species, {})
        description = primary_profile.get("description", "Specie non identificata")
        lines.append(f"<p><strong>Specie dominante</strong>: <em>Boletus {primary_species}</em> ({description})</p>")
        lines.append(f"<p><strong>Scenario</strong>: Dominanza netta - Condizioni ottimali per una singola specie in questo habitat ({habitat_used}) e elevazione ({elevation}m)</p>")
        
    elif coexistence_scenario == "codominanza":
        primary_profile = SPECIES_PROFILES_V30.get(primary_species, {})
        secondary_profile = SPECIES_PROFILES_V30.get(secondary_species, {})
        lines.append(f"<p><strong>Codominanza rilevata</strong>: <em>B. {primary_species}</em> + <em>B. {secondary_species}</em></p>")
        lines.append(f"<p><strong>Specie primaria</strong>: {primary_profile.get('description', 'N/A')}</p>")
        lines.append(f"<p><strong>Specie secondaria</strong>: {secondary_profile.get('description', 'N/A')}</p>")
        lines.append(f"<p><strong>Scenario</strong>: Zona di transizione ecologica - Due specie coesistono con probabilità significative. Questo riflette la realtà documentata in aree come Borgotaro (Parma) dove multiple specie di porcini crescono negli stessi boschi.</p>")
        
    elif coexistence_scenario == "comunita_mista":
        lines.append(f"<p><strong>Comunità mista complessa</strong>: Ambiente che supporta multiple specie di porcini</p>")
        lines.append(f"<p><strong>Specie principale</strong>: <em>B. {primary_species}</em></p>")
        lines.append(f"<p><strong>Scenario</strong>: Ecosistema ricco e diversificato - Condizioni favorevoli per una comunità fungina complessa. Tipico di boschi maturi con habitat diversificati.</p>")
    
    # Strategia di ricerca dettagliata
    lines.append("<h4>🎯 Strategia di Ricerca Scientificamente Informata</h4>")
    lines.append("<div class='return-advice'>")
    
    if coexistence_scenario in ["codominanza", "comunita_mista"]:
        lines.append("<p><strong>Approccio multi-target</strong>: Cerca in microhabitat differenti per massimizzare le probabilità di successo:</p>")
        lines.append("<ul style='margin:8px 0 0 20px'>")
        
        primary_profile = SPECIES_PROFILES_V30.get(primary_species, {})
        if primary_species == "edulis":
            lines.append("<li><strong>Per B. edulis</strong>: Zone più fresche e umide, versanti nord, sotto faggi maturi, terreno ricco di humus</li>")
        elif primary_species == "reticulatus":
            lines.append("<li><strong>Per B. reticulatus</strong>: Zone più soleggiate, margini del bosco, sotto querce e castagni, terreno ben drenato</li>")
        elif primary_species == "aereus":
            lines.append("<li><strong>Per B. aereus</strong>: Zone calde e asciutte, querceti esposti a sud, terreni calcarei</li>")
        elif primary_species == "pinophilus":
            lines.append("<li><strong>Per B. pinophilus</strong>: Pinete pure, terreni sabbiosi acidi, zone montane</li>")
            
        if secondary_species:
            secondary_profile = SPECIES_PROFILES_V30.get(secondary_species, {})
            if secondary_species == "edulis":
                lines.append("<li><strong>Per B. edulis (secondario)</strong>: Versanti settentrionali più freschi, faggete dense</li>")
            elif secondary_species == "reticulatus":
                lines.append("<li><strong>Per B. reticulatus (secondario)</strong>: Radure assolate, bordi del sentiero</li>")
            elif secondary_species == "aereus":
                lines.append("<li><strong>Per B. aereus (secondario)</strong>: Querceti termofili, pendii esposti</li>")
        
        lines.append("</ul>")
        lines.append("<p><strong>Timing ottimale</strong>: Pianifica uscite multiple per intercettare i diversi lag biologici delle specie (differenze 2-4 giorni)</p>")
        
    else:
        # Dominanza netta
        primary_profile = SPECIES_PROFILES_V30.get(primary_species, {})
        lines.append(f"<p><strong>Focus mirato su B. {primary_species}</strong>:</p>")
        lines.append("<ul style='margin:8px 0 0 20px'>")
        
        if primary_species == "edulis":
            lines.append("<li><strong>Habitat preferito</strong>: Faggete mature (>50 anni), versanti nord-nordest, terreno umido ricco di humus</li>")
            lines.append("<li><strong>Microhabitat</strong>: Sotto faggi isolati, radure piccole protette, zone con tappeto di foglie spesso</li>")
            lines.append("<li><strong>Altitudine ottimale</strong>: 800-1500m, evita quote troppo basse in estate</li>")
        elif primary_species == "reticulatus":
            lines.append("<li><strong>Habitat preferito</strong>: Querceti e castagneti, zone più soleggiate, margini dei boschi</li>")
            lines.append("<li><strong>Microhabitat</strong>: Ai piedi di querce mature, terreno asciutto e ben drenato, zone con luce filtrata</li>")
            lines.append("<li><strong>Stagionalità</strong>: Fruttificazione estiva (maggio-agosto), cerca dopo temporali caldi</li>")
        elif primary_species == "aereus":
            lines.append("<li><strong>Habitat preferito</strong>: Querceti mediterranei, terreni calcarei, esposizioni sud</li>")
            lines.append("<li><strong>Microhabitat</strong>: Sotto querce da sughero, lecci, terreno compatto e asciutto</li>")
            lines.append("<li><strong>Condizioni ideali</strong>: Dopo piogge estive intense, temperature 20-28°C</li>")
        elif primary_species == "pinophilus":
            lines.append("<li><strong>Habitat preferito</strong>: Pinete pure, terreni sabbiosi acidi, altitudini medie</li>")
            lines.append("<li><strong>Microhabitat</strong>: Sotto pini silvestri maturi, terreno coperto di aghi, zone aperte</li>")
            lines.append("<li><strong>Rarità</strong>: Specie meno comune, richiede ricerca sistematica in habitat specifici</li>")
        
        lines.append("</ul>")
    
    lines.append("</div>")
    
    # Condizioni ambientali specifiche
    lines.append("<h4>🌡️ Condizioni Ambientali e Timing</h4>")
    best_window = payload.get("best_window", {})
    if best_window and best_window.get("mean", 0) > 30:
        start_day = best_window.get("start", 0) + 1
        end_day = best_window.get("end", 0) + 1
        lines.append(f"<p><strong>Finestra ottimale</strong>: Giorni {start_day}-{end_day} (indice medio {best_window.get('mean', 0)})</p>")
        
        lines.append("<ul style='margin:8px 0 0 20px'>")
        lines.append(f"<li><strong>Condizioni attuali</strong>: Elevazione {elevation}m, esposizione {aspect}, habitat {habitat_used}</li>")
        lines.append("<li><strong>Temperatura ideale</strong>: 15-22°C per la fruttificazione ottimale</li>")
        lines.append("<li><strong>Umidità</strong>: Terreno umido ma non saturo, 24-48h dopo precipitazioni</li>")
        lines.append("<li><strong>Pressure atmosferica</strong>: Preferibilmente in aumento dopo sistemi perturbati</li>")
        lines.append("</ul>")
    else:
        lines.append("<p><strong>Condizioni attuali non ottimali</strong>: Attendi precipitazioni significative (>8mm) seguite da tempo stabile</p>")
    
    # Lag biologico e tempistiche
    species_analysis = payload.get("species_analysis", {})
    if "species_lags" in species_analysis:
        lines.append("<h4>⏱️ Lag Biologico Specie-Specifico</h4>")
        species_lags = species_analysis["species_lags"]
        lines.append("<p><strong>Tempistiche di fruttificazione previste</strong>:</p>")
        lines.append("<ul style='margin:8px 0 0 20px'>")
        for species, lag_days in species_lags.items():
            profile = SPECIES_PROFILES_V30.get(species, {})
            lag_range = profile.get("lag_range", (5, 14))
            lines.append(f"<li><em>B. {species}</em>: {lag_days} giorni (range normale: {lag_range[0]}-{lag_range[1]} giorni)</li>")
        lines.append("</ul>")
        lines.append("<p><em>I lag sono calcolati considerando umidità suolo, shock termico, VPD stress e umidità cumulativa secondo letteratura scientifica</em></p>")
    
    # Note finali scientifiche
    lines.append("<h4>📚 Base Scientifica</h4>")
    lines.append("<p><em>Questo modello integra evidenze da:</em></p>")
    lines.append("<ul style='margin:8px 0 0 20px'>")
    lines.append("<li>Borgotaro (Parma): Documentata coesistenza di 4 specie di porcini nella stessa area</li>")
    lines.append("<li>Van der Linde (2004): Studio morfologico e filogenetico europeo</li>")
    lines.append("<li>Leonardi et al. (2005): Analisi molecolare ITS del complesso B. edulis</li>")
    lines.append("<li>Studi di sovrapposizione ecologica in ecosistemi mediterranei e temperati</li>")
    lines.append("</ul>")
    
    return "\n".join(lines)

# ===== ENDPOINT PRINCIPALE MULTI-SPECIE (RIFATTO) =====
@app.get("/api/score")
async def api_score_multi_species(
    lat: float = Query(..., description="Latitudine"),
    lon: float = Query(..., description="Longitudine"),
    half: float = Query(8.5, gt=3.0, lt=20.0, description="Half-life API"),
    habitat: str = Query("", description="Habitat forzato"),
    autohabitat: int = Query(1, description="1=auto OSM"),
    hours: int = Query(4, ge=2, le=8, description="Ore sul campo"),
    aspect: str = Query("", description="Esposizione manuale"),
    autoaspect: int = Query(1, description="1=automatico DEM"),
    advanced_lag: int = Query(0, description="1=lag biologico avanzato"),
    use_era5: int = Query(0, description="1=abilita ERA5-Land"), 
    background_tasks: BackgroundTasks = None
):
    """
    🚀 ENDPOINT MULTI-SPECIE v3.0.0
    Sistema a curve multiple con coesistenza e soglie di pioggia specie-specifiche
    """
    start_time = time.time()
    
    try:
        logger.info(f"Multi-species analysis for ({lat:.4f}, {lon:.4f}) - ERA5: {bool(use_era5)}")
        
        # Fetch dati in parallelo
        tasks = [
            fetch_hybrid_weather_data(lat, lon, total_past_days=20, future_days=10, enable_era5=bool(use_era5)),
            fetch_elevation_grid_super_advanced(lat, lon),
        ]
        
        if autohabitat == 1:
            tasks.append(fetch_osm_habitat_super_advanced(lat, lon))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        weather_data = results[0] if not isinstance(results[0], Exception) else {}
        elev_data = results[1] if not isinstance(results[1], Exception) else (800.0, 8.0, 180.0, "S", 0.0, 1.0)
        
        if autohabitat == 1 and len(results) > 2:
            osm_habitat_data = results[2] if not isinstance(results[2], Exception) else ("misto", 0.15, {})
        else:
            osm_habitat_data = ("misto", 0.15, {})
        
        elev_m, slope_deg, aspect_deg, aspect_oct, concavity, drainage_proxy = elev_data
        
        # Esposizione
        aspect_source = "automatico DEM"
        aspect_used = aspect_oct or ""
        if autoaspect == 0:
            manual_oct = normalize_octant(aspect)
            if manual_oct:
                aspect_used = manual_oct
                aspect_source = "manuale"
        
        # Habitat
        habitat_used = (habitat or "").strip().lower()
        habitat_source = "manuale"
        habitat_confidence = 0.6
        auto_scores = {}
        
        if autohabitat == 1:
            auto_habitat, auto_conf, auto_scores = osm_habitat_data
            if not habitat_used and auto_habitat:
                habitat_used = auto_habitat
                habitat_confidence = auto_conf
                habitat_source = f"automatico OSM (conf {auto_conf:.2f})"
            elif habitat_used:
                habitat_source = "manuale (override)"
        
        if not habitat_used:
            habitat_used = "misto"
        
        # Sistema meteorologico
        if not weather_data or "daily" not in weather_data:
            raise HTTPException(500, "Errore sistema meteorologico ibrido")
        
        weather_quality = weather_data.get("metadata", {}).get("quality_score", 0.5)
        weather_sources = weather_data.get("metadata", {}).get("sources", {})
        era5_data = weather_data.get("era5_data")
        
        daily = weather_data["daily"]
        time_series = daily["time"]
        P_series = [float(x or 0.0) for x in daily["precipitation_sum"]]
        Tmin_series = [float(x or 0.0) for x in daily["temperature_2m_min"]]
        Tmax_series = [float(x or 0.0) for x in daily["temperature_2m_max"]]
        Tmean_series = [float(x or 0.0) for x in daily.get("temperature_2m_mean", [])]
        if not Tmean_series or not any(Tmean_series):
            Tmean_series = [(mn + mx) / 2.0 for mn, mx in zip(Tmin_series, Tmax_series)]
        ET0_series = daily.get("et0_fao_evapotranspiration", [2.0] * len(P_series))
        RH_series = daily.get("relative_humidity_2m_mean", [65.0] * len(P_series))
        
        past_days = 20
        future_days = 10
        
        P_past = P_series[:past_days]
        P_future = P_series[past_days:past_days + future_days]
        Tmean_past = Tmean_series[:past_days]
        Tmean_future = Tmean_series[past_days:past_days + future_days]
        Tmin_past = Tmin_series[:past_days]
        Tmax_past = Tmax_series[:past_days]
        RH_past = RH_series[:past_days]
        RH_future = RH_series[past_days:past_days + future_days]
        
        # Calcoli avanzati
        api_value = api_index(P_past, half_life=half)
        smi_series = smi_from_p_et0_advanced(P_series, ET0_series, era5_data)
        smi_current = smi_series[past_days - 1] if past_days - 1 < len(smi_series) else 0.5
        
        cumulative_moisture_series = cumulative_moisture_index(P_series, days_window=14)
        cumulative_moisture_current = cumulative_moisture_series[past_days - 1] if past_days - 1 < len(cumulative_moisture_series) else 0.0
        
        tmean_7d = sum(Tmean_past[-7:]) / max(1, len(Tmean_past[-7:]))
        thermal_shock = thermal_shock_index_advanced(Tmin_past, window_days=3)
        rh_7d = sum(RH_past[-7:]) / max(1, len(RH_past[-7:]))
        vpd_series_future = [vpd_hpa(Tmean_future[i], RH_future[i]) for i in range(min(len(Tmean_future), len(RH_future)))]
        vpd_current = vpd_series_future[0] if vpd_series_future else 5.0
        
        month_current = datetime.now(timezone.utc).month
        microclimate_energy = microclimate_energy_advanced(aspect_used or aspect_oct, slope_deg, month_current, lat, elev_m)
        k_aspect = 0.35 if aspect_source.startswith("automatico") else 1.0
        microclimate_energy = blend_to_neutral(microclimate_energy, 1.0, k_aspect)
        
        # SISTEMA MULTI-SPECIE
        species_probabilities = calculate_species_probabilities(habitat_used, month_current, elev_m, aspect_oct, lat)
        coexistence_scenario = determine_coexistence_scenario(species_probabilities)
        
        sorted_species = sorted(species_probabilities.items(), key=lambda x: x[1], reverse=True)
        primary_species = sorted_species[0][0] if sorted_species else "reticulatus"
        secondary_species = sorted_species[1][0] if len(sorted_species) > 1 and sorted_species[1][1] > 0.25 else ""
        primary_probability = sorted_species[0][1] if sorted_species else 1.0
        
        logger.info(f"Species analysis: {primary_species} ({primary_probability:.2f}) + {secondary_species} - {coexistence_scenario}")
        
        species_lags = calculate_weighted_lag(species_probabilities, smi_current, thermal_shock, tmean_7d, 
                                            vpd_current/10.0, cumulative_moisture_current)
        
        # Genera forecast per specie multiple con soglie diverse
        forecast_combined = [0.0] * future_days
        species_forecasts = {}
        all_flush_events = {}

        for species, probability in species_probabilities.items():
            if probability < 0.05:
                continue
                
            species_profile = SPECIES_PROFILES_V30[species]
            
            # 1. Rileva eventi piovosi CON SOGLIA SPECIFICA per questa specie
            rain_events_species = detect_rain_events_for_species(
                species_profile=species_profile,
                rains=P_past + P_future, 
                smi_series=smi_series,
                month=month_current, 
                elevation=elev_m, 
                lat=lat,
                cumulative_moisture_series=cumulative_moisture_series
            )
            all_flush_events[species] = rain_events_species

            # 2. Calcola la curva di previsione per questa specie basata SUI SUOI eventi
            species_forecast = [0.0] * future_days
            lag_days = species_lags.get(species, int(species_profile["lag_base"]))
            
            for event_idx, event_mm, event_strength in rain_events_species:
                peak_idx = event_idx + lag_days
                base_amplitude = event_strength * microclimate_energy
                
                if event_idx >= past_days:
                    future_idx = event_idx - past_days
                    vpd_penalty = vpd_penalty_advanced(vpd_series_future[future_idx], species_profile["vpd_sens"], elev_m) if future_idx < len(vpd_series_future) else 1.0
                else:
                    vpd_penalty = 1.0
                
                final_amplitude = base_amplitude * vpd_penalty
                sigma = 2.2 if event_strength > 0.8 else 1.8
                skew = 0.3 if species in ["aereus", "reticulatus"] else 0.1
                
                for day_idx in range(future_days):
                    abs_day_idx = past_days + day_idx
                    kernel_value = gaussian_kernel_advanced(abs_day_idx, peak_idx, sigma, skewness=skew)
                    species_forecast[day_idx] += 100.0 * final_amplitude * kernel_value
            
            species_forecast_clamped = [clamp(v, 0.0, 100.0) for v in species_forecast]
            species_forecast_smoothed = savitzky_golay_advanced(species_forecast_clamped)
            species_forecasts[species] = [int(round(x)) for x in species_forecast_smoothed]
            
            # 3. Pesa la curva per la sua probabilità e aggiungila al totale
            for i in range(future_days):
                forecast_combined[i] += species_forecast_smoothed[i] * probability
        
        # Forecast finale combinato
        forecast_final = [int(round(clamp(x, 0, 100))) for x in forecast_combined]
        
        # Dettagli eventi per la specie primaria
        primary_species_events = all_flush_events.get(primary_species, [])
        flush_events_details = []
        for event_idx, event_mm, event_strength in primary_species_events:
             when_str = time_series[event_idx] if event_idx < len(time_series) else f"+{event_idx - past_days + 1}d"
             flush_events_details.append({
                "event_day_index": event_idx, "event_when": when_str, "event_mm": round(event_mm, 1),
                "event_strength": round(event_strength, 2), "lag_days": species_lags.get(primary_species, 8),
                "species": primary_species, "observed": event_idx < past_days,
                "cumulative_moisture": round(cumulative_moisture_series[event_idx] if event_idx < len(cumulative_moisture_series) else 0.0, 1)
            })

        # Analisi Best window
        best_window = {"start": 0, "end": 2, "mean": 0}
        if len(forecast_final) >= 3:
            best_mean = 0
            for i in range(len(forecast_final) - 2):
                window_mean = sum(forecast_final[i:i+3]) / 3.0
                if window_mean > best_mean:
                    best_mean = window_mean
                    best_window = {"start": i, "end": i+2, "mean": int(round(window_mean))}
        
        current_index = forecast_final[0] if forecast_final else 0
        
        # Validazioni e confidence
        has_validations, validation_count, validation_accuracy = check_recent_validations_super_advanced(lat, lon)
        coexistence_stability = 1.0 - (len(species_probabilities) - 1) * 0.1
        era5_bonus = weather_sources.get("era5_quality", 0.0)
        
        confidence_5d = confidence_5d_multi_species(
            weather_agreement=weather_quality,
            habitat_confidence=habitat_confidence,
            smi_reliability=0.9 if era5_data else 0.75,
            vpd_validity=(vpd_current <= 12.0),
            has_recent_validation=has_validations,
            coexistence_stability=coexistence_stability,
            era5_quality=era5_bonus
        )
        
        # Stime raccolto
        def estimate_harvest_multi_species(index, species_probs, confidence):
            base_harvest = index * confidence
            diversity_bonus = 1.0 + (len(species_probs) - 1) * 0.15
            total_harvest = base_harvest * diversity_bonus
            
            if total_harvest > 80: return "Eccellente", f"Ambiente ricco con {len(species_probs)} specie potenziali"
            elif total_harvest > 60: return "Buono", "Buone probabilità di raccolto diversificato"
            elif total_harvest > 40: return "Moderato", f"Possibile raccolta con {primary_species} dominante"
            elif total_harvest > 20: return "Scarso", "Condizioni subottimali"
            else: return "Molto scarso", "Attendere condizioni migliori"
        
        def estimate_sizes_multi_species(events, tmean, rh, species_probs):
            total_size, total_weight, size_ranges = 0.0, 0.0, []
            for species, prob in species_probs.items():
                if prob < 0.05: continue
                profile = SPECIES_PROFILES_V30[species]
                size = 11
                s_range = [7, 15]
                if tmean < 15 and rh > profile["humidity_requirement"]: size, s_range = 14, [10, 18]
                elif tmean > 20 or rh < profile["humidity_requirement"]: size, s_range = 8, [5, 12]
                total_size += size * prob
                total_weight += prob
                size_ranges.extend(s_range)
            avg_size = int(total_size / total_weight) if total_weight > 0 else 11
            overall_range = [min(size_ranges), max(size_ranges)] if size_ranges else [7, 15]
            return {"avg_size": avg_size, "size_class": "Variabile", "size_range": overall_range}

        harvest_estimate, harvest_note = estimate_harvest_multi_species(current_index, species_probabilities, confidence_5d["overall"])
        size_estimates = estimate_sizes_multi_species(flush_events_details, tmean_7d, rh_7d, species_probabilities)
        
        processing_time = round((time.time() - start_time) * 1000, 1)
        
        # Tabelle meteo
        weather_past_table = {time_series[i]: {"precipitation_mm": round(P_past[i], 1), "temp_min": round(Tmin_past[i], 1), "temp_max": round(Tmax_past[i], 1), "temp_mean": round(Tmean_past[i], 1)} for i in range(min(past_days, len(time_series)))}
        weather_future_table = { (time_series[past_days + i] if past_days + i < len(time_series) else f"+{i+1}d"): {"precipitation_mm": round(P_future[i], 1) if i < len(P_future) else 0.0, "temp_min": round(Tmin_series[past_days + i], 1) if past_days + i < len(Tmin_series) else 0.0, "temp_max": round(Tmax_series[past_days + i], 1) if past_days + i < len(Tmax_series) else 0.0, "temp_mean": round(Tmean_future[i], 1) if i < len(Tmean_future) else 0.0} for i in range(future_days)}
        
        # Response finale
        response_payload = {
            "lat": lat, "lon": lon, "elevation_m": round(elev_m), "slope_deg": round(slope_deg, 1),
            "aspect_deg": round(aspect_deg, 1), "aspect_octant": aspect_used or (aspect_oct or "N/A"), "aspect_source": aspect_source,
            "API_star_mm": round(api_value, 1), "P7_mm": round(sum(P_past[-7:]), 1), "P20_mm": round(sum(P_past), 1),
            "Tmean7_c": round(tmean_7d, 1), "RH7_pct": round(rh_7d, 1), "thermal_shock_index": round(thermal_shock, 2),
            "smi_current": round(smi_current, 2), "vpd_current_hpa": round(vpd_current, 1), "cumulative_moisture_index": round(cumulative_moisture_current, 1),
            "index": current_index, "forecast": forecast_final, "best_window": best_window, "confidence_detailed": confidence_5d,
            "species_analysis": {
                "primary_species": primary_species, "secondary_species": secondary_species,
                "species_probabilities": {k: round(v, 3) for k, v in species_probabilities.items()},
                "species_forecasts": species_forecasts, "species_lags": species_lags, "coexistence_scenario": coexistence_scenario,
                "coexistence_probability": round(1.0 - primary_probability, 2) if len(species_probabilities) > 1 else 0.0
            },
            "harvest_estimate": harvest_estimate, "harvest_note": harvest_note, "size_cm": size_estimates["avg_size"],
            "size_class": size_estimates["size_class"], "size_range_cm": size_estimates["size_range"],
            "habitat_used": habitat_used, "habitat_source": habitat_source, "habitat_confidence": round(habitat_confidence, 3),
            "flush_events": flush_events_details, "total_events_detected": len(flush_events_details),
            "weather_past": weather_past_table, "weather_future": weather_future_table,
            "has_local_validations": has_validations, "validation_count": validation_count,
            "model_version": "3.0.0", "model_type": "multi_species_coexistence", "processing_time_ms": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat(), "weather_sources": weather_sources, "weather_quality_score": round(weather_quality, 3),
            "diagnostics": {
                "era5_land_used": bool(era5_data), "species_count": len(species_probabilities), "coexistence_detected": coexistence_scenario != "dominanza_netta",
                "weather_enhancement": bool(era5_bonus > 0),
                "scientific_improvements": { "species_specific_thresholds": True, "borgotaro_model": True, "habitat_overlap_analysis": True }
            }
        }
        
        response_payload["dynamic_explanation"] = build_analysis_multi_species_v30(response_payload)
        
        if background_tasks:
            weather_metadata = {"api_value": api_value, "smi_current": smi_current, "era5_quality": era5_bonus, "weather_quality": weather_quality}
            background_tasks.add_task(save_prediction_multi_species, lat, lon, datetime.now().date().isoformat(), primary_species, secondary_species,
                response_payload["species_analysis"]["coexistence_probability"], current_index, confidence_5d, weather_metadata)

        logger.info(f"Analysis complete: {primary_species} ({primary_probability:.2f}) - {coexistence_scenario} ({processing_time}ms)")
        return response_payload

    except Exception as e:
        processing_time = round((time.time() - start_time) * 1000, 1)
        logger.exception(f"Error in multi-species analysis for ({lat:.5f}, {lon:.5f})")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ALTRI ENDPOINTS =====
@app.get("/api/health")
async def health():
    capabilities = { "numpy": NUMPY_AVAILABLE, "scipy": SCIPY_AVAILABLE, "geohash": GEOHASH_AVAILABLE, "cds": CDS_AVAILABLE }
    weather_sources = { "open_meteo": True, "visual_crossing": bool(VISUAL_CROSSING_KEY), "era5_land": bool(CDS_API_KEY and CDS_AVAILABLE), "hybrid_enabled": True }
    return { "ok": True, "time": datetime.now(timezone.utc).isoformat(), "version": "3.0.0", "model": "multi_species_coexistence_era5",
        "capabilities": capabilities, "weather_sources": weather_sources,
        "features": [ "multi_species_coexistence", "species_specific_thresholds", "era5_land_integration", "borgotaro_model", "scientific_documentation" ]
    }

@app.get("/api/geocode")
async def api_geocode(q: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = { "format": "json", "q": q, "addressdetails": 1, "limit": 1, "email": os.getenv("NOMINATIM_EMAIL", "info@porcinicast.com") }
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        if data:
            return { "lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display": data[0].get("display_name", ""), "source": "nominatim" }
    except Exception as e:
        logger.warning(f"Nominatim failed: {e}")
    
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": q, "count": 1, "language": "it"}
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            j = r.json()
        res = (j.get("results") or [])
        if not res: raise HTTPException(404, "Località non trovata")
        it = res[0]
        return { "lat": float(it["latitude"]), "lon": float(it["longitude"]), "display": f"{it.get('name')} ({(it.get('country_code') or '').upper()})", "source": "open_meteo" }
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        raise HTTPException(404, "Errore nel geocoding")

@app.post("/api/report-sighting")
async def report_sighting(
    lat: float, lon: float, species: str, 
    secondary_species: str = "", quantity: int = 1, size_cm_avg: float = None,
    confidence: float = 0.8, notes: str = "", habitat_observed: str = ""
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sightings 
            (lat, lon, date, species, secondary_species, quantity, size_cm_avg, 
             confidence, notes, habitat_observed, model_version, geohash,
             coexistence_predicted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, species, secondary_species or None, quantity, size_cm_avg,
              confidence, notes, habitat_observed, "3.0.0", geohash, bool(secondary_species)))
        
        conn.commit()
        conn.close()
        
        coex_text = f" + {secondary_species}" if secondary_species else ""
        logger.info(f"Multi-species sighting: {species}{coex_text} x{quantity} at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Segnalazione multi-specie registrata", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"Sighting error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.post("/api/report-no-findings")
async def report_no_findings(
    lat: float, lon: float, searched_hours: float = 2.0,
    search_method: str = "visual", habitat_searched: str = "", 
    notes: str = "", search_thoroughness: int = 3
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO no_sightings 
            (lat, lon, date, searched_hours, search_method, habitat_searched,
             notes, search_thoroughness, geohash, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, searched_hours, search_method, habitat_searched,
              notes, search_thoroughness, geohash, "3.0.0"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"No-finding report: {searched_hours}h at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Report registrato", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"No-finding error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.get("/api/validation-stats")
async def validation_stats_multi_species():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Stats base
        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM sightings")
        pos_stats = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM no_sightings")
        neg_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        
        # Stats coesistenza
        cursor.execute("SELECT COUNT(*) FROM sightings WHERE secondary_species IS NOT NULL")
        coexistence_sightings = cursor.fetchone()[0]
        cursor.execute("""
            SELECT species, secondary_species, COUNT(*) as count FROM sightings 
            WHERE secondary_species IS NOT NULL GROUP BY species, secondary_species ORDER BY count DESC LIMIT 5
        """)
        coexistence_pairs = [{"primary": row[0], "secondary": row[1], "count": row[2]} for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT species, COUNT(*) as count, AVG(quantity), AVG(size_cm_avg) FROM sightings 
            GROUP BY species ORDER BY count DESC LIMIT 5
        """)
        top_species = { s: { "count": c, "avg_quantity": round(aq or 0, 1), "avg_size_cm": round(asz or 0, 1) } for s, c, aq, asz in cursor.fetchall() }
        
        conn.close()
        total_validations = (pos_stats[0] or 0) + neg_count
        
        return {
            "positive_sightings": pos_stats[0] or 0, "negative_reports": neg_count, "predictions_logged": pred_count,
            "total_validations": total_validations, "coexistence_sightings": coexistence_sightings, "coexistence_pairs": coexistence_pairs,
            "avg_confidence": round(pos_stats[1] or 0, 2), "top_species_detailed": top_species, "ready_for_ml": total_validations >= 100,
            "model_version": "3.0.0",
            "multi_species_features": { "coexistence_tracking": True, "species_pair_analysis": True, "borgotaro_model_implemented": True, "era5_land_integration": bool(CDS_API_KEY) }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os, httpx, math, asyncio, tempfile, time, sqlite3, logging
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime, timezone, timedelta, date
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

# ===== PESO FENOLOGICO CONTINUO (DOY) =====
def phenology_weight(species: str, doy: int, elev_m: float) -> float:
    """
    Peso fenologico dolce [0..1] in funzione del giorno dell'anno e quota.
    Shift ~10 DOY / 1000 m per compensare ritardi altitudinali.
    """
    windows = {
        "reticulatus": (150, 230, 280),
        "edulis":      (220, 275, 330),
        "aereus":      (170, 240, 295),
        "pinophilus":  (210, 260, 310),
    }
    s, p, e = windows.get(species, (200, 260, 320))
    shift = int(round((elev_m or 0.0) / 1000.0 * 10.0))
    s += shift; p += shift; e += shift
    def ramp(x, x0, x1):
        if x <= x0: return 0.0
        if x >= x1: return 1.0
        return (x - x0) / (x1 - x0 + 1e-6)
    w_up = ramp(doy, s-20, p)
    w_down = 1.0 - ramp(doy, p, e+20)
    w = max(0.0, min(1.0, min(w_up, w_down)))
    return float(w)

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
    title="BoletusLab® v3.4.0 - Rilevamento Multi-Finestra & Budget Idrico",
    version="3.4.0",
    description="Sistema avanzato con data di chiusura intelligente basata su budget idrico/hazard e riconoscimento di finestre multiple."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

HEADERS = {"User-Agent":"BoletusLab/3.4.0 (+scientific)", "Accept-Language":"it"}
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
                model_version TEXT DEFAULT '3.4.0',
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
                model_version TEXT DEFAULT '3.4.0',
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
                model_version TEXT DEFAULT '3.4.0',
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
        "min_precip_flush": 10.5,
        "humidity_requirement": 85.0,
        "drought_days_limit": 4, # Specie termofila, tollera più giorni di stress idrico
        "description": "Porcino Nero, termofilo, preferisce querce e zone calde e secche"
    },
    "reticulatus": {
        "hosts": ["quercia", "castagno", "faggio", "misto"],
        "season": {"start_m": 5, "end_m": 9, "peak_m": [6, 7]},
        "tm7_opt": (16.0, 22.0), "tm7_critical": (10.0, 26.0),
        "lag_base": 7.8, "lag_range": (5, 10),
        "vpd_sens": 1.0, "drought_tolerance": 0.9,
        "elevation_opt": (100, 1300),
        "min_precip_flush": 9.0,
        "humidity_requirement": 80.0,
        "drought_days_limit": 3,
        "description": "Porcino Estivo, comune in faggete estive e querceti"
    },
    "edulis": {
        "hosts": ["faggio", "conifere", "misto"],
        "season": {"start_m": 8, "end_m": 11, "peak_m": [9, 10]},
        "tm7_opt": (12.0, 18.0), "tm7_critical": (6.0, 22.0),
        "lag_base": 10.2, "lag_range": (8, 14),
        "vpd_sens": 1.2, "drought_tolerance": 0.6,
        "elevation_opt": (500, 2000),
        "min_precip_flush": 7.5,
        "humidity_requirement": 90.0,
        "drought_days_limit": 2, # Specie igrofila, soffre subito la siccità
        "description": "Porcino Classico autunnale, ama il fresco e l'umido di faggi e abeti"
    },
    "pinophilus": {
        "hosts": ["conifere", "pino", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [8, 9]},
        "tm7_opt": (14.0, 20.0), "tm7_critical": (8.0, 24.0),
        "lag_base": 9.3, "lag_range": (7, 12),
        "vpd_sens": 0.9, "drought_tolerance": 1.1,
        "elevation_opt": (400, 1800),
        "min_precip_flush": 8.5,
        "humidity_requirement": 85.0,
        "drought_days_limit": 3,
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
        score_edulis = 1.0
        score_reticulatus = 1.0
        if month >= 9: score_edulis *= 1.5
        if elev_m > 1100: score_edulis *= 1.5
        if month < 9: score_reticulatus *= 1.5
        if elev_m <= 1100: score_reticulatus *= 1.5
        scores["edulis"] = score_edulis
        scores["reticulatus"] = score_reticulatus
    elif h == "quercia":
        scores["aereus"] = 1.0
    elif h in ["conifere", "pino"]:
        scores["pinophilus"] = 1.0
    elif h == "castagno":
        scores["reticulatus"] = 1.0
        scores["aereus"] = 0.7
    else:
        if "quercia" in h or "castagno" in h or h == "misto":
            scores["aereus"] = 1.0 if 6 <= month <= 9 and elev_m < 1000 else 0.1
        if "faggio" in h or "quercia" in h or "castagno" in h or h == "misto":
            scores["reticulatus"] = 1.0 if 5 <= month <= 9 else 0.2
        if "faggio" in h or "conifere" in h or h == "misto":
            scores["edulis"] = 1.0 if 8 <= month <= 11 and elev_m > 800 else 0.2
        if "conifere" in h or "pino" in h or h == "misto":
            scores["pinophilus"] = 0.9 if elev_m > 700 else 0.1

    total = sum(scores.values())
    if total > 0:
        probabilities = {species: score / total for species, score in scores.items()}
    else:
        return {"reticulatus": 1.0}
    
    significant_species = {k: v for k, v in probabilities.items() if v > 0.05}
    
    if not significant_species:
        if probabilities:
            most_probable = max(probabilities, key=probabilities.get)
            return {most_probable: 1.0}
        else:
            return {"reticulatus": 1.0}
        
    return significant_species


def determine_coexistence_scenario(species_probabilities: Dict[str, float]) -> str:
    """
    Determina lo scenario di coesistenza basato su letteratura
    """
    sorted_species = sorted(species_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_species or len(sorted_species) == 1:
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
        if probability > 0.05:
            profile = SPECIES_PROFILES_V30[species]
            
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
async def fetch_hybrid_weather_data(lat: float, lon: float, total_past_days: int = 20, 
                                   future_days: int = 10, enable_era5: bool = False) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching weather: Open-Meteo + ERA5-Land({enable_era5})")

        era5_task = None
        if enable_era5:
            era5_task = asyncio.create_task(_prefetch_era5l_sm_advanced(lat, lon, total_past_days, enable_era5))

        om_result = await fetch_open_meteo_recent_and_forecast(lat, lon, past=total_past_days, future=future_days)
        om_data_list = om_result.get("data", [])
        
        if not om_data_list or len(om_data_list) < total_past_days:
            raise RuntimeError(f"Open-Meteo insufficient data: {len(om_data_list)} rows")

        era5_data, era5_quality = None, 0.0
        if era5_task:
            try:
                await era5_task
                cache_key = f"{round(lat,3)},{round(lon,3)}"
                if cache_key in SM_CACHE:
                    era5_data, era5_quality = SM_CACHE[cache_key], 0.9
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
            "time": time_all, "precipitation_sum": P_all, "temperature_2m_min": Tmin_all,
            "temperature_2m_max": Tmax_all, "temperature_2m_mean": Tmean_all,
            "relative_humidity_2m_mean": RH_all, "et0_fao_evapotranspiration": ET0_all
        }

        completeness = min(1.0, len(P_all) / (total_past_days + future_days))
        base_quality = min(0.97, 0.9 + 0.07 * completeness)
        enhanced_quality = base_quality + (era5_quality * 0.1)

        return {
            "daily": daily_payload, "era5_data": era5_data,
            "metadata": {
                "sources": {
                    "open_meteo_past_days": min(total_past_days, len(P_all)),
                    "open_meteo_forecast_days": max(0, len(P_all) - total_past_days),
                    "era5_land_enabled": enable_era5, "era5_quality": era5_quality
                },
                "quality_score": min(0.99, enhanced_quality)
            }
        }
    except Exception as e:
        logger.error(f"Hybrid weather system failed: {e}")
        raise HTTPException(500, "Errore sistema meteorologico ibrido")

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
    
    if cache_key in _elev_cache and (time.time() - _elev_cache[cache_key].get("timestamp", 0) < 3600):
        return _elev_cache[cache_key]["grid"]
    
    try:
        deg_lat = step_m / 111320.0
        deg_lon = step_m / (111320.0 * max(0.2, math.cos(math.radians(lat))))
        
        coords = [{"latitude": lat + dy, "longitude": lon + dx} for dy in [-deg_lat, 0, deg_lat] for dx in [-deg_lon, 0, deg_lon]]
        
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
            r = await c.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": coords})
            r.raise_for_status()
            j = r.json()
        
        elevations = [p["elevation"] for p in j["results"]]
        grid = [elevations[0:3], elevations[3:6], elevations[6:9]]
        
        _elev_cache[cache_key] = {"grid": grid, "timestamp": time.time()}
        
        if len(_elev_cache) > 1000:
            oldest_keys = sorted(_elev_cache.keys(), key=lambda k: _elev_cache[k]["timestamp"])[:200]
            for k in oldest_keys: _elev_cache.pop(k, None)
        
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
        return round(slope_deg, 2), 0.0, None
    else:
        aspect_rad = math.atan2(-dzdx, dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
        octants = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        idx = int((aspect_deg + 22.5) // 45)
        octant = octants[idx] if slope_deg > 2.0 else None
        return round(slope_deg, 2), round(aspect_deg, 1), octant

def concavity_from_grid_super_advanced(grid: List[List[float]]) -> float:
    z, center = grid, grid[1][1]
    curvatures = [z[0][1] + z[2][1] - 2*center, z[1][0] + z[1][2] - 2*center,
                  z[0][0] + z[2][2] - 2*center, z[0][2] + z[2][0] - 2*center]
    mean_curvature = sum(curvatures) / len(curvatures)
    return clamp(mean_curvature / 10.0, -0.5, 0.5)

def drainage_proxy_from_grid_advanced(grid: List[List[float]]) -> float:
    z, center = grid, grid[1][1]
    draining_cells = sum(1 for i in range(3) for j in range(3) if not (i == 1 and j == 1) and z[i][j] > center)
    return clamp(draining_cells / 8.0, 0.1, 1.0)

# ===== OSM HABITAT =====
async def fetch_osm_habitat_super_advanced(lat: float, lon: float, radius_m: int = 500) -> Tuple[str, float, Dict[str, float]]:
    query = f"""[out:json][timeout:30];(way(around:{radius_m},{lat},{lon})["landuse"="forest"];way(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];relation(around:{radius_m},{lat},{lon})["landuse"="forest"];relation(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest)$"];node(around:{radius_m},{lat},{lon})["natural"="tree"];);out tags qt;"""
    urls = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=35, headers=HEADERS) as c:
                r = await c.post(url, data={"data": query})
                r.raise_for_status()
                j = r.json()
            scores = score_osm_elements_super_advanced(j.get("elements", []))
            habitat, confidence = choose_dominant_habitat_advanced(scores)
            logger.info(f"OSM habitat via {url}: {habitat} (conf: {confidence:.2f})")
            return habitat, confidence, scores
        except Exception as e:
            logger.warning(f"OSM URL {url} failed: {e}")
            continue
    logger.info("OSM failed, using heuristic")
    return "misto", 0.2, {}

def score_osm_elements_super_advanced(elements: List[Dict]) -> Dict[str, float]:
    scores = {"castagno": 0.0, "faggio": 0.0, "quercia": 0.0, "conifere": 0.0, "misto": 0.0}
    for element in elements:
        tags = {k.lower(): str(v).lower() for k, v in (element.get("tags", {})).items()}
        genus, species = tags.get("genus", ""), tags.get("species", "")
        if "castanea" in genus or "castagna" in species: scores["castagno"] += 4.0
        elif "quercus" in genus or "querce" in species: scores["quercia"] += 4.0
        elif "fagus" in genus or "faggio" in species: scores["faggio"] += 4.0
        elif any(g in genus for g in ["pinus", "picea", "abies", "larix"]): scores["conifere"] += 3.5
        if "needleleaved" in tags.get("leaf_type", ""): scores["conifere"] += 2.0
        elif "broadleaved" in tags.get("leaf_type", ""): scores["misto"] += 1.0
    return scores

def choose_dominant_habitat_advanced(scores: Dict[str, float]) -> Tuple[str, float]:
    total_score = sum(scores.values())
    if total_score < 0.5: return "misto", 0.15
    habitat, max_score = max(scores.items(), key=lambda x: x[1])
    dominance_ratio = max_score / total_score
    confidence = clamp(dominance_ratio ** 0.7 * 0.9, 0.1, 0.95)
    return habitat, confidence

# ===== EVENT DETECTION MULTI-SPECIE =====
def detect_rain_events_multi_species(rains: List[float], smi_series: List[float], month: int, elevation: float, lat: float, cumulative_moisture_series: List[float], months_series: Optional[List[int]] = None) -> List[Tuple[int, float, float]]:
    events, n, i = [], len(rains), 0
    while i < n:
        smi = smi_series[i] if i < len(smi_series) else 0.5
        c_moist = cumulative_moisture_series[i] if i < len(cumulative_moisture_series) else 0.0
        month_local = months_series[i] if months_series and i < len(months_series) else month
        thr1d = dynamic_rain_threshold_v30(8.5, smi, month_local, elevation, lat, 0.0, c_moist)
        if rains[i] >= thr1d:
            events.append((i, rains[i], event_strength_advanced(rains[i], antecedent_smi=smi))); i += 1; continue
        if i + 1 < n and rains[i] + rains[i+1] >= thr1d * 1.4:
            avg_smi = (smi + (smi_series[i+1] if i+1 < len(smi_series) else 0.5)) / 2
            events.append((i+1, rains[i] + rains[i+1], event_strength_advanced(rains[i] + rains[i+1], 36.0, avg_smi))); i += 2; continue
        if i + 2 < n and sum(rains[i:i+3]) >= thr1d * 1.8:
            avg_smi = sum(smi_series[i:i+3])/3 if i+2 < len(smi_series) else 0.5
            events.append((i+2, sum(rains[i:i+3]), event_strength_advanced(sum(rains[i:i+3]), 60.0, avg_smi))); i += 3; continue
        i += 1
    return events

def event_strength_advanced(mm: float, duration_hours: float = 24.0, antecedent_smi: float = 0.5) -> float:
    base = 1.0 - math.exp(-mm / 15.0)
    dur_factor = min(1.2, 1.0 + (duration_hours - 12.0) / 48.0)
    smi_factor = 0.7 + 0.6 * antecedent_smi
    return clamp(base * dur_factor * smi_factor, 0.0, 1.5)

def gaussian_kernel_advanced(x: float, mu: float, sigma: float, skewness: float = 0.0) -> float:
    base_gauss = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    if skewness != 0.0:
        skew_factor = 1.0 + skewness * ((x - mu) / sigma)
        return base_gauss * max(0.1, skew_factor)
    return base_gauss

# ===== DATABASE UTILS MULTI-SPECIE =====
def save_prediction_multi_species(lat: float, lon: float, date: str, primary_species: str, secondary_species: str,
                                 coexistence_prob: float, primary_score: int, confidence_data: dict, weather_data: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        geohash = geohash_encode_advanced(lat, lon, precision=8)
        cursor.execute('''INSERT INTO predictions (lat, lon, date, predicted_score, species, secondary_species, coexistence_probability, confidence_data, weather_data, model_version, geohash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (lat, lon, date, primary_score, primary_species, secondary_species, coexistence_prob, json.dumps(confidence_data), json.dumps(weather_data), "3.4.0", geohash))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")

def check_recent_validations_super_advanced(lat: float, lon: float, days: int = 30, radius_km: float = 15.0) -> Tuple[bool, int, float]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        lat_d, lon_d = radius_km / 111.0, radius_km / (111.0 * math.cos(math.radians(lat)))
        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
        cursor.execute('SELECT COUNT(*), AVG(confidence) FROM sightings WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND date >= ?', (lat-lat_d, lat+lat_d, lon-lon_d, lon+lon_d, cutoff))
        pos_count, pos_conf = cursor.fetchone()
        pos_count = pos_count or 0
        cursor.execute('SELECT COUNT(*) FROM no_sightings WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND date >= ?', (lat-lat_d, lat+lat_d, lon-lon_d, lon+lon_d, cutoff))
        neg_count = (cursor.fetchone() or [0])[0]
        conn.close()
        return (pos_count + neg_count) >= 3, pos_count + neg_count, pos_conf or 0.0
    except Exception as e:
        logger.error(f"Error checking validations: {e}")
        return False, 0, 0.0

# ===== NUOVA SEZIONE TESTUALE "INTELLIGENTE" v3.4.0 =====
class FruitingWindow(NamedTuple):
    start_index: int
    end_index: int
    peak_index: int
    peak_value: float
    
def _it_date(d: date, style: str = 'short') -> str:
    mesi = ['gen', 'feb', 'mar', 'apr', 'mag', 'giu', 'lug', 'ago', 'set', 'ott', 'nov', 'dic']
    return f"{d.day} {mesi[d.month-1]}"

def get_intelligent_closing_date(
    window: FruitingWindow, forecast: List[float], weather_data: Dict[str, Any], species: str,
    today: date, past_days: int
) -> int:
    """
    Stima la data di chiusura usando un modello ibrido: matematico, budget idrico e hazard.
    Ritorna l'indice del giorno di chiusura relativo all'inizio del forecast (giorno 0 = oggi).
    """
    n_forecast = len(forecast)
    
    # 1. Metodo Matematico (EWF)
    t_close_math = n_forecast + 2  # Fallback
    if window.end_index >= n_forecast - 1:
        k = min(4, n_forecast - window.peak_index if n_forecast > window.peak_index else 2)
        if k >= 2:
            tail_values = forecast[-k:]
            deltas = [tail_values[i] - tail_values[i-1] for i in range(1, len(tail_values))]
            avg_slope = sum(deltas) / len(deltas) if deltas else 0.0
            last_value = tail_values[-1]
            if avg_slope < -0.5:
                extra_days = math.ceil(max(1.0, (last_value - 15.0) / abs(avg_slope)))
                t_close_math = (n_forecast - 1) + extra_days
            else:
                t_close_math = (n_forecast - 1) + (4 if last_value > 50 else 3)

    # 2. Metodo Budget Idrico
    t_close_hydro = n_forecast + 5  # Fallback
    try:
        w_crit = {"edulis": 0.25, "reticulatus": 0.2, "aereus": 0.18, "pinophilus": 0.22}.get(species, 0.2)
        k_days_limit = SPECIES_PROFILES_V30.get(species, {}).get('drought_days_limit', 3)
        
        p_series = weather_data['daily']['precipitation_sum']
        et0_series = weather_data['daily']['et0_fao_evapotranspiration']
        
        # Inizializza il budget idrico odierno (semplificato)
        wt = 0.5 
        drought_days = 0
        
        # Simula per i prossimi 20 giorni a partire da oggi
        for i in range(n_forecast + 10):
            day_index_total = past_days + i
            p = p_series[day_index_total] if day_index_total < len(p_series) else 0.2
            et0 = et0_series[day_index_total] if day_index_total < len(et0_series) else 2.8
            
            # Semplice formula di bilancio idrico
            wt = clamp(wt + p / 20.0 - et0 / 50.0, 0, 1)

            if wt < w_crit:
                drought_days += 1
            else:
                drought_days = 0
            
            if drought_days >= k_days_limit:
                t_close_hydro = i
                break
    except Exception as e:
        logger.warning(f"Calcolo budget idrico fallito: {e}")

    # 3. Fusione
    final_closing_index = int(round(0.4 * t_close_math + 0.6 * t_close_hydro))
    return min(n_forecast + 15, final_closing_index) # Limita a 15 giorni extra per sicurezza

def build_analysis_multi_species_v340(payload: Dict[str, Any]) -> str:
    lines = []
    try:
        today = datetime.utcnow().date()
        species_info = payload['species_analysis']
        forecast = species_info['species_forecasts'].get(species_info['primary_species'], [])
        
        lines.append("<h4>🧬 Analisi Biologica Multi-Specie v3.4.0</h4>")
        lines.append("<p><em>Modello con data di chiusura intelligente (budget idrico) e rilevamento di finestre multiple.</em></p>")

        lines.append('<h4>🪵 Finestra di Fruttificazione (dinamica)</h4>')
        
        THR_MINIMA = 15.0
        
        windows: List[FruitingWindow] = []
        in_window = False
        start_idx = -1
        for i, val in enumerate(forecast):
            if val >= THR_MINIMA and not in_window:
                in_window = True
                start_idx = i
            elif val < THR_MINIMA and in_window:
                in_window = False
                end_idx = i - 1
                if end_idx >= start_idx:
                    window_slice = forecast[start_idx : end_idx + 1]
                    peak_val = max(window_slice)
                    peak_idx = start_idx + window_slice.index(peak_val)
                    windows.append(FruitingWindow(start_idx, end_idx, peak_idx, peak_val))
        if in_window:
            window_slice = forecast[start_idx:]
            peak_val = max(window_slice)
            peak_idx = start_idx + window_slice.index(peak_val)
            windows.append(FruitingWindow(start_idx, len(forecast) - 1, peak_idx, peak_val))

        current_window = next((w for w in windows if w.start_index <= 0 and w.end_index >= 0), None)
        next_window = next((w for w in windows if w.start_index > (current_window.end_index if current_window else -1)), None)

        if not current_window and not next_window:
            lines.append("<p><strong>Situazione</strong>: Nessuna finestra attiva o in arrivo nei prossimi 10 giorni. Condizioni non favorevoli.</p>")

        elif current_window and not next_window:
            d_open = today
            d_peak = today + timedelta(days=current_window.peak_index)
            t_close_final = get_intelligent_closing_date(current_window, forecast, payload, species_info['primary_species'], today, payload['past_days'])
            d_close = today + timedelta(days=t_close_final)
            lines.append(f"<p><strong>Situazione</strong>: Finestra attiva. <strong>Apertura</strong> già in corso, <strong>Picco</strong> attorno al {_it_date(d_peak)}, <strong>Chiusura stimata il {_it_date(d_close)}</strong>.</p>")

        elif not current_window and next_window:
            d_open = today + timedelta(days=next_window.start_index)
            d_peak = today + timedelta(days=next_window.peak_index)
            t_close_final = get_intelligent_closing_date(next_window, forecast, payload, species_info['primary_species'], today, payload['past_days'])
            d_close = today + timedelta(days=t_close_final)
            lines.append(f"<p><strong>Situazione</strong>: Nessuna finestra attiva. <strong>Prossima finestra in arrivo</strong>: <strong>Apertura</strong> il {_it_date(d_open)}, <strong>Picco</strong> attorno al {_it_date(d_peak)}, <strong>Chiusura stimata il {_it_date(d_close)}</strong>.</p>")
        
        elif current_window and next_window:
            d_peak_curr = today + timedelta(days=current_window.peak_index)
            d_close_curr = today + timedelta(days=current_window.end_index)
            lines.append("<h4>🍄 Finestra Attuale (in chiusura)</h4>")
            lines.append(f"<p><strong>Apertura</strong>: già attiva, <strong>Picco</strong> passato attorno al {_it_date(d_peak_curr)}, <strong>Chiusura</strong> imminente il {_it_date(d_close_curr)}.</p>")
            
            d_open_next = today + timedelta(days=next_window.start_index)
            d_peak_next = today + timedelta(days=next_window.peak_index)
            t_close_final_next = get_intelligent_closing_date(next_window, forecast, payload, species_info['primary_species'], today, payload['past_days'])
            d_close_next = today + timedelta(days=t_close_final_next)
            lines.append("<h4>🔮 Finestra Futura (in arrivo)</h4>")
            lines.append(f"<p><strong>Apertura prevista</strong> il {_it_date(d_open_next)}, con <strong>Picco</strong> attorno al {_it_date(d_peak_next)} e <strong>Chiusura stimata il {_it_date(d_close_next)}</strong>.</p>")

            lines.append("<h4>🎯 Consigli Pratici</h4><div class='return-advice'><p>Concentra la ricerca nelle prossime 24h per gli ultimi esemplari della finestra attuale. Preparati a tornare nei boschi a partire dal {_it_date(d_open_next)} per intercettare la nuova buttata.</p></div>")
            
        # Strategia di Ricerca Dettagliata
        # ... (Questa sezione può essere ulteriormente dettagliata se necessario)

    except Exception as e:
        logger.exception("Errore nella generazione del testo dinamico della finestra v3.4.0")
        lines.append("<p class='muted'>[Dettagli finestra non disponibili a causa di un errore interno]</p>")
    
    return "\n".join(lines)

# ===== ENDPOINT PRINCIPALE MULTI-SPECIE =====
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
    start_time = time.time()
    try:
        weather_data, elev_data, osm_habitat_data = await asyncio.gather(
            fetch_hybrid_weather_data(lat, lon, 20, 10, bool(use_era5)),
            fetch_elevation_grid_super_advanced(lat, lon),
            fetch_osm_habitat_super_advanced(lat, lon) if autohabitat == 1 else asyncio.sleep(0, result=("misto", 0.15, {}))
        )
        
        elev_m, slope_deg, _, aspect_oct, concavity, drainage_proxy = elev_data
        
        aspect_used, aspect_source = aspect_oct or "", "automatico DEM"
        if autoaspect == 0 and normalize_octant(aspect):
            aspect_used, aspect_source = normalize_octant(aspect), "manuale"
        
        habitat_used, habitat_source, habitat_confidence = (habitat or "").strip().lower(), "manuale", 0.6
        if autohabitat == 1:
            auto_habitat, auto_conf, _ = osm_habitat_data
            if not habitat_used and auto_habitat:
                habitat_used, habitat_confidence, habitat_source = auto_habitat, auto_conf, f"OSM (conf {auto_conf:.2f})"
        if not habitat_used: habitat_used = "misto"

        daily = weather_data["daily"]
        P_series, Tmin_series, Tmax_series, Tmean_series, RH_series, ET0_series = (
            daily["precipitation_sum"], daily["temperature_2m_min"], daily["temperature_2m_max"],
            daily["temperature_2m_mean"], daily["relative_humidity_2m_mean"], daily["et0_fao_evapotranspiration"]
        )
        
        past_days, future_days = 20, 10
        P_past, Tmean_past = P_series[:past_days], Tmean_series[:past_days]
        Tmean_future, RH_future = Tmean_series[past_days:], RH_series[past_days:]

        base_date = datetime.now(timezone.utc).date()
        months_series = [(base_date - timedelta(days=(past_days - 1 - i))).month if i < past_days else (base_date + timedelta(days=(i - past_days + 1))).month for i in range(len(P_series))]
        future_dates = [(base_date + timedelta(days=i+1)) for i in range(future_days)]
        doys_future = [d.timetuple().tm_yday for d in future_dates]
        
        api_value = api_index(P_past, half)
        smi_series = smi_from_p_et0_advanced(P_series, ET0_series, weather_data.get("era5_data"))
        smi_current = smi_series[past_days - 1]
        cumulative_moisture_series = cumulative_moisture_index(P_series)
        cumulative_moisture_current = cumulative_moisture_series[past_days - 1]
        
        tmean7 = sum(Tmean_past[-7:]) / 7
        thermal_shock = thermal_shock_index_advanced(Tmin_series[:past_days])
        vpd_current = vpd_hpa(Tmean_future[0], RH_future[0]) if Tmean_future and RH_future else 5.0

        month_current = base_date.month
        microclimate_energy = microclimate_energy_advanced(aspect_used or aspect_oct, slope_deg, month_current, lat, elev_m)
        
        species_probabilities = calculate_species_probabilities(habitat_used, month_current, elev_m, aspect_oct, lat)
        coexistence_scenario = determine_coexistence_scenario(species_probabilities)
        sorted_species = sorted(species_probabilities.items(), key=lambda x: x[1], reverse=True)
        primary_species = sorted_species[0][0]
        secondary_species = sorted_species[1][0] if len(sorted_species) > 1 and sorted_species[1][1] > 0.25 else ""
        
        species_lags = calculate_weighted_lag(species_probabilities, smi_current, thermal_shock, tmean7, vpd_current/10.0, cumulative_moisture_current)
        rain_events = detect_rain_events_multi_species(P_series, smi_series, month_current, elev_m, lat, cumulative_moisture_series, months_series)
        
        forecast_combined = [0.0] * future_days
        species_forecasts = {}
        
        for species, probability in species_probabilities.items():
            if probability < 0.05: continue
            species_forecast = [0.0] * future_days
            profile = SPECIES_PROFILES_V30[species]
            for event_idx, event_mm, event_strength in rain_events:
                lag_days = species_lags.get(species, 8)
                peak_idx = event_idx + lag_days
                amplitude = event_strength * microclimate_energy * probability * (min(1.6, max(0.6, (event_mm / profile["min_precip_flush"])**0.85)))
                
                # Durata dinamica
                sigma_base = 2.2 if event_strength > 0.8 else 1.8
                tm_opt_min, tm_opt_max = profile["tm7_opt"]
                giorno_pioggia, giorno_picco = event_idx, peak_idx
                mod_incubazione, mod_raccolta = 1.0, 1.0
                if giorno_picco < len(Tmean_series):
                    temps_incubazione = Tmean_series[giorno_pioggia:giorno_picco]
                    if temps_incubazione:
                        temp_media_incubazione = sum(temps_incubazione) / len(temps_incubazione)
                        if tm_opt_min < temp_media_incubazione < tm_opt_max: mod_incubazione = 1.15
                        elif temp_media_incubazione > tm_opt_max + 2.0 or temp_media_incubazione < tm_opt_min - 2.0: mod_incubazione = 0.85
                if giorno_picco < len(Tmean_series) - 4:
                    temps_raccolta = Tmean_series[giorno_picco : giorno_picco + 4]
                    if temps_raccolta:
                        temp_media_raccolta = sum(temps_raccolta) / len(temps_raccolta)
                        if temp_media_raccolta > tm_opt_max + 1.0: mod_raccolta = 0.80
                        elif temp_media_raccolta < tm_opt_min: mod_raccolta = 1.20
                sigma = clamp(sigma_base * mod_incubazione * mod_raccolta, sigma_base * 0.5, sigma_base * 1.5)
                
                for day_idx in range(future_days):
                    abs_day_idx = past_days + day_idx
                    kernel = gaussian_kernel_advanced(abs_day_idx, peak_idx, sigma)
                    pheno = phenology_weight(species, doys_future[day_idx], elev_m)
                    species_forecast[day_idx] += 100.0 * amplitude * kernel * pheno
            
            species_forecast_smoothed = savitzky_golay_advanced([clamp(v, 0, 100) for v in species_forecast])
            species_forecasts[species] = [int(round(x)) for x in species_forecast_smoothed]
            for i in range(future_days):
                forecast_combined[i] += species_forecast_smoothed[i] * probability
        
        forecast_final = [int(round(clamp(x, 0, 100))) for x in forecast_combined]
        current_index = forecast_final[0]
        
        threshold, fruiting_window_start, fruiting_window_end = 15, -1, -1
        if any(v > threshold for v in forecast_final):
            fruiting_window_start = next((i for i, v in enumerate(forecast_final) if v > threshold), -1)
            fruiting_window_end = len(forecast_final) - 1 - next((i for i, v in enumerate(reversed(forecast_final)) if v > threshold), len(forecast_final))

        has_validations, _, _ = check_recent_validations_super_advanced(lat, lon)
        confidence_5d = confidence_5d_multi_species(
            weather_agreement=weather_data['metadata']['quality_score'], habitat_confidence=habitat_confidence,
            smi_reliability=0.9 if weather_data.get("era5_data") else 0.75, vpd_validity=(vpd_current <= 12.0),
            has_recent_validation=has_validations
        )
        
        response_payload = {
            "lat": lat, "lon": lon, "elevation_m": round(elev_m), "slope_deg": round(slope_deg, 1),
            "aspect_octant": aspect_used or (aspect_oct or "N/A"),
            "index": current_index, "forecast": forecast_final,
            "fruiting_window_start": fruiting_window_start, "fruiting_window_end": fruiting_window_end,
            "confidence_detailed": confidence_5d,
            "species_analysis": {
                "primary_species": primary_species, "secondary_species": secondary_species,
                "species_probabilities": species_probabilities, "species_forecasts": species_forecasts, 
                "species_lags": species_lags, "coexistence_scenario": coexistence_scenario
            },
            "model_version": "3.4.0",
            # Passiamo dati necessari per i calcoli idrici nella funzione di analisi testuale
            "daily": daily,
            "past_days": past_days
        }
        
        response_payload["dynamic_explanation"] = build_analysis_multi_species_v340(response_payload)
        
        del response_payload["daily"] # Rimuoviamo i dati grezzi dalla risposta finale
        del response_payload["past_days"]

        if background_tasks:
            background_tasks.add_task(save_prediction_multi_species, lat, lon, datetime.now().date().isoformat(),
                                      primary_species, secondary_species, 1.0 - species_probabilities.get(primary_species, 1.0),
                                      current_index, confidence_5d, {"smi": smi_current})

        return response_payload

    except Exception as e:
        logger.exception(f"Error in multi-species analysis for ({lat:.5f}, {lon:.5f})")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "ok": True, "time": datetime.now(timezone.utc).isoformat(), "version": "3.4.0",
        "model": "multi_window_detection_hydric_budget",
        "features": ["multi_window_detection", "hydric_budget_closing_date", "stress_hazard_model"]
    }

@app.get("/api/geocode")
async def api_geocode(q: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"format": "json", "q": q, "addressdetails": 1, "limit": 1, "email": os.getenv("NOMINATIM_EMAIL", "info@porcinicast.com")}
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        if data: return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display": data[0].get("display_name", ""), "source": "nominatim"}
    except Exception: pass
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
        return {"lat": float(it["latitude"]), "lon": float(it["longitude"]), "display": f"{it.get('name')} ({(it.get('country_code') or '').upper()})", "source": "open_meteo"}
    except Exception as e: raise HTTPException(404, f"Errore nel geocoding: {e}")

@app.post("/api/report-sighting")
async def report_sighting(lat: float, lon: float, species: str, secondary_species: str = "", quantity: int = 1, size_cm_avg: float = None, confidence: float = 0.8, notes: str = "", habitat_observed: str = ""):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO sightings (lat, lon, date, species, secondary_species, quantity, size_cm_avg, confidence, notes, habitat_observed, model_version, geohash, coexistence_predicted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (lat, lon, datetime.now().date().isoformat(), species, secondary_species or None, quantity, size_cm_avg, confidence, notes, habitat_observed, "3.4.0", geohash_encode_advanced(lat, lon), bool(secondary_species)))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Segnalazione registrata", "id": cursor.lastrowid}
    except Exception as e: raise HTTPException(500, f"Errore interno: {e}")

@app.post("/api/report-no-findings")
async def report_no_findings(lat: float, lon: float, searched_hours: float = 2.0, search_method: str = "visual", habitat_searched: str = "", notes: str = "", search_thoroughness: int = 3):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO no_sightings (lat, lon, date, searched_hours, search_method, habitat_searched, notes, search_thoroughness, geohash, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (lat, lon, datetime.now().date().isoformat(), searched_hours, search_method, habitat_searched, notes, search_thoroughness, geohash_encode_advanced(lat, lon), "3.4.0"))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Report registrato", "id": cursor.lastrowid}
    except Exception as e: raise HTTPException(500, f"Errore interno: {e}")

@app.get("/api/validation-stats")
async def validation_stats_multi_species():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(confidence) FROM sightings"); pos_stats = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM no_sightings"); neg_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions"); pred_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sightings WHERE secondary_species IS NOT NULL"); coexistence_sightings = cursor.fetchone()[0]
        cursor.execute("""SELECT species, secondary_species, COUNT(*) as count FROM sightings WHERE secondary_species IS NOT NULL GROUP BY species, secondary_species ORDER BY count DESC LIMIT 5""")
        coexistence_pairs = [{"primary": row[0], "secondary": row[1], "count": row[2]} for row in cursor.fetchall()]
        cursor.execute("""SELECT species, COUNT(*) as count, AVG(quantity), AVG(size_cm_avg) FROM sightings GROUP BY species ORDER BY count DESC LIMIT 5""")
        top_species = {species: {"count": count, "avg_quantity": round(avg_qty or 0, 1), "avg_size_cm": round(avg_size or 0, 1)} for species, count, avg_qty, avg_size in cursor.fetchall()}
        conn.close()
        return {
            "positive_sightings": pos_stats[0] or 0, "negative_reports": neg_count, "predictions_logged": pred_count,
            "total_validations": (pos_stats[0] or 0) + neg_count, "coexistence_sightings": coexistence_sightings,
            "coexistence_pairs": coexistence_pairs, "avg_confidence": round(pos_stats[1] or 0, 2),
            "top_species_detailed": top_species, "ready_for_ml": ((pos_stats[0] or 0) + neg_count) >= 100,
            "model_version": "3.4.0"
        }
    except Exception as e: return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

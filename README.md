# 🍄 PorciniCast v2.5.0 - SUPER AVANZATO

Modello fenologico avanzato per previsione fruttificazione **Boletus edulis** s.l. basato su letteratura scientifica peer-reviewed.

## ✨ Funzionalità v2.5.0

### 🧬 **Modello Biologico Avanzato**
- **Lag biologico dinamico** basato su Boddy et al. (2014), Büntgen et al. (2012)
- **Soglie pluviometriche adattive** con feedback SMI, stagione, quota
- **Smoothing Savitzky-Golay** preserva-picchi con fallback intelligente
- **VPD specie-specifico** con correzioni altimetriche

### 🎯 **Sistema Confidence 5D**
- **Meteorologico**: accordo tra fonti multiple
- **Ecologico**: qualità inferenza habitat + validazioni campo
- **Idrologico**: affidabilità SMI + consistenza temporale  
- **Atmosferico**: validità VPD + correzioni altimetriche
- **Empirico**: presenza validazioni + densità dati locali

### 🗺️ **Microtopografia Multi-Scala**
- Calcolo slope/aspect da griglia 3x3 con algoritmo Horn (1981)
- Indice concavity per proxy di drenaggio
- TWI (Topographic Wetness Index) avanzato
- Energia microclimática con correzioni stagionali

### 🌿 **Habitat Inference Avanzata**
- **OSM Overpass** con scoring sofisticato genus/species/wood
- Fallback euristico geografico-altimetrico per Italia
- Confidence calibrata per qualità inferenza

### 💧 **SMI Multi-Sorgente**
- **P-ET0** avanzato con correzioni stagionali alpha
- **ERA5-Land** soil moisture (se disponibile CDS API)
- Normalizzazione percentile robusta

### 📊 **Crowd-Sourcing ML-Ready**
- Database SQLite con metadati completi
- Segnalazioni positive/negative con geohash
- Statistiche performance modello
- Auto-miglioramento basato su validazioni

## 🚀 Deploy

### Backend (Render)
```bash
# Deploy automatico via GitHub
# Usa Dockerfile per gestire dipendenze scientifiche
```

### Frontend (Netlify)
```bash
# Deploy automatico da /frontend
# Proxy API configurato in netlify.toml
```

## 🔑 Variabili Ambiente

```env
OPENWEATHER_API_KEY=your_key_here
CDS_API_KEY=your_cds_key_here  # Opzionale ma raccomandato
NOMINATIM_EMAIL=your_email_here
```

## 📚 Riferimenti Scientifici

- **Boddy, L. & Heilmann-Clausen, J. (2014)**. Basidiomycete community development in temperate angiosperm wood. *Fungal Ecology*.
- **Büntgen, U. et al. (2012)**. Drought-induced decline in Mediterranean tree growth. *Global Change Biology*.
- **Kauserud, H. et al. (2010)**. Climate change and spring-fruiting fungi. *Proceedings of the Royal Society B*.

## 🏗️ Architettura Tecnica

- **FastAPI** 0.104.1 - Backend API
- **NumPy/SciPy** - Calcoli scientifici avanzati
- **httpx** - HTTP client asincrono
- **SQLite** - Database embedded
- **Docker** - Containerizzazione per deploy

## 📊 Endpoints API

- `GET /api/health` - Status sistema
- `GET /api/geocode` - Geocoding località
- `GET /api/score` - **Endpoint principale** analisi
- `POST /api/report-sighting` - Segnala ritrovamento
- `POST /api/report-no-findings` - Segnala ricerca vuota
- `GET /api/validation-stats` - Statistiche crowd-sourcing

---

© 2025 PorciniCast v2.5.0 - Modello scientifico avanzato per previsione funghi
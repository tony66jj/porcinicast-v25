habitat_source = "manuale (override)"
        
        if not habitat_used:
            habitat_used = "misto"
        
        # === BLOCCO METEO OPENMETEO COMPLETO ===
        if not weather_data or "daily" not in weather_data:
            logger.error("Sistema meteorologico OpenMeteo completo fallito. Impossibile procedere.")
            raise HTTPException(500, "Errore dati meteorologici da sistema OpenMeteo completo")
        
        weather_quality = weather_data.get("metadata", {}).get("quality_score", 0.5)
        weather_sources = weather_data.get("metadata", {}).get("sources", {})
        
        logger.info(f"OpenMeteo complete system successful: quality={weather_quality:.3f}, sources={weather_sources}")
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
        
        past_days = 20  # NUOVO: 20 giorni storici invece di 15
        future_days = 10
        
        P_past = P_series[:past_days]
        P_future = P_series[past_days:past_days + future_days]
        Tmean_past = Tmean_series[:past_days]
        Tmean_future = Tmean_series[past_days:past_days + future_days]
        Tmin_past = Tmin_series[:past_days]
        Tmax_past = Tmax_series[:past_days]
        Tmin_future = Tmin_series[past_days:past_days + future_days] if past_days + future_days <= len(Tmin_series) else Tmin_series[past_days:] + [0.0] * (future_days - (len(Tmin_series) - past_days))
        Tmax_future = Tmax_series[past_days:past_days + future_days] if past_days + future_days <= len(Tmax_series) else Tmax_series[past_days:] + [0.0] * (future_days - (len(Tmax_series) - past_days))
        RH_past = RH_series[:past_days]
        RH_future = RH_series[past_days:past_days + future_days]
        
        # NUOVI INDICATORI SCIENTIFICI MIGLIORATI CON 20 GIORNI
        api_value = api_index(P_past, half_life=half)
        smi_series = smi_from_p_et0_advanced(P_series, ET0_series)
        smi_current = smi_series[past_days - 1] if past_days - 1 < len(smi_series) else 0.5
        
        # UMIDITÀ CUMULATIVA MIGLIORATA CON PIÙ DATI
        cumulative_moisture_series = cumulative_moisture_index(P_series, days_window=18)  # Finestra più ampia
        cumulative_moisture_current = cumulative_moisture_series[past_days - 1] if past_days - 1 < len(cumulative_moisture_series) else 0.0
        
        tmean_7d = sum(Tmean_past[-7:]) / max(1, len(Tmean_past[-7:]))
        # THERMAL SHOCK MIGLIORATO CON PIÙ DATI STORICI
        thermal_shock = thermal_shock_index_advanced(Tmin_past, window_days=5)  # Finestra più ampia
        rh_7d = sum(RH_past[-7:]) / max(1, len(RH_past[-7:]))
        vpd_series_future = [vpd_hpa(Tmean_future[i], RH_future[i]) for i in range(min(len(Tmean_future), len(RH_future)))]
        vpd_current = vpd_series_future[0] if vpd_series_future else 5.0
        
        # ESPOSIZIONE CORRETTA - Advanced microclimate
        month_current = datetime.now(timezone.utc).month
        microclimate_energy = microclimate_energy_advanced(aspect_used or aspect_oct, slope_deg, month_current, lat, elev_m)
        # Ponderazione esposizione: auto pesa meno (k=0.35), manuale pesa pieno (k=1.0)
        k_aspect = 0.35 if aspect_source.startswith("automatico") else 1.0
        microclimate_energy = blend_to_neutral(microclimate_energy, 1.0, k_aspect)
        twi_index = twi_advanced_proxy(slope_deg, concavity, drainage_proxy)
        
        # SPECIE INFERENCE
        species = infer_porcino_species_super_advanced(habitat_used, month_current, elev_m, aspect_oct, lat)
        species_profile = SPECIES_PROFILES_V26[species]
        
        logger.info(f"Species inferred: {species} for habitat {habitat_used} at {elev_m}m with 20d historical data")
        
        # RAIN EVENTS CON UMIDITÀ CUMULATIVA E 20 GIORNI STORICI
        rain_events = detect_rain_events_super_advanced_v26(
            P_past + P_future, smi_series, month_current, elev_m, lat, cumulative_moisture_series
        )
        
        forecast = [0.0] * future_days
        flush_events_details = []
        
        for event_idx, event_mm, event_strength in rain_events:
            smi_local = smi_series[event_idx] if event_idx < len(smi_series) else smi_current
            smi_adjusted = clamp(smi_local + species_profile["smi_bias"], 0.0, 1.0)
            cum_moisture_local = cumulative_moisture_series[event_idx] if event_idx < len(cumulative_moisture_series) else cumulative_moisture_current
            
            if event_idx >= past_days:
                future_idx = event_idx - past_days
                vpd_stress = max(0.0, (vpd_series_future[future_idx] - 8.0) / 10.0) if future_idx < len(vpd_series_future) else 0.0
            else:
                vpd_stress = 0.0
            
            # LAG BIOLOGICO SPECIE-SPECIFICO v2.5.7
            lag_days = stochastic_lag_super_advanced_v26(
                smi=smi_adjusted,
                thermal_shock=thermal_shock,
                tmean7=tmean_7d,
                species=species,
                vpd_stress=vpd_stress,
                photoperiod_factor=1.0,
                cumulative_moisture=cum_moisture_local
            )
            
            peak_idx = event_idx + lag_days
            base_amplitude = event_strength * microclimate_energy
            
            if event_idx >= past_days:
                future_peak_idx = peak_idx - past_days
                if 0 <= future_peak_idx < len(vpd_series_future):
                    vpd_penalty = vpd_penalty_advanced(vpd_series_future[future_peak_idx], species_profile["vpd_sens"], elev_m)
                else:
                    vpd_penalty = 0.8
            else:
                vpd_penalty = 1.0
            
            final_amplitude = base_amplitude * vpd_penalty
            
            # STEP FUNCTION - Sigma più stretto per fruttificazione più improvvisa
            sigma = 2.2 if event_strength > 0.8 else 1.8
            skew = 0.3 if species in ["aereus", "reticulatus"] else 0.1
            
            for day_idx in range(future_days):
                abs_day_idx = past_days + day_idx
                kernel_value = gaussian_kernel_advanced(abs_day_idx, peak_idx, sigma, skewness=skew)
                forecast[day_idx] += 100.0 * final_amplitude * kernel_value
            
            when_str = time_series[event_idx] if event_idx < len(time_series) else f"+{event_idx - past_days + 1}d"
            flush_events_details.append({
                "event_day_index": event_idx,
                "event_when": when_str,
                "event_mm": round(event_mm, 1),
                "event_strength": round(event_strength, 2),
                "lag_days": lag_days,
                "predicted_peak_abs_index": peak_idx,
                "observed": event_idx < past_days,
                "smi_local": round(smi_adjusted, 2),
                "vpd_penalty": round(vpd_penalty, 2),
                "cumulative_moisture": round(cum_moisture_local, 1)
            })
        
        # Advanced smoothing
        forecast_clamped = [clamp(v, 0.0, 100.0) for v in forecast]
        forecast_smoothed = savitzky_golay_advanced(forecast_clamped, window_length=5, polyorder=2)
        forecast_final = [int(round(x)) for x in forecast_smoothed]
        
        # Best window analysis
        best_window = {"start": 0, "end": 2, "mean": 0}
        if len(forecast_final) >= 3:
            best_mean = 0
            for i in range(len(forecast_final) - 2):
                window_mean = sum(forecast_final[i:i+3]) / 3.0
                if window_mean > best_mean:
                    best_mean = window_mean
                    best_window = {"start": i, "end": i+2, "mean": int(round(window_mean))}
        
        current_index = forecast_final[0] if forecast_final else 0
        
        # Validations and 5D confidence con qualità weather
        has_validations, validation_count, validation_accuracy = check_recent_validations_super_advanced(lat, lon)
        
        confidence_5d = confidence_5d_super_advanced(
            weather_agreement=weather_quality,  # NUOVO: usa qualità weather OpenMeteo completo
            habitat_confidence=habitat_confidence,
            smi_reliability=0.95 if CDS_AVAILABLE else 0.85,  # Migliorata con 20 giorni
            vpd_validity=(vpd_current <= 12.0),
            has_recent_validation=has_validations,
            elevation_reliability=0.9 if slope_deg > 1.0 else 0.7,
            temporal_consistency=0.9  # Migliorata con più dati storici
        )
        
        # Harvest and size estimates
        def estimate_harvest_super_advanced(index, hours, species, confidence):
            base_harvest = index * confidence * (hours / 4.0)
            if base_harvest > 80: return "Eccellente", "Raccolto potenzialmente molto abbondante con fruttificazione improvvisa."
            if base_harvest > 60: return "Buono", "Probabilità elevate di un buon raccolto dopo lag biologico."
            if base_harvest > 40: return "Moderato", "Raccolto possibile, ma dipende dall'umidità cumulativa."
            if base_harvest > 20: return "Scarso", "Probabilità di raccolto basse, servono ulteriori precipitazioni."
            return "Molto scarso", "Condizioni sfavorevoli, umidità insufficiente per fruttificazione."
        
        def estimate_mushroom_sizes_advanced(events, tmean, rh, species):
            profile = SPECIES_PROFILES_V26.get(species, SPECIES_PROFILES_V26["reticulatus"])
            if tmean < 15 and rh > profile["humidity_requirement"]: 
                return {"avg_size": 14, "size_class": "Grande", "size_range": [10, 18]}
            if tmean > 20 or rh < profile["humidity_requirement"]:
                return {"avg_size": 8, "size_class": "Piccolo", "size_range": [5, 12]}
            return {"avg_size": 11, "size_class": "Medio", "size_range": [7, 15]}

        harvest_estimate, harvest_note = estimate_harvest_super_advanced(current_index, hours, species, confidence_5d["overall"])
        size_estimates = estimate_mushroom_sizes_advanced(flush_events_details, tmean_7d, rh_7d, species)
        
        processing_time = round((time.time() - start_time) * 1000, 1)
        
        # Tabelle combinate meteo (pioggia + temperature) - AGGIORNATE PER 20 GIORNI
        weather_past_table = {}
        for i in range(min(past_days, len(time_series))):
            date_key = time_series[i]
            weather_past_table[date_key] = {
                "precipitation_mm": round(P_past[i], 1),
                "temp_min": round(Tmin_past[i], 1),
                "temp_max": round(Tmax_past[i], 1),
                "temp_mean": round(Tmean_past[i], 1)
            }
        
        weather_future_table = {}
        for i in range(future_days):
            date_key = time_series[past_days + i] if past_days + i < len(time_series) else f"+{i+1}d"
            weather_future_table[date_key] = {
                "precipitation_mm": round(P_future[i], 1),
                "temp_min": round(Tmin_future[i], 1) if i < len(Tmin_future) else 0.0,
                "temp_max": round(Tmax_future[i], 1) if i < len(Tmax_future) else 0.0,
                "temp_mean": round(Tmean_future[i], 1)
            }
        
        # Manteniamo le tabelle separate per retrocompatibilità 
        rain_past_table = {time_series[i]: round(P_past[i], 1) for i in range(min(past_days, len(time_series)))}
        rain_future_table = {
            time_series[past_days + i] if past_days + i < len(time_series) else f"+{i+1}d": round(P_future[i], 1) 
            for i in range(future_days)
        }
        temp_past_table = {time_series[i]: round(Tmean_past[i], 1) for i in range(min(past_days, len(time_series)))}
        temp_future_table = {time_series[past_days + i] if past_days + i < len(time_series) else f"+{i+1}d": round(Tmean_future[i],1) for i in range(future_days)}

        # Final response
        response_payload = {
            "lat": lat, "lon": lon,
            "elevation_m": round(elev_m),
            "slope_deg": round(slope_deg, 1),
            "aspect_deg": round(aspect_deg, 1),
            "aspect_octant": (aspect_used if aspect_source=="manuale" else (aspect_oct or "N/A")),
            "aspect_used": aspect_used or (aspect_oct or "N/A"),
            "aspect_source": aspect_source,
            "concavity": round(concavity, 3),
            "drainage_proxy": round(drainage_proxy, 2),
            
            "API_star_mm": round(api_value, 1),
            "P7_mm": round(sum(P_past[-7:]), 1),
            "P20_mm": round(sum(P_past), 1),  # NUOVO: P20 invece di P15
            "Tmean7_c": round(tmean_7d, 1),
            "RH7_pct": round(rh_7d, 1),
            "thermal_shock_index": round(thermal_shock, 2),
            "smi_current": round(smi_current, 2),
            "vpd_current_hpa": round(vpd_current, 1),
            "cumulative_moisture_index": round(cumulative_moisture_current, 1),
            
            "microclimate_energy": round(microclimate_energy, 2),
            "twi_index": round(twi_index, 2),
            
            "index": current_index,
            "forecast": forecast_final,
            "best_window": best_window,
            "confidence_detailed": confidence_5d,
            
            "harvest_estimate": harvest_estimate,
            "harvest_note": harvest_note,
            "size_cm": size_estimates["avg_size"],
            "size_class": size_estimates["size_class"], 
            "size_range_cm": size_estimates["size_range"],
            
            "habitat_used": habitat_used,
            "habitat_source": habitat_source,
            "habitat_confidence": round(habitat_confidence, 3),
            "auto_habitat_scores": auto_scores,
            "species": species,
            
            "flush_events": flush_events_details,
            "total_events_detected": len(rain_events),
            "events_observed": len([e for e in flush_events_details if e["observed"]]),
            "events_predicted": len([e for e in flush_events_details if not e["observed"]]),
            
            "rain_past": rain_past_table,
            "rain_future": rain_future_table,
            "temp_past": temp_past_table,
            "temp_future": temp_future_table,
            
            # NUOVO: Tabelle meteo combinate (pioggia + temperature)
            "weather_past": weather_past_table,
            "weather_future": weather_future_table,
            
            "has_local_validations": has_validations,
            "validation_count": validation_count,
            "validation_accuracy": round(validation_accuracy, 2),
            
            "model_version": "2.5.7",
            "model_type": "openmeteo_archive_forecast_complete",
            "processing_time_ms": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # NUOVO: Metadata sistema weather OpenMeteo completo
            "weather_sources": weather_sources,
            "weather_quality_score": round(weather_quality, 3),
            
            "diagnostics": {
                "smi_source": "P-ET0 advanced" + (" + ERA5" if CDS_AVAILABLE else ""),
                "weather_primary_source": weather_sources.get("primary_source", "openmeteo_complete"),
                "weather_backup_used": weather_sources.get("backup_used", False),
                "weather_data_completeness": weather_sources.get("completeness", 0.0),
                "historical_days": past_days,  # NUOVO: 20 giorni
                "elevation_quality": "multi_scale_grid",
                "habitat_method": habitat_source,
                "lag_algorithm": "stochastic_v26_specie_specific",
                "smoothing_method": "savitzky_golay_advanced" if SCIPY_AVAILABLE else "custom_advanced",
                "confidence_system": "5d_multidimensional_openmeteo_complete",
                "thresholds": "dynamic_adaptive_v26_cumulative",
                "scientific_improvements": {
                    "exposure_lag": bool(advanced_lag),
                    "lag_specie_specifico": True,
                    "umidita_cumulativa": True,
                    "esposizione_corretta": True,
                    "step_function": True,
                    "openmeteo_archive_forecast": True,
                    "20_days_historical": True
                },
                "capabilities": {
                    "numpy": NUMPY_AVAILABLE,
                    "scipy": SCIPY_AVAILABLE,
                    "geohash": GEOHASH_AVAILABLE,
                    "cds": CDS_AVAILABLE,
                    "visual_crossing": bool(VISUAL_CROSSING_KEY)
                }
            }
        }
        
        # Analisi scientifica aggiornata
        response_payload["dynamic_explanation"] = build_analysis_openmeteo_complete_v27(response_payload, species_profile)
        
        # Save prediction for ML
        if background_tasks:
            weather_metadata = {
                "api_value": api_value, "smi_current": smi_current,
                "tmean_7d": tmean_7d, "thermal_shock": thermal_shock,
                "vpd_current": vpd_current, "processing_time_ms": processing_time,
                "cumulative_moisture": cumulative_moisture_current,
                "weather_quality": weather_quality,
                "weather_sources": weather_sources,
                "historical_days": past_days
            }
            model_features = {
                "elevation": elev_m, "slope": slope_deg, "aspect": aspect_oct,
                "microclimate_energy": microclimate_energy, "twi_index": twi_index,
                "species": species, "events_count": len(rain_events),
                "lag_range": species_profile["lag_range"],
                "humidity_requirement": species_profile["humidity_requirement"],
                "days_historical": past_days
            }
            
            background_tasks.add_task(
                save_prediction_super_advanced,
                lat, lon, datetime.now().date().isoformat(),
                current_index, species, habitat_used,
                confidence_5d, weather_metadata, model_features
            )

        primary_source = weather_sources.get("primary_source", "openmeteo_complete")
        backup_status = "backup attivo" if weather_sources.get("backup_used", False) else "completo"
        
        logger.info(
            f"OpenMeteo complete analysis finished: {current_index}/100 for {species} "
            f"(quality: {weather_quality:.3f}, source: {primary_source} {backup_status}, 20d historical, {processing_time}ms)"
        )
        return response_payload

    except Exception as e:
        # tempo di elaborazione fino al fallimento
        processing_time = round((time.time() - start_time) * 1000, 1)

        # log con stack-trace completo
        logger.exception(f"Error in OpenMeteo complete /api/score for ({lat:.5f}, {lon:.5f})")

        # risposta JSON coerente col resto dell'API
        raise HTTPException(status_code=500, detail=str(e))

# ===== ALTRI ENDPOINTS =====
@app.get("/api/health")
async def health():
    capabilities = {
        "numpy": NUMPY_AVAILABLE,
        "scipy": SCIPY_AVAILABLE, 
        "geohash": GEOHASH_AVAILABLE,
        "cds": CDS_AVAILABLE
    }
    
    weather_sources = {
        "open_meteo_archive": True,
        "open_meteo_forecast": True,
        "visual_crossing": bool(VISUAL_CROSSING_KEY),
        "primary_strategy": "openmeteo_archive_forecast_complete",
        "backup_strategy": "visual_crossing_emergencial_fallback",
        "historical_days": 20
    }
    
    return {
        "ok": True, 
        "time": datetime.now(timezone.utc).isoformat(), 
        "version": "2.5.7",
        "model": "openmeteo_archive_forecast_complete",
        "capabilities": capabilities,
        "weather_sources": weather_sources,
        "features": [
            "lag_biologico_specie_specifico", "umidita_cumulativa", "esposizione_corretta",
            "step_function_fruttificazione", "soglie_dinamiche_memoria", "confidence_5d_evolutiva",
            "openmeteo_archive_forecast_complete", "visual_crossing_backup_emergencial", 
            "20_days_historical_data", "data_quality_scoring_enhanced",
            "crowd_sourcing", "savitzky_golay", "microtopografia_avanzata",
            "era5_land_smi", "osm_habitat_avanzato"
        ]
    }

@app.get("/api/geocode")
async def api_geocode(q: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "format": "json", "q": q, "addressdetails": 1, "limit": 1,
            "email": os.getenv("NOMINATIM_EMAIL", "info@porcinicast.com")
        }
        async with httpx.AsyncClient(timeout=20, headers=HEADERS) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0].get("display_name", ""),
                "source": "nominatim"
            }
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
        if not res: 
            raise HTTPException(404, "Località non trovata")
        
        it = res[0]
        return {
            "lat": float(it["latitude"]),
            "lon": float(it["longitude"]),
            "display": f"{it.get('name')} ({(it.get('country_code') or '').upper()})",
            "source": "open_meteo"
        }
    except Exception as e:
        logger.error(f"Geocoding completely failed: {e}")
        raise HTTPException(404, "Errore nel geocoding")

@app.post("/api/report-sighting")
async def report_sighting(
    lat: float, lon: float, species: str, 
    quantity: int = 1, size_cm_avg: float = None, size_cm_max: float = None,
    confidence: float = 0.8, photo_url: str = "", notes: str = "",
    habitat_observed: str = "", weather_conditions: str = "",
    user_experience_level: int = 3
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sightings 
            (lat, lon, date, species, quantity, size_cm_avg, size_cm_max, confidence, 
             photo_url, notes, habitat_observed, weather_conditions, user_experience_level,
             geohash, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, species, quantity, size_cm_avg, size_cm_max, confidence,
              photo_url, notes, habitat_observed, weather_conditions, user_experience_level,
              geohash, "2.5.7"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Advanced sighting: {species} x{quantity} at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Segnalazione registrata con successo", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"Sighting error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.post("/api/report-no-findings")
async def report_no_findings(
    lat: float, lon: float, searched_hours: float = 2.0,
    search_method: str = "visual", habitat_searched: str = "", 
    weather_conditions: str = "", notes: str = "",
    search_thoroughness: int = 3
):
    try:
        date = datetime.now().date().isoformat()
        geohash = geohash_encode_advanced(lat, lon)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO no_sightings 
            (lat, lon, date, searched_hours, search_method, habitat_searched,
             weather_conditions, notes, search_thoroughness, geohash, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, searched_hours, search_method, habitat_searched,
              weather_conditions, notes, search_thoroughness, geohash, "2.5.7"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Advanced no-finding: {searched_hours}h at ({lat:.4f}, {lon:.4f})")
        return {"status": "success", "message": "Report registrato con successo", "id": cursor.lastrowid}
        
    except Exception as e:
        logger.error(f"No-finding error: {e}")
        raise HTTPException(500, "Errore interno del server")

@app.get("/api/validation-stats")
async def validation_stats_super_advanced():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), AVG(confidence), AVG(user_experience_level) FROM sightings")
        pos_stats = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*), AVG(search_thoroughness) FROM no_sightings")
        neg_stats = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT species, COUNT(*) as count, AVG(quantity), AVG(size_cm_avg)
            FROM sightings 
            WHERE size_cm_avg IS NOT NULL
            GROUP BY species 
            ORDER BY count DESC 
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

# ===== OSM HABITAT SUPER AVANZATO =====
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
    
    logger.info("OSM failed, using advanced heuristic")
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

# ===== EVENT DETECTION CON UMIDITÀ CUMULATIVA =====
def detect_rain_events_super_advanced_v26(rains: List[float], smi_series: List[float], 
                                         month: int, elevation: float, lat: float,
                                         cumulative_moisture_series: List[float]) -> List[Tuple[int, float, float]]:
    """MODIFICA: Include umidità cumulativa nella rilevazione eventi"""
    events = []
    n = len(rains)
    i = 0
    
    while i < n:
        smi_local = smi_series[i] if i < len(smi_series) else 0.5
        cum_moisture = cumulative_moisture_series[i] if i < len(cumulative_moisture_series) else 0.0
        temp_trend = 0.0  # placeholder
        
        threshold_1d = dynamic_rain_threshold_v26(smi_local, month, elevation, lat, temp_trend, cum_moisture)
        threshold_2d = threshold_1d * 1.4
        threshold_3d = threshold_1d * 1.8
        
        if rains[i] >= threshold_1d:
            strength = event_strength_advanced(rains[i], antecedent_smi=smi_local)
            events.append((i, rains[i], strength))
            i += 1
            continue
        
        if i + 1 < n:
            rain_2d = rains[i] + rains[i + 1]
            if rain_2d >= threshold_2d:
                avg_smi = (smi_local + (smi_series[i+1] if i+1 < len(smi_series) else 0.5)) / 2
                strength = event_strength_advanced(rain_2d, duration_hours=36.0, antecedent_smi=avg_smi)
                events.append((i + 1, rain_2d, strength))
                i += 2
                continue
        
        if i + 2 < n:
            rain_3d = rains[i] + rains[i + 1] + rains[i + 2]
            if rain_3d >= threshold_3d:
                avg_smi = sum(smi_series[i:i+3]) / 3 if i+2 < len(smi_series) else 0.5
                strength = event_strength_advanced(rain_3d, duration_hours=60.0, antecedent_smi=avg_smi)
                events.append((i + 2, rain_3d, strength))
                i += 3
                continue
        
        i += 1
    
    return events

# ===== DATABASE UTILS SUPER AVANZATI =====
def save_prediction_super_advanced(lat: float, lon: float, date: str, score: int, 
                                  species: str, habitat: str, confidence_data: dict,
                                  weather_data: dict, model_features: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        geohash = geohash_encode_advanced(lat, lon, precision=8)
        
        cursor.execute('''
            INSERT INTO predictions 
            (lat, lon, date, predicted_score, species, habitat, confidence_data, 
             weather_data, model_features, model_version, geohash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (lat, lon, date, score, species, habitat, 
              json.dumps(confidence_data), json.dumps(weather_data),
              json.dumps(model_features), "2.5.7", geohash))
        
        conn.commit()
        conn.close()
        logger.info(f"Prediction saved: {score}/100 for {species}")
        
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
            AND date >= ? AND validation_status != 'rejected'
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

# ===== ANALISI TESTUALE OPENMETEO COMPLETO AGGIORNATA =====
def build_analysis_openmeteo_complete_v27(payload: Dict[str, Any], species_profile: Dict[str, Any]) -> str:
    idx = payload["index"]
    weather_sources = payload.get("weather_sources", {})
    weather_quality = payload.get("weather_quality_score", 0.5)
    
    lines = []
    
    lines.append("<h4>🌦️ Analisi OpenMeteo Completo v2.5.7</h4>")
    lines.append(f"<p><em>Sistema OpenMeteo Archive API (20 giorni storici) + Forecast API (10 giorni) • Visual Crossing solo backup emergenziale</em></p>")
    
    # Qualità dati meteorologici
    primary_source = weather_sources.get("primary_source", "unknown")
    backup_used = weather_sources.get("backup_used", False)
    
    lines.append(f"<h4>📊 Qualità Dati Meteorologici Avanzata</h4>")
    if backup_used:
        vc_hist_days = weather_sources.get("visual_crossing_historical_days", 0)
        vc_forecast_days = weather_sources.get("visual_crossing_forecast_days", 0)
        lines.append(f"<p><strong>⚠️ Modalità Backup Visual Crossing Attiva</strong>:</p>")
        lines.append("<ul style='margin:8px 0 0 20px'>")
        lines.append(f"<li><strong>Visual Crossing storico</strong>: {vc_hist_days} giorni (backup emergenziale)</li>")
        lines.append(f"<li><strong>Visual Crossing forecast</strong>: {vc_forecast_days} giorni (backup emergenziale)</li>")
        lines.append("<li><strong>OpenMeteo Archive+Forecast</strong>: Non disponibile (errore di sistema)</li>")
        lines.append("</ul>")
    else:
        archive_days = weather_sources.get("archive_days", 0)
        forecast_days = weather_sources.get("forecast_days", 0)
        total_days = weather_sources.get("total_days", 0)
        lines.append(f"<p><strong>✅ OpenMeteo Completo Attivo</strong>:</p>")
        lines.append("<ul style='margin:8px 0 0 20px'>")
        lines.append(f"<li><strong>Archive API</strong>: {archive_days} giorni storici (fino a 20d)</li>")
        lines.append(f"<li><strong>Forecast API</strong>: {forecast_days} giorni futuri</li>")
        lines.append(f"<li><strong>Totale dati</strong>: {total_days} giorni (qualità massima)</li>")
        lines.append("<li><strong>Visual Crossing</strong>: Standby (non utilizzato)</li>")
        lines.append("</ul>")
    
    quality_color = "#66e28a" if weather_quality >= 0.9 else "#ffc857" if weather_quality >= 0.7 else "#ff6b6b"
    completeness = weather_sources.get("completeness", 0.0)
    lines.append(f"<p><strong>Qualità complessiva</strong>: <span style='color:{quality_color};font-weight:bold'>{weather_quality:.3f}</span>/1.000 (completezza: {completeness:.1%})</p>")
    
    # Resto dell'analisi scientifica
    lines.append(build_analysis_scientifically_corrected_v26(payload, species_profile))
    
    # Sezione aggiornata sul sistema OpenMeteo completo
    lines.append(f"<h4>🔄 Sistema Meteorologico OpenMeteo Completo v2.5.7</h4>")
    lines.append("<div style='background:#0a0f14;padding:12px;border-radius:8px;border-left:3px solid #62d5b4'>")
    lines.append("<ul style='margin:0;padding-left:20px'>")
    lines.append("<li><strong>Archive API</strong>: Dati storici fino a 20 giorni (vs 7 giorni Forecast API) per analisi SMI e thermal shock avanzate</li>")
    lines.append("<li><strong>Forecast API</strong>: Previsioni 10 giorni con alta accuratezza per l'Europa</li>")
    lines.append("<li><strong>Backup intelligente</strong>: Visual Crossing attivato solo per emergenze sistemiche</li>")
    lines.append("<li><strong>Qualità ottimizzata</strong>: OpenMeteo Archive+Forecast = 0.98/1.00 vs backup = 0.80/1.00</li>")
    lines.append(f"<li><strong>Vantaggi 20 giorni</strong>: Migliore SMI, thermal shock detection, pattern recognition, umidità cumulativa</li>")
    lines.append("</ul>")
    lines.append("</div>")
    
    return "\n".join(lines)

def build_analysis_scientifically_corrected_v26(payload: Dict[str, Any], species_profile: Dict[str, Any]) -> str:
    idx = payload["index"]
    best = payload.get("best_window", {})
    elev = payload["elevation_m"]
    slope = payload["slope_deg"]
    aspect = payload.get("aspect_octant", "N/A")
    habitat_used = (payload.get("habitat_used") or "").capitalize() or "Misto"
    species = payload.get("species", "reticulatus")
    
    confidence_detailed = payload.get("confidence_detailed", {})
    overall_conf = confidence_detailed.get("overall", 0.0)
    
    flush_events = payload.get("flush_events", [])
    lag_info = f"{len([e for e in flush_events if e.get('observed')])} osservati, {len([e for e in flush_events if not e.get('observed')])} previsti"
    
    lines = []
    
    lines.append("<h4>🧬 Analisi Biologica Scientificamente Corretta v2.5.7</h4>")
    lines.append(f"<p><em>Modello fenologico con lag specie-specifico e umidità cumulativa basato su: Viitanen (1997), Frontiers Soil Science (2023), ScienceDirect (2017)</em></p>")
    
    lines.append(f"<h4>🍄 Specie e Habitat (Esposizione Corretta)</h4>")
    lines.append(f"<p><strong>Specie dominante predetta</strong>: <em>Boletus {species}</em></p>")
    lines.append(f"<p><strong>Habitat principale</strong>: {habitat_used} • <strong>Localizzazione</strong>: {elev}m, pendenza {slope}°, esposizione {aspect}</p>")
    
    profile = species_profile
    season_text = f"{profile['season']['start_m']:02d}→{profile['season']['end_m']:02d}"
    lines.append(f"<p><strong>Ecologia specie scientifica</strong>: Stagione {season_text} • Lag biologico base ~{profile['lag_base']:.1f} giorni • VPD sensibilità {profile['vpd_sens']:.1f} • RH richiesta ≥{profile['humidity_requirement']:.0f}%</p>")
    
    lines.append(f"<h4>📊 Indice e Previsione Step-Function</h4>")
    lines.append(f"<p><strong>Indice corrente</strong>: <strong style='font-size:1.2em'>{idx}/100</strong> - ")
    
    if idx >= 75:
        lines.append("<span style='color:#66e28a;font-weight:bold'>ECCELLENTE</span> - Fruttificazione massiva improvvisa attesa (step-function)")
    elif idx >= 60:
        lines.append("<span style='color:#8bb7ff;font-weight:bold'>MOLTO BUONE</span> - Fruttificazione abbondante dopo lag biologico")
    elif idx >= 45:
        lines.append("<span style='color:#ffc857;font-weight:bold'>BUONE</span> - Fruttificazione moderata possibile")
    elif idx >= 30:
        lines.append("<span style='color:#ff9966;font-weight:bold'>MODERATE</span> - Fruttificazione limitata")
    else:
        lines.append("<span style='color:#ff6b6b;font-weight:bold'>SCARSE</span> - Fruttificazione improbabile")
    lines.append("</p>")
    
    if best and best.get("mean", 0) > 0:
        start, end, mean = best.get("start", 0), best.get("end", 0), best.get("mean", 0)
        lines.append(f"<p><strong>Finestra ottimale prossimi 10 giorni</strong>: Giorni <strong>{start+1}-{end+1}</strong> (indice medio ~<strong>{mean}</strong>)</p>")
    
    lines.append(f"<h4>🎯 Affidabilità Multi-Dimensionale</h4>")
    lines.append("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:12px 0'>")
    
    for dimension, value in confidence_detailed.items():
        if dimension == "overall": continue
        color = "#66e28a" if value >= 0.7 else "#ffc857" if value >= 0.5 else "#ff6b6b"
        dim_name = {
            "meteorological": "☁️ Meteorol.",
            "ecological": "🌿 Ecologica", 
            "hydrological": "💧 Idrologica",
            "atmospheric": "🌡️ Atmosferica",
            "empirical": "📊 Empirica"
        }.get(dimension, dimension.title())
        lines.append(f"<div style='text-align:center;padding:8px;background:#0a0f14;border-radius:6px'><div style='color:{color};font-weight:bold'>{value:.2f}</div><div style='font-size:11px;color:#8aa0b6'>{dim_name}</div></div>")
    
    lines.append("</div>")
    lines.append(f"<p><strong>Affidabilità complessiva</strong>: <strong style='color:{'#66e28a' if overall_conf >= 0.7 else '#ffc857' if overall_conf >= 0.5 else '#ff6b6b'}'>{overall_conf:.2f}</strong>/1.00</p>")
    
    lines.append(f"<h4>⏱️ Eventi Piovosi, Lag Specie-Specifico e Umidità Cumulativa</h4>")
    lines.append(f"<p><strong>Eventi rilevati</strong>: {lag_info}</p>")
    
    if flush_events:
        lines.append("<ul style='margin:8px 0 0 20px'>")
        for event in flush_events[:3]:
            when = event.get("event_when", "?")
            mm = event.get("event_mm", 0)
            lag = event.get("lag_days", 0)
            obs_text = "📊 Osservato" if event.get("observed") else "🔮 Previsto"
            strength = event.get("event_strength", 0)
            lines.append(f"<li><strong>{when}</strong>: {mm:.1f}mm → flush ~{lag} giorni (forza: {strength:.2f}) ({obs_text})</li>")
        lines.append("</ul>")
        
        # Mostra variabilità lag biologico
        lag_values = [e.get("lag_days", 7) for e in flush_events if e.get("lag_days")]
        if lag_values and len(set(lag_values)) > 1:
            min_lag, max_lag = min(lag_values), max(lag_values)
            lines.append(f"<p class='lag-explanation' style='background:#0e1419;border:1px solid #62d5b4;border-radius:8px;padding:10px;margin:8px 0;font-size:11px;color:#b9f3cf'>")
            lines.append(f"💡 <strong>Lag Biologico Specie-Specifico Attivo:</strong> Range {min_lag}-{max_lag} giorni per <em>B. {species}</em> basato su umidità suolo, shock termico, temperatura ottimale e stress VPD secondo letteratura.</p>")
    
    return "\n".join(lines)

# ===== ENDPOINT PRINCIPALE SISTEMA OPENMETEO COMPLETO =====
@app.get("/api/score")
async def api_score_openmeteo_complete(
    lat: float = Query(..., description="Latitudine"),
    lon: float = Query(..., description="Longitudine"),
    half: float = Query(8.5, gt=3.0, lt=20.0, description="Half-life API (giorni)"),
    habitat: str = Query("", description="Habitat forzato"),
    autohabitat: int = Query(1, description="1=auto OSM, 0=manuale"),
    hours: int = Query(4, ge=2, le=8, description="Ore sul campo"),
    aspect: str = Query("", description="Esposizione manuale (N, NE, E, SE, S, SW, W, NW)"),
    autoaspect: int = Query(1, description="1=automatico DEM, 0=manuale"),
    advanced_lag: int = Query(0, description="1=abilita lag biologico modulato da esposizione"),
    background_tasks: BackgroundTasks = None
):
    """
    🚀 ENDPOINT PRINCIPALE OPENMETEO COMPLETO v2.5.7
    OpenMeteo Archive API (20 giorni storici) + Forecast API (10 giorni)
    Visual Crossing solo backup emergenziale
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting OpenMeteo complete analysis for ({lat:.4f}, {lon:.4f})")
        
        # Prefetch ERA5-Land in background se disponibile
        if CDS_AVAILABLE:
            asyncio.create_task(_prefetch_era5l_sm_advanced(lat, lon))
        
        # Fetch paralleli con sistema OpenMeteo completo (20+10 giorni)
        tasks = [
            fetch_weather_data_openmeteo_complete(lat, lon, past_days=20, future_days=10),
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
        
        # Esposizione: auto (DEM) vs manuale
        aspect_source = "automatico DEM"
        aspect_used = aspect_oct or ""
        if autoaspect == 0:
            manual_oct = normalize_octant(aspect)
            if manual_oct:
                aspect_used = manual_oct
                aspect_source = "manuale"
        
        # Habitat determination
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
                habitat_source = "    """Return a multiplicative lag factor considering species, aspect and slope.
    Dampen effect on gentle slopes and in flat/multi-exposure terrains.
    - If is_flat_multi is True -> near-neutral (0.95..1.05 depending on species).
    - If slope < 5° -> 20% of nominal effect; <10° -> 50%; otherwise 100%.
    """
    species_key = species if species in ASPECT_LAG_FACTORS else "edulis"
    table = ASPECT_LAG_FACTORS[species_key]
    base = table.get(aspect_oct or "", 1.0)

    # Multi/flat terrains → negligible aspect effect
    if is_flat_multi:
        return 1.0 if 0.95 <= base <= 1.05 else (1.0 + (base - 1.0) * 0.15)

    # Slope-based damping (use degrees; thresholds conservative)
    damp = 1.0
    try:
        s = float(slope_deg)
    except Exception:
        s = 0.0
    if s < 5.0:
        damp = 0.2
    elif s < 10.0:
        damp = 0.5

    return 1.0 + (base - 1.0) * damp

SPECIES_PROFILES_V26 = {
    "aereus": {
        "hosts": ["quercia", "castagno", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [7, 8]},
        "tm7_opt": (18.0, 24.0), "tm7_critical": (12.0, 28.0),
        "lag_base": 8.5, "lag_range": (6, 11),
        "vpd_sens": 1.15, "drought_tolerance": 0.8,
        "soil_ph_opt": (5.5, 7.0), "smi_bias": 0.0,
        "elevation_opt": (200, 1000), "min_precip_flush": 12.0,
        "humidity_requirement": 85.0
    },
    "reticulatus": {
        "hosts": ["quercia", "castagno", "faggio", "misto"],
        "season": {"start_m": 5, "end_m": 9, "peak_m": [6, 7]},
        "tm7_opt": (16.0, 22.0), "tm7_critical": (10.0, 26.0),
        "lag_base": 7.8, "lag_range": (5, 10),
        "vpd_sens": 1.0, "drought_tolerance": 0.9,
        "soil_ph_opt": (5.0, 7.5), "smi_bias": 0.0,
        "elevation_opt": (100, 1200), "min_precip_flush": 10.0,
        "humidity_requirement": 80.0
    },
    "edulis": {
        "hosts": ["faggio", "conifere", "misto"],
        "season": {"start_m": 8, "end_m": 11, "peak_m": [9, 10]},
        "tm7_opt": (12.0, 18.0), "tm7_critical": (6.0, 22.0),
        "lag_base": 10.2, "lag_range": (8, 14),
        "vpd_sens": 1.2, "drought_tolerance": 0.6,
        "soil_ph_opt": (4.5, 6.5), "smi_bias": +0.05,
        "elevation_opt": (600, 2000), "min_precip_flush": 8.0,
        "humidity_requirement": 90.0
    },
    "pinophilus": {
        "hosts": ["conifere", "misto"],
        "season": {"start_m": 6, "end_m": 10, "peak_m": [8, 9]},
        "tm7_opt": (14.0, 20.0), "tm7_critical": (8.0, 24.0),
        "lag_base": 9.3, "lag_range": (7, 12),
        "vpd_sens": 0.9, "drought_tolerance": 1.1,
        "soil_ph_opt": (4.0, 6.0), "smi_bias": -0.02,
        "elevation_opt": (400, 1800), "min_precip_flush": 9.0,
        "humidity_requirement": 85.0
    }
}

def infer_porcino_species_super_advanced(habitat_used: str, month: int, elev_m: float, 
                                        aspect_oct: Optional[str], lat: float) -> str:
    h = (habitat_used or "misto").lower()
    candidates = []
    
    for species, profile in SPECIES_PROFILES_V26.items():
        if h not in profile["hosts"]: continue
        
        score = 1.0
        
        if month in profile["season"]["peak_m"]:
            score *= 1.5
        elif profile["season"]["start_m"] <= month <= profile["season"]["end_m"]:
            score *= 1.0
        else:
            score *= 0.3
        
        elev_min, elev_max = profile["elevation_opt"]
        if elev_min <= elev_m <= elev_max:
            score *= 1.2
        elif elev_m < elev_min:
            score *= max(0.4, 1.0 - (elev_min - elev_m) / 500.0)
        else:
            score *= max(0.4, 1.0 - (elev_m - elev_max) / 800.0)
        
        if aspect_oct:
            if species in ["aereus", "reticulatus"] and aspect_oct in ["S", "SE", "SW"]:
                score *= 1.1
            elif species in ["edulis", "pinophilus"] and aspect_oct in ["N", "NE", "NW"]:
                score *= 1.1
        
        if species == "aereus" and lat < 42.0: score *= 1.2
        elif species == "edulis" and lat > 45.0: score *= 1.15
        elif species == "pinophilus" and 44.0 <= lat <= 46.0: score *= 1.1
        
        candidates.append((species, score))
    
    if not candidates:
        return "reticulatus"
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# ===== LAG BIOLOGICO DINAMICO SPECIE-SPECIFICO =====
def stochastic_lag_super_advanced_v26(smi: float, thermal_shock: float, tmean7: float, 
                                     species: str, vpd_stress: float = 0.0, 
                                     photoperiod_factor: float = 1.0,
                                     cumulative_moisture: float = 0.0) -> int:
    """
    LAG BIOLOGICO DINAMICO BASATO SU LETTERATURA SCIENTIFICA
    Fonte: Vititanen (1997), Boddy et al (2014)
    """
    profile = SPECIES_PROFILES_V26.get(species, SPECIES_PROFILES_V26["reticulatus"])
    base_lag = profile["lag_base"]
    
    # Effetto umidità suolo - quanto più alto SMI, tanto più veloce
    smi_effect = -3.8 * (smi ** 1.3)
    
    # Shock termico accelera fruttificazione
    shock_effect = -1.8 * thermal_shock
    
    # Temperatura ottimale specie-specifica
    tm_opt_min, tm_opt_max = profile["tm7_opt"]
    tm_crit_min, tm_crit_max = profile["tm7_critical"]
    
    if tm_opt_min <= tmean7 <= tm_opt_max:
        temp_effect = -1.2  # Temperatura perfetta
    elif tm_crit_min <= tmean7 < tm_opt_min:
        temp_effect = 1.8 * (tm_opt_min - tmean7) / (tm_opt_min - tm_crit_min)
    elif tm_opt_max < tmean7 <= tm_crit_max:
        temp_effect = 1.4 * (tmean7 - tm_opt_max) / (tm_crit_max - tm_opt_max)
    else:
        temp_effect = 2.8  # Fuori range critico
    
    # VPD stress ritarda fruttificazione
    vpd_effect = 1.3 * vpd_stress * profile["vpd_sens"]
    
    # Fotoperiodo
    photoperiod_effect = 0.4 * (1.0 - photoperiod_factor)
    
    # NUOVO: Effetto umidità cumulativa - più umidità pregressa accelera
    moisture_effect = -0.8 * min(1.0, cumulative_moisture / 20.0)
    
    final_lag = (base_lag + smi_effect + shock_effect + temp_effect + 
                 vpd_effect + photoperiod_effect + moisture_effect)
    
    lag_min, lag_max = profile["lag_range"]
    return int(round(clamp(final_lag, lag_min, lag_max)))

def gaussian_kernel_advanced(x: float, mu: float, sigma: float, skewness: float = 0.0) -> float:
    base_gauss = math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    if skewness != 0.0:
        skew_factor = 1.0 + skewness * ((x - mu) / sigma)
        return base_gauss * max(0.1, skew_factor)
    
    return base_gauss

def event_strength_advanced(mm: float, duration_hours: float = 24.0, 
                          antecedent_smi: float = 0.5) -> float:
    base_strength = 1.0 - math.exp(-mm / 15.0)
    duration_factor = min(1.2, 1.0 + (duration_hours - 12.0) / 48.0)
    smi_factor = 0.7 + 0.6 * antecedent_smi
    return clamp(base_strength * duration_factor * smi_factor, 0.0, 1.5)

# ===== METEO OPENMETEO COMPLETO: ARCHIVE API + FORECAST API =====

async def fetch_openmeteo_archive(lat: float, lon: float, days_back: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch dati storici da OpenMeteo Archive API (fino a 92 giorni indietro)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    end_date = datetime.now() - timedelta(days=1)  # Ieri
    start_date = end_date - timedelta(days=days_back-1)  # 20 giorni indietro
    
    daily_vars = [
        "precipitation_sum", "temperature_2m_mean", "temperature_2m_min", 
        "temperature_2m_max", "relative_humidity_2m_mean", "et0_fao_evapotranspiration"
    ]
    
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(daily_vars)
    }
    
    async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    
    # Converti in formato uniforme
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
            "source": "openmeteo_archive"
        })
    
    return results

async def fetch_openmeteo_forecast(lat: float, lon: float, future_days: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch forecast da OpenMeteo Forecast API (10 giorni futuri)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "precipitation_sum", "temperature_2m_mean", "temperature_2m_min", 
        "temperature_2m_max", "relative_humidity_2m_mean", "et0_fao_evapotranspiration"
    ]
    
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": ",".join(daily_vars),
        "past_days": 0, "forecast_days": future_days,
    }
    
    async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    
    # Converti in formato uniforme
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
            "source": "openmeteo_forecast"
        })
    
    return results

async def fetch_visual_crossing_backup_full(lat: float, lon: float, days_back: int = 20, future_days: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch completo da Visual Crossing solo come backup emergenziale
    """
    if not VISUAL_CROSSING_KEY:
        logger.warning("Visual Crossing API key not configured for backup")
        return {"historical": [], "forecast": []}
    
    try:
        # Dati storici
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back-1)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        url_hist = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_str}/{end_str}"
        params_hist = {
            "key": VISUAL_CROSSING_KEY,
            "include": "days",
            "elements": "datetime,temp,tempmin,tempmax,precip,humidity",
            "unitGroup": "metric"
        }
        
        # Forecast (limitato, Visual Crossing ha forecast ridotto)
        future_end = datetime.now() + timedelta(days=min(future_days, 7))  # Max 7 giorni forecast
        url_forecast = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{datetime.now().strftime('%Y-%m-%d')}/{future_end.strftime('%Y-%m-%d')}"
        params_forecast = params_hist.copy()
        
        async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
            # Fetch parallelo
            tasks = [
                c.get(url_hist, params=params_hist),
                c.get(url_forecast, params=params_forecast)
            ]
            hist_resp, forecast_resp = await asyncio.gather(*tasks, return_exceptions=True)
        
        historical = []
        forecast = []
        
        # Processa storico
        if not isinstance(hist_resp, Exception):
            hist_resp.raise_for_status()
            hist_data = hist_resp.json()
            for day in hist_data.get("days", []):
                historical.append({
                    "date": day["datetime"],
                    "precipitation_mm": float(day.get("precip", 0.0)),
                    "temp_min": float(day.get("tempmin", 0.0)),
                    "temp_max": float(day.get("tempmax", 0.0)),
                    "temp_mean": float(day.get("temp", 0.0)),
                    "humidity": float(day.get("humidity", 65.0)),
                    "et0": 2.5,  # Stima per Visual Crossing
                    "source": "visual_crossing_backup"
                })
        
        # Processa forecast
        if not isinstance(forecast_resp, Exception):
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()
            forecast_days = forecast_data.get("days", [])[1:]  # Escludi oggi
            for day in forecast_days:
                forecast.append({
                    "date": day["datetime"],
                    "precipitation_mm": float(day.get("precip", 0.0)),
                    "temp_min": float(day.get("tempmin", 0.0)),
                    "temp_max": float(day.get("tempmax", 0.0)),
                    "temp_mean": float(day.get("temp", 0.0)),
                    "humidity": float(day.get("humidity", 65.0)),
                    "et0": 2.5,  # Stima per Visual Crossing
                    "source": "visual_crossing_backup"
                })
        
        logger.info(f"Visual Crossing backup: {len(historical)} historical + {len(forecast)} forecast days")
        return {"historical": historical, "forecast": forecast}
        
    except Exception as e:
        logger.error(f"Visual Crossing backup completely failed: {e}")
        return {"historical": [], "forecast": []}

async def fetch_weather_data_openmeteo_complete(lat: float, lon: float, past_days: int = 20, future_days: int = 10) -> Dict[str, Any]:
    """
    SISTEMA OPENMETEO COMPLETO: Archive API + Forecast API
    Visual Crossing solo backup emergenziale se entrambi falliscono
    """
    try:
        logger.info(f"Attempting OpenMeteo complete system: Archive({past_days}d) + Forecast({future_days}d)")
        
        # TENTATIVO PRIMARIO: OpenMeteo Archive + Forecast
        try:
            tasks = [
                fetch_openmeteo_archive(lat, lon, days_back=past_days),
                fetch_openmeteo_forecast(lat, lon, future_days=future_days)
            ]
            
            archive_data, forecast_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verifica successo di entrambe le chiamate
            if (not isinstance(archive_data, Exception) and len(archive_data) >= past_days - 2 and
                not isinstance(forecast_data, Exception) and len(forecast_data) >= future_days - 1):
                
                logger.info(f"✅ OpenMeteo complete success: Archive={len(archive_data)}d, Forecast={len(forecast_data)}d")
                
                # Combina i dati
                combined_data = archive_data + forecast_data
                
                # Costruisci serie temporali
                time_series = [item["date"] for item in combined_data]
                P_series = [item["precipitation_mm"] for item in combined_data]
                Tmin_series = [item["temp_min"] for item in combined_data]
                Tmax_series = [item["temp_max"] for item in combined_data]
                Tmean_series = [item["temp_mean"] for item in combined_data]
                RH_series = [item["humidity"] for item in combined_data]
                ET0_series = [item["et0"] for item in combined_data]
                
                return {
                    "daily": {
                        "time": time_series,
                        "precipitation_sum": P_series,
                        "temperature_2m_min": Tmin_series,
                        "temperature_2m_max": Tmax_series,
                        "temperature_2m_mean": Tmean_series,
                        "relative_humidity_2m_mean": RH_series,
                        "et0_fao_evapotranspiration": ET0_series
                    },
                    "metadata": {
                        "sources": {
                            "primary_source": "openmeteo_complete",
                            "backup_used": False,
                            "archive_days": len(archive_data),
                            "forecast_days": len(forecast_data),
                            "total_days": len(combined_data),
                            "completeness": len(combined_data) / (past_days + future_days)
                        },
                        "quality_score": 0.98  # OpenMeteo completo = massima qualità
                    }
                }
                
        except Exception as e:
            logger.warning(f"OpenMeteo complete system failed: {e}")
            
        # FALLBACK: Visual Crossing completo come backup emergenziale
        logger.info("Falling back to Visual Crossing complete backup...")
        
        vc_data = await fetch_visual_crossing_backup_full(lat, lon, days_back=past_days, future_days=future_days)
        
        historical = vc_data["historical"]
        forecast = vc_data["forecast"]
        
        if not historical and not forecast:
            raise HTTPException(500, "Errore critico: tutte le fonti meteorologiche non disponibili")
            
        # Combina i dati Visual Crossing
        combined_data = historical + forecast
        
        if not combined_data:
            raise HTTPException(500, "Nessun dato meteorologico disponibile da nessuna fonte")
            
        # Costruisci serie temporali
        time_series = [item["date"] for item in combined_data]
        P_series = [item["precipitation_mm"] for item in combined_data]
        Tmin_series = [item["temp_min"] for item in combined_data]
        Tmax_series = [item["temp_max"] for item in combined_data]
        Tmean_series = [item["temp_mean"] for item in combined_data]
        RH_series = [item["humidity"] for item in combined_data]
        ET0_series = [item["et0"] for item in combined_data]
        
        completeness = len(combined_data) / (past_days + future_days)
        
        logger.info(f"Visual Crossing backup successful: {len(historical)} historical + {len(forecast)} forecast days")
        
        return {
            "daily": {
                "time": time_series,
                "precipitation_sum": P_series,
                "temperature_2m_min": Tmin_series,
                "temperature_2m_max": Tmax_series,
                "temperature_2m_mean": Tmean_series,
                "relative_humidity_2m_mean": RH_series,
                "et0_fao_evapotranspiration": ET0_series
            },
            "metadata": {
                "sources": {
                    "primary_source": "openmeteo_failed",
                    "backup_used": True,
                    "visual_crossing_historical_days": len(historical),
                    "visual_crossing_forecast_days": len(forecast),
                    "total_days": len(combined_data),
                    "completeness": completeness
                },
                "quality_score": min(0.80, completeness * 0.75 + 0.15)  # Qualità ridotta per backup
            }
        }
        
    except Exception as e:
        logger.error(f"Complete weather system catastrophic failure: {e}")
        raise HTTPException(500, f"Errore sistema meteorologico completo: {e}")

# ===== ELEVAZIONE E MICROTOPOGRAFIA SUPER AVANZATA =====
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
        
        octants = ["N", "NE", "E",from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
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

# Setup logging professionale - FIX: Gestione logs directory
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
    title="BoletusLab® v2.5.7 - Sistema Previsionale Micologico",
    version="2.5.7",
    description="Sistema meteorologico OpenMeteo Archive+Forecast • Visual Crossing backup • Algoritmi Scientifici Boletus spp."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

HEADERS = {"User-Agent":"BoletusLab/2.5.7 (+scientific)", "Accept-Language":"it"}
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

# CHIAVI API
VISUAL_CROSSING_KEY = os.environ.get("VISUAL_CROSSING_KEY", "")

# Database avanzato - FIX: Gestione data directory
data_dir = os.getenv('DATA_DIR', 'data')
os.makedirs(data_dir, exist_ok=True)
DB_PATH = os.path.join(data_dir, "porcini_validations.db")

def init_database():
    """Inizializza database SQLite avanzato per machine learning"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Tabella segnalazioni con metadati completi
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                species TEXT NOT NULL,
                quantity INTEGER DEFAULT 1,
                size_cm_avg REAL,
                size_cm_max REAL,
                confidence REAL DEFAULT 0.8,
                photo_url TEXT,
                notes TEXT,
                habitat_observed TEXT,
                weather_conditions TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '2.5.7',
                user_experience_level INTEGER DEFAULT 3,
                validation_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT,
                elevation_m REAL,
                slope_deg REAL,
                aspect_deg REAL
            )
        ''')
        
        # Tabella ricerche negative
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS no_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                searched_hours REAL DEFAULT 2.0,
                search_method TEXT DEFAULT 'visual',
                habitat_searched TEXT,
                weather_conditions TEXT,
                notes TEXT,
                predicted_score INTEGER,
                model_version TEXT DEFAULT '2.5.7',
                search_thoroughness INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT,
                elevation_m REAL
            )
        ''')
        
        # Tabella predizioni per ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                date TEXT NOT NULL,
                predicted_score INTEGER NOT NULL,
                species TEXT NOT NULL,
                habitat TEXT,
                confidence_data TEXT,
                weather_data TEXT,
                model_features TEXT,
                model_version TEXT DEFAULT '2.5.7',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                geohash TEXT,
                validated BOOLEAN DEFAULT FALSE,
                validation_date TEXT,
                validation_result TEXT
            )
        ''')
        
        # Tabella performance modello
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model_version TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                rmse REAL,
                total_predictions INTEGER,
                total_validations INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database avanzato inizializzato con successo")
    except Exception as e:
        logger.error(f"Errore inizializzazione database: {e}")

init_database()

# ===== UTILITIES MATEMATICHE AVANZATE =====
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
    """Esposizione ridotta; microclima smorzato su pendii dolci e terreni piatti/multi-esposizione"""
    # Terreni pianeggianti/multi-esposizione → effetto quasi neutro
    if (not aspect_oct) or aspect_oct in {'FLAT','MULTI','MULTI_FLAT'}:
        return 0.5
    # Damping su pendii dolci
    if slope_deg < 5.0:
        slope_damp = 0.2
    elif slope_deg < 10.0:
        slope_damp = 0.5
    else:
        slope_damp = 1.0
    
    # ESPOSIZIONE RIDOTTA - I funghi preferiscono zone meno esposte
    aspect_energy = { 'N':0.6, 'NE':0.65, 'E':0.7, 'SE':0.75, 'S':0.7, 'SW':0.72, 'W':0.7, 'NW':0.62 }
    base_energy = aspect_energy.get(aspect_oct, 0.5)
    base_energy = 1.0 - (1.0 - base_energy) * slope_damp  # smorza l'effetto con poca pendenza
    
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

# ===== UMIDITÀ CUMULATIVA SCIENTIFICA =====
def cumulative_moisture_index(P_series: List[float], days_window: int = 14) -> List[float]:
    """
    Calcola l'umidità cumulativa considerando eventi di pioggia precedenti
    Basato su: "cumulative time of high humidity needed for development" (Viitanen 1997)
    """
    cmi_values = []
    
    for i in range(len(P_series)):
        # Finestra mobile degli ultimi 'days_window' giorni
        start_idx = max(0, i - days_window + 1)
        window_precip = P_series[start_idx:i+1]
        
        # Peso decrescente per eventi più vecchi (half-life 7 giorni)
        weights = []
        for j, p in enumerate(window_precip):
            days_ago = len(window_precip) - 1 - j
            weight = math.exp(-days_ago / 7.0)  # Half-life 7 giorni
            weights.append(weight)
        
        # Indice cumulativo pesato
        if weights:
            weighted_sum = sum(w * p for w, p in zip(weights, window_precip))
            weight_sum = sum(weights)
            cmi = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            cmi = 0.0
        
        cmi_values.append(cmi)
    
    return cmi_values

# ===== SOGLIE DINAMICHE CON UMIDITÀ CUMULATIVA =====
def dynamic_rain_threshold_v26(smi: float, month: int, elevation: float, 
                              lat: float, recent_temp_trend: float, 
                              cumulative_moisture: float) -> float:
    """MODIFICA: Include umidità cumulativa secondo letteratura"""
    base_threshold = 7.5
    
    if smi > 0.8: smi_factor = 0.6
    elif smi > 0.6: smi_factor = 0.8
    elif smi < 0.3: smi_factor = 1.4
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
    
    # NUOVA COMPONENTE: Fattore umidità cumulativa
    # Più alta l'umidità pregressa, minore la soglia necessaria
    if cumulative_moisture > 15.0: moisture_factor = 0.7
    elif cumulative_moisture > 10.0: moisture_factor = 0.85
    elif cumulative_moisture > 5.0: moisture_factor = 0.95
    else: moisture_factor = 1.1
    
    final_threshold = (base_threshold * smi_factor * et_factor * 
                      alt_factor * lat_factor * temp_factor * moisture_factor)
    return clamp(final_threshold, 3.0, 20.0)

# ===== SMOOTHING SAVITZKY-GOLAY AVANZATO =====
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
            
            # Preserva picchi importanti
            for i, (orig, smooth) in enumerate(zip(forecast, smoothed)):
                if orig > 75 and smooth < orig * 0.8:
                    smoothed[i] = orig * 0.9
            
            return np.clip(smoothed, 0, 100).tolist()
            
        except Exception as e:
            logger.warning(f"Savgol failed: {e}, using custom smoothing")
    
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

# ===== SMI AVANZATO CON ERA5-LAND =====
SM_CACHE: Dict[str, Dict[str, Any]] = {}

async def _prefetch_era5l_sm_advanced(lat: float, lon: float, days: int = 40) -> None:
    if not CDS_API_KEY or not CDS_AVAILABLE: return
    
    key = f"{round(lat,3)},{round(lon,3)}"
    if key in SM_CACHE and (time.time() - SM_CACHE[key].get("ts", 0)) < 12*3600:
        return
    
    def _blocking_download():
        try:
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
            return {"daily": daily, "ts": time.time()}
            
        except Exception as e:
            logger.warning(f"ERA5-Land fetch failed: {e}")
            return None
    
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _blocking_download)
        if data and "daily" in data:
            SM_CACHE[key] = data
    except Exception as e:
        logger.warning(f"ERA5-Land processing failed: {e}")

def smi_from_p_et0_advanced(P: List[float], ET0: List[float]) -> List[float]:
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

# ===== CONFIDENCE SYSTEM 5D SUPER AVANZATO =====
def confidence_5d_super_advanced(
    weather_agreement: float,
    habitat_confidence: float,
    smi_reliability: float,
    vpd_validity: bool,
    has_recent_validation: bool,
    elevation_reliability: float = 0.8,
    temporal_consistency: float = 0.7
) -> Dict[str, float]:
    
    met_conf = clamp(weather_agreement, 0.15, 0.98)
    
    eco_base = clamp(habitat_confidence, 0.1, 0.9)
    if has_recent_validation: eco_base *= 1.15
    eco_conf = clamp(eco_base, 0.1, 0.95)
    
    hydro_base = clamp(smi_reliability, 0.2, 0.9)
    hydro_conf = hydro_base * clamp(temporal_consistency, 0.5, 1.0)
    
    atmo_base = 0.85 if vpd_validity else 0.35
    atmo_conf = atmo_base * clamp(elevation_reliability, 0.6, 1.0)
    
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

# ===== PROFILI SPECIE SCIENTIFICAMENTE VERIFICATI =====

# ===== LAG FACTORS BY ASPECT (SPECIES-SPECIFIC) =====
# Factors derived conservatively from peer-reviewed literature cited in app footer.
# Values < 1 shorten lag; > 1 lengthen lag. These are intentionally conservative.
ASPECT_LAG_FACTORS = {
    "edulis": {"N": 0.80, "NE": 0.88, "E": 0.96, "SE": 1.08, "S": 1.18, "SW": 1.15, "W": 1.02, "NW": 0.85},
    "pinophilus": {"N": 0.85, "NE": 0.90, "E": 0.98, "SE": 1.05, "S": 1.15, "SW": 1.12, "W": 1.03, "NW": 0.88},
    "aereus": {"SE": 0.88, "S": 0.90, "E": 0.94, "SW": 0.92, "W": 1.06, "NW": 1.15, "N": 1.22, "NE": 1.12},
    # For species with limited literature, use conservative estimates derived from edulis
    "reticulatus": {"N": 0.88, "NE": 0.92, "E": 0.98, "SE": 1.04, "S": 1.12, "SW": 1.08, "W": 1.01, "NW": 0.94},
    "aestivalis": {"N": 0.82, "NE": 0.90, "E": 0.97, "SE": 1.06, "S": 1.16, "SW": 1.12, "W": 1.03, "NW": 0.87},

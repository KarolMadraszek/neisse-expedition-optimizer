import math
import numpy as np
import pandas as pd
import requests

GRAPH_CSV = "nysa_graph_1km.csv"
IMGW_HYDRO_URL = "https://danepubliczne.imgw.pl/api/data/hydro/"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BASE_CRAFT_SPEED_MS = 1.8
MANNING_N = 0.040
WIND_EFFECT_FACTOR = 0.08
CURRENT_CALIBRATION = 1.0      # (do dostrojenia po teście GPS na Nysie)

IMGW_STATION_COORDS = {
    "SIENIAWKA": (50.8936, 14.8358),
    "ZGORZELEC": (51.1508, 14.9856),
    "PRZEWÓZ": (51.4764, 14.9489),
    "GUBIN": (51.9508, 14.7219),
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fetch_imgw_neisse() -> pd.DataFrame:
    print("1/3 Pobieranie aktualnych stanów wód i przepływów z API IMGW...")
    resp = requests.get(IMGW_HYDRO_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    stations = []
    for row in data:
        river = str(row.get("rzeka", "")).lower()
        if "nysa łużycka" in river:
            name = str(row.get("stacja", "")).upper().strip()
            water_level = pd.to_numeric(row.get("stan_wody"), errors="coerce")
            flow_m3s = pd.to_numeric(row.get("przelyw"), errors="coerce")
            water_temp = pd.to_numeric(row.get("temperatura_wody"), errors="coerce")

            coords = next((v for k, v in IMGW_STATION_COORDS.items() if k in name), None)
            if coords:
                stations.append({
                    "station_name": name,
                    "lat": coords[0],
                    "lon": coords[1],
                    "water_level_cm": water_level,
                    "flow_m3s": flow_m3s,
                    "water_temp_c": water_temp,
                    "measurement_time": row.get("stan_wody_data_pomiaru", ""),
                })

    df_stations = pd.DataFrame(stations)
    if df_stations.empty:
        raise RuntimeError("Nie znaleziono aktywnych stacji dla Nysy Łużyckiej w API IMGW.")

    print("    -> Pobrane stacje IMGW:")
    for _, s in df_stations.iterrows():
        print(f"       * {s['station_name']}: stan={s['water_level_cm']} cm, przepływ={s['flow_m3s']} m3/s ({s['measurement_time']})")

    return df_stations


def fetch_wind_forecast(df_graph: pd.DataFrame) -> pd.DataFrame:
    print("2/3 Pobieranie prognozy wiatru (Open-Meteo) wzdłuż biegu rzeki...")

    sample_df = df_graph.iloc[::10].copy()
    lats = ",".join(f"{lat:.4f}" for lat in sample_df["lat"])
    lons = ",".join(f"{lon:.4f}" for lon in sample_df["lon"])

    resp = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lats,
            "longitude": lons,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "forecast_days": 2,
            "timezone": "Europe/Berlin",
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()

    weather_rows = []
    for idx, item in enumerate(results):
        hourly = item["hourly"]
        weather_rows.append({
            "sample_km": int(sample_df.iloc[idx]["km_start"]),
            "forecast_time": hourly["time"][0],
            "air_temp_c": hourly["temperature_2m"][0],
            "wind_speed_ms": hourly["wind_speed_10m"][0],
            "wind_from_deg": hourly["wind_direction_10m"][0],
        })

    print(f"    -> Pobrano warunki wiatrowe dla {len(weather_rows)} punktów kontrolnych (czas: {weather_rows[0]['forecast_time']}).")
    return pd.DataFrame(weather_rows)


def compute_tailwind_ms(river_bearing_deg: float, wind_speed_ms: float, wind_from_deg: float) -> float:
    wind_to_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    river_rad = math.radians(river_bearing_deg)
    return round(wind_speed_ms * math.cos(wind_to_rad - river_rad), 2)


def classify_obstacle_and_portage(weir_names: str, water_level_cm: float) -> tuple[str, float]:
    if not isinstance(weir_names, str) or not weir_names.strip():
        return "brak", 0.0

    names_lower = weir_names.lower()
    if "wehr" in names_lower or "jaz" in names_lower or "mühle" in names_lower or "bremenwerk" in names_lower:
        return "jaz_przenoska", 14.0
    if "sohl" in names_lower or "próg" in names_lower:
        if pd.notna(water_level_cm) and water_level_cm >= 140:
            return "prog_splywalny", 0.0
        return "prog_niska_woda", 5.0
    if "brückenreste" in names_lower or "mostu" in names_lower:
        return "filary_mostu", 0.0

    return "inna_przeszkoda", 8.0


def estimate_current_speed_ms(slope_pct: float, water_level_cm: float) -> float:
    slope_dec = max(slope_pct / 100.0, 0.0001)

    if pd.notna(water_level_cm):
        depth_m = float(np.clip((water_level_cm - 30.0) / 120.0, 0.4, 2.5))
    else:
        depth_m = 0.9

    v_manning = CURRENT_CALIBRATION * (1.0 / MANNING_N) * (depth_m ** (2.0 / 3.0)) * (slope_dec ** 0.5)
    return round(float(np.clip(v_manning, 0.25, 2.2)), 2)


def compute_segment_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    df["current_speed_ms"] = [
        estimate_current_speed_ms(s, wl)
        for s, wl in zip(df["slope_pct_clean"], df["water_level_cm"])
    ]

    df["sog_ms"] = (
        BASE_CRAFT_SPEED_MS
        + df["current_speed_ms"]
        + WIND_EFFECT_FACTOR * df["tailwind_ms"]
    ).clip(lower=0.3).round(2)
    df["sog_kmh"] = (df["sog_ms"] * 3.6).round(2)

    df["paddle_time_min"] = (1000.0 / (df["sog_ms"] * 60.0)).round(2)
    df["total_segment_min"] = (df["paddle_time_min"] + df["portage_min"]).round(2)
    df["cumulative_time_h"] = (df["total_segment_min"].cumsum() / 60.0).round(2)

    return df

def enrich_river_graph():
    df_graph = pd.read_csv(GRAPH_CSV)
    df_imgw = fetch_imgw_neisse()
    df_weather = fetch_wind_forecast(df_graph)

    print("3/3 Łączenie danych hydrologicznych, wiatrowych i klasyfikacji przeszkód...")

    nearest_stations = []
    for _, row in df_graph.iterrows():
        distances = [
            haversine_km(row["lat"], row["lon"], s["lat"], s["lon"])
            for _, s in df_imgw.iterrows()
        ]
        best_idx = int(np.argmin(distances))
        nearest_stations.append(df_imgw.iloc[best_idx])

    df_nearest_imgw = pd.DataFrame(nearest_stations).reset_index(drop=True)
    df_graph["imgw_station"] = df_nearest_imgw["station_name"]
    df_graph["water_level_cm"] = df_nearest_imgw["water_level_cm"]
    df_graph["flow_m3s"] = df_nearest_imgw["flow_m3s"]
    df_graph["water_temp_c"] = df_nearest_imgw["water_temp_c"]

    df_graph["sample_km"] = (df_graph["km_start"] // 10) * 10
    df_graph = df_graph.merge(df_weather, on="sample_km", how="left").drop(columns=["sample_km"])

    df_graph["tailwind_ms"] = [
        compute_tailwind_ms(b, ws, wd)
        for b, ws, wd in zip(df_graph["bearing_deg"], df_graph["wind_speed_ms"], df_graph["wind_from_deg"])
    ]

    classified = [
        classify_obstacle_and_portage(wn, wl)
        for wn, wl in zip(df_graph["weir_names"], df_graph["water_level_cm"])
    ]
    df_graph["obstacle_type"] = [c[0] for c in classified]
    df_graph["portage_min"] = [c[1] for c in classified]

    df_graph["slope_pct_clean"] = df_graph["slope_pct"].replace(0.0, np.nan).interpolate(method="linear").bfill().ffill().round(4)
    df_graph = compute_segment_kinematics(df_graph)

    output_file = "nysa_telemetry_now.csv"
    df_graph.to_csv(output_file, index=False)

    total_paddle_h = df_graph["paddle_time_min"].sum() / 60.0
    total_portage_h = df_graph["portage_min"].sum() / 60.0
    total_h = df_graph["total_segment_min"].sum() / 60.0
    avg_sog_kmh = df_graph["sog_kmh"].mean()
    avg_current_kmh = (df_graph["current_speed_ms"].mean()) * 3.6

    print(f"\nZapisano pełną symulację warunków do: {output_file}")
    print("\n=== PODSUMOWANIE SPŁYWU (PRZY DZISIEJSZYCH WARUNKACH, BEZ ZMĘCZENIA) ===")
    print(f" - Dystans całkowity:    {len(df_graph)} km")
    print(f" - Średnia prędkość nurtu: {avg_current_kmh:.2f} km/h")
    print(f" - Średnia prędkość (SOG): {avg_sog_kmh:.2f} km/h")
    print(f" - Czysty czas na wodzie:  {total_paddle_h:.2f} h")
    print(f" - Czas na przenoski:      {total_portage_h:.2f} h ({df_graph['portage_min'].sum():.0f} min)")
    print(f" - ŁĄCZNY CZAS TRASY:      {total_h:.2f} h")

    fastest = df_graph.loc[df_graph["sog_kmh"].idxmax()]
    slowest = df_graph.loc[df_graph["sog_kmh"].idxmin()]
    print(
        f"\n - Najszybszy odcinek: km {int(fastest['km_start'])} ({fastest['sog_kmh']} km/h, nurt {fastest['current_speed_ms']} m/s, wiatr {fastest['tailwind_ms']} m/s)")
    print(
        f" - Najwolniejszy odcinek: km {int(slowest['km_start'])} ({slowest['sog_kmh']} km/h, nurt {slowest['current_speed_ms']} m/s, wiatr {slowest['tailwind_ms']} m/s)")


if __name__ == "__main__":
    enrich_river_graph()
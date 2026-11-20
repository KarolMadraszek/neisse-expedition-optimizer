import json
import math
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

MIN_LAT = 50.80
SEGMENT_LENGTH_M = 1000
WEIR_BUFFER_M = 50
METRIC_CRS = "EPSG:32633"

CACHE_RIVER_FILE = "cache_nysa_full.json"
CACHE_ELEV_FILE = "cache_nysa_elevation.json"

HEADERS = {
    "User-Agent": "NeisseExpeditionOptimizer/0.3 (lokalny-projekt-badawczy)",
    "Accept": "*/*",
}

LOCAL_WEIRS_CSV = "jazy_nysa.csv"
LOCAL_WEIRS_GEOJSON = "jazy_osm.geojson"
DEFAULT_PORTAGE_MIN = 15.0


def find_river_relation_id() -> int:
    print("1/5 Szukanie ID relacji Nysy Łużyckiej w Nominatim...")
    for name in ("Nysa Łużycka", "Lausitzer Neiße"):
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": name, "format": "json", "limit": 10},
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        for item in resp.json():
            if item.get("osm_type") == "relation" and item.get("class") == "waterway":
                rel_id = int(item["osm_id"])
                print(f"    -> Znaleziono relację: ID={rel_id}")
                return rel_id

    raise RuntimeError("Nie udało się znaleźć relacji waterway dla Nysy Łużyckiej.")


def fetch_relation_full(rel_id: int) -> dict:
    if os.path.exists(CACHE_RIVER_FILE):
        print(f"2/5 Wczytywanie geometrii rzeki z cache ({CACHE_RIVER_FILE})...")
        with open(CACHE_RIVER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://api.openstreetmap.org/api/0.6/relation/{rel_id}/full.json"
    print("2/5 Pobieranie pełnej relacji z OSM API...")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    with open(CACHE_RIVER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def parse_river_geometry(data: dict, rel_id: int) -> tuple[LineString, list[dict]]:
    elements = data.get("elements", [])
    nodes_coords = {}
    node_weirs = []
    ways_map = {}
    target_rel = None

    for el in elements:
        el_type = el["type"]
        if el_type == "node":
            nodes_coords[el["id"]] = (el["lon"], el["lat"])
            tags = el.get("tags", {})
            w_tag = tags.get("waterway", "")
            if w_tag in ("weir", "dam", "waterfall", "rapids") and el["lat"] >= MIN_LAT:
                node_weirs.append({
                    "osm_id": f"node/{el['id']}",
                    "name": tags.get("name", f"Jaz/próg ({w_tag})"),
                    "type": w_tag,
                    "geometry": Point(el["lon"], el["lat"]),
                })
        elif el_type == "way":
            ways_map[el["id"]] = el
        elif el_type == "relation" and el["id"] == rel_id:
            target_rel = el

    if not target_rel:
        raise RuntimeError("Brak docelowej relacji w pobranym pliku JSON.")

    way_ids = [
        m["ref"]
        for m in target_rel.get("members", [])
        if m["type"] == "way" and m.get("role", "") in ("", "main_stream")
    ]
    if not way_ids:
        way_ids = [m["ref"] for m in target_rel.get("members", []) if m["type"] == "way"]

    lines = []
    for wid in way_ids:
        way = ways_map.get(wid)
        if not way:
            continue
        coords = [nodes_coords[nid] for nid in way.get("nodes", []) if nid in nodes_coords]
        if len(coords) >= 2 and any(pt[1] >= MIN_LAT for pt in coords):
            lines.append(LineString(coords))

    merged = linemerge(lines)

    if merged.geom_type == "MultiLineString":
        sorted_geoms = sorted(merged.geoms, key=lambda g: min(pt[1] for pt in g.coords))
        all_coords = []
        for g in sorted_geoms:
            coords = list(g.coords)
            if coords[0][1] > coords[-1][1]:
                coords.reverse()
            all_coords.extend(coords)
        river_line = LineString(all_coords)
    else:
        coords = list(merged.coords)
        if coords[0][1] > coords[-1][1]:
            coords.reverse()
        river_line = LineString(coords)

    return river_line, node_weirs


def load_local_weirs(initial_weirs: list[dict]) -> gpd.GeoDataFrame:
    print("3/5 Wczytywanie lokalnej bazy jazów i przeszkód...")
    records = []

    for w in initial_weirs:
        records.append({
            "name": w["name"],
            "type": w["type"],
            "bank": "",
            "carry_m": 0,
            "portage_min": DEFAULT_PORTAGE_MIN,
            "priority": 3,
            "geometry": w["geometry"],
        })

    if os.path.exists(LOCAL_WEIRS_GEOJSON):
        gdf_osm = gpd.read_file(LOCAL_WEIRS_GEOJSON).to_crs("EPSG:4326")
        print(f"    -> Wczytano {len(gdf_osm)} obiektów z {LOCAL_WEIRS_GEOJSON}")
        for _, row in gdf_osm.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Jeśli jaz jest linią lub poligonem, bierzemy jego punkt środkowy
            pt = geom if geom.geom_type == "Point" else geom.centroid
            if pt.y < MIN_LAT:
                continue
            name = row.get("name")
            w_type = row.get("waterway", "weir")
            records.append({
                "name": name if pd.notna(name) and str(name).strip() else f"Jaz OSM ({w_type})",
                "type": w_type if pd.notna(w_type) else "weir",
                "bank": "",
                "carry_m": 0,
                "portage_min": DEFAULT_PORTAGE_MIN,
                "priority": 2,
                "geometry": pt,
            })

    if os.path.exists(LOCAL_WEIRS_CSV):
        df_csv = pd.read_csv(LOCAL_WEIRS_CSV)
        print(f"    -> Wczytano {len(df_csv)} jazów z własnej bazy {LOCAL_WEIRS_CSV}")
        for _, row in df_csv.iterrows():
            if row["lat"] < MIN_LAT:
                continue
            records.append({
                "name": str(row.get("name", "Jaz (CSV)")),
                "type": "weir",
                "bank": str(row.get("bank", "")) if pd.notna(row.get("bank")) else "",
                "carry_m": float(row.get("carry_m", 0)) if pd.notna(row.get("carry_m")) else 0.0,
                "portage_min": float(row.get("portage_min", DEFAULT_PORTAGE_MIN))
                if pd.notna(row.get("portage_min"))
                else DEFAULT_PORTAGE_MIN,
                "priority": 1,
                "geometry": Point(float(row["lon"]), float(row["lat"])),
            })

    if not records:
        return gpd.GeoDataFrame(
            columns=["name", "type", "bank", "carry_m", "portage_min", "priority", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def fetch_elevations(lats: list[float], lons: list[float]) -> list[float]:
    print("4/5 Pobieranie wysokości n.p.m. (Open-Meteo Elevation API)...")

    if os.path.exists(CACHE_ELEV_FILE):
        with open(CACHE_ELEV_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(lats):
            print(f"    -> Wczytano {len(cached)} punktów wysokościowych z cache.")
            return cached

    elevations = []
    batch_size = 100
    for i in range(0, len(lats), batch_size):
        batch_lats = ",".join(f"{lat:.5f}" for lat in lats[i : i + batch_size])
        batch_lons = ",".join(f"{lon:.5f}" for lon in lons[i : i + batch_size])

        resp = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": batch_lats, "longitude": batch_lons},
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        elevations.extend(resp.json()["elevation"])
        time.sleep(0.3)

    with open(CACHE_ELEV_FILE, "w", encoding="utf-8") as f:
        json.dump(elevations, f)

    print(f"    -> Pobrano wysokość dla {len(elevations)} punktów.")
    return elevations


def calculate_bearing(p1: Point, p2: Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = math.degrees(math.atan2(dx, dy))
    return round((angle + 360) % 360, 1)


def build_river_graph():
    rel_id = find_river_relation_id()
    data = fetch_relation_full(rel_id)
    river_wgs84, initial_weirs = parse_river_geometry(data, rel_id)
    weirs_wgs84 = load_local_weirs(initial_weirs)

    river_gdf = gpd.GeoDataFrame(geometry=[river_wgs84], crs="EPSG:4326").to_crs(METRIC_CRS)
    river_metric: LineString = river_gdf.geometry.iloc[0]
    total_length_m = river_metric.length
    print(f"    -> Długość rzeki: {total_length_m / 1000:.2f} km")

    if not weirs_wgs84.empty:
        weirs_metric = weirs_wgs84.to_crs(METRIC_CRS)
        weirs_metric["dist_to_river_m"] = weirs_metric.geometry.distance(river_metric)
        river_weirs = weirs_metric[weirs_metric["dist_to_river_m"] <= WEIR_BUFFER_M].copy()
        river_weirs["river_km"] = river_weirs.geometry.apply(lambda pt: river_metric.project(pt) / 1000.0)
        river_weirs = river_weirs.sort_values("river_km")
        river_weirs["km_bucket"] = (river_weirs["river_km"] * 5).round()
        river_weirs = river_weirs.sort_values(["priority", "river_km"]).drop_duplicates(subset=["km_bucket"]).copy()
    else:
        river_weirs = pd.DataFrame(columns=["name", "type", "river_km"])

    print(f"    -> Przypisano {len(river_weirs)} unikalnych jazów/progów na trasie.")

    print(f"5/5 Cięcie rzeki co {SEGMENT_LENGTH_M} m i obliczanie spadku koryta...")
    nodes = []
    num_segments = int(total_length_m // SEGMENT_LENGTH_M)

    for i in range(num_segments):
        start_m = i * SEGMENT_LENGTH_M
        end_m = (i + 1) * SEGMENT_LENGTH_M

        p_start = river_metric.interpolate(start_m)
        p_end = river_metric.interpolate(end_m)
        bearing = calculate_bearing(p_start, p_end)

        if not river_weirs.empty:
            seg_weirs = river_weirs[
                (river_weirs["river_km"] >= i) & (river_weirs["river_km"] < i + 1)
            ]
        else:
            seg_weirs = pd.DataFrame()

        nodes.append({
            "segment_id": i,
            "km_start": i,
            "km_end": i + 1,
            "bearing_deg": bearing,
            "weir_count": len(seg_weirs),
            "has_weir": len(seg_weirs) > 0,
            "weir_names": "; ".join(seg_weirs["name"].tolist()) if not seg_weirs.empty else "",
            "geometry": p_start,
            "portage_min": round(float(seg_weirs["portage_min"].sum()), 1) if not seg_weirs.empty else 0.0,
        })

    graph_gdf = gpd.GeoDataFrame(nodes, crs=METRIC_CRS).to_crs("EPSG:4326")
    graph_gdf["lat"] = graph_gdf.geometry.y.round(6)
    graph_gdf["lon"] = graph_gdf.geometry.x.round(6)

    end_pt_wgs84 = gpd.GeoSeries([river_metric.interpolate(num_segments * SEGMENT_LENGTH_M)], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
    all_lats = graph_gdf["lat"].tolist() + [round(end_pt_wgs84.y, 6)]
    all_lons = graph_gdf["lon"].tolist() + [round(end_pt_wgs84.x, 6)]

    elevations = fetch_elevations(all_lats, all_lons)
    graph_gdf["elevation_m"] = elevations[:-1]

    elev_series = pd.Series(elevations)
    smooth_elev = elev_series.rolling(window=5, min_periods=1, center=True).mean()
    monotonic_elev = np.minimum.accumulate(smooth_elev)

    drops = -np.diff(monotonic_elev)
    graph_gdf["elevation_drop_m"] = np.round(drops, 2)
    graph_gdf["slope_pct"] = np.round((drops / SEGMENT_LENGTH_M) * 100, 4)

    csv_cols = [
        "segment_id",
        "km_start",
        "km_end",
        "lat",
        "lon",
        "elevation_m",
        "elevation_drop_m",
        "slope_pct",
        "bearing_deg",
        "has_weir",
        "weir_count",
        "weir_names",
    ]
    graph_gdf[csv_cols].to_csv("nysa_graph_1km.csv", index=False)
    graph_gdf.to_file("nysa_graph_1km.geojson", driver="GeoJSON")

    print("\nPodsumowanie profilu rzeki:")
    print(f" - Wysokość startu: {elevations[0]:.1f} m n.p.m. -> Ujście: {elevations[-1]:.1f} m n.p.m.")
    print(f" - Całkowita różnica poziomów: {elevations[0] - elevations[-1]:.1f} m")
    print(f" - Wykryte jazy/przeszkody: {graph_gdf['weir_count'].sum()}")
    print(" - Zapisano: nysa_graph_1km.csv oraz nysa_graph_1km.geojson")


if __name__ == "__main__":
    build_river_graph()
import json
import os.path
import subprocess
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Iterable

import requests
from shapely import Point, unary_union
from tqdm import tqdm

from data_processing.image_rendering.tools_rendering import get_city_info, get_city_geojson_expanded
from datasets import  get_all_cities
from paths import PBF_CITIES_DIR, PBF_COUNTRIES_DIR, TMP_DIR

os.makedirs(TMP_DIR, exist_ok=True)
COUNTRIES_SPECIAL_CASES = {
    "kuwait": "saudi-arabia",
    "aman": "saudi-arabia",
    "united arab emirates": "saudi-arabia",
    "oman": "saudi-arabia",
    "qatar": "saudi-arabia",
    "singapore": "malaysia",
    "pr": "pr"

}
EXCEPTIONAL_CASES_URL = {
    "russia": "https://download.geofabrik.de/russia-latest.osm.pbf",
    "united-states": "https://download.geofabrik.de/north-america/us-latest.osm.pbf",
    "australia": "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf",
    "new-zealand": "https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf",
    "saudi-arabia": "https://download.geofabrik.de/asia/gcc-states-latest.osm.pbf",
    "costa-rica": "https://download.geofabrik.de/central-america/costa-rica-latest.osm.pbf",
    "czechia": "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf",
    "guatemala": "https://download.geofabrik.de/central-america/guatemala-latest.osm.pbf",
    "ireland": "https://download.geofabrik.de/europe/ireland-and-northern-ireland-latest.osm.pbf",
    "israel": "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf",
    "malaysia": "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf",
    "turkey": "https://download.geofabrik.de/europe/turkey-latest.osm.pbf",
    "panama": "https://download.geofabrik.de/central-america/panama-latest.osm.pbf",
    "pr": "https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf"
}


def to_wsl_path(p: str | Path) -> str:
    p = str(p).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def get_url(continent, country):
    continent = continent.lower().replace(" ", "-")
    country = country.lower().replace(" ", "-")
    if country in EXCEPTIONAL_CASES_URL:
        return EXCEPTIONAL_CASES_URL[country]
    return f"https://download.geofabrik.de/{continent}/{country}-latest.osm.pbf"


def make_session():
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=8, pool_maxsize=8, max_retries=3
    )
    s.mount("https://", adapter)
    return s


def download_country_data(country, continent, session):
    download_url = get_url(continent, country)
    path = PBF_COUNTRIES_DIR / f"{country.lower().replace(' ', '_')}.osm.pbf"
    if os.path.exists(path):
        return path, download_url
    r = session.get(download_url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = open(path, "wb")
    for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return Path(tmp.name), download_url


def batch_list(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


def expand_polygon(polygon, points, pad=0.03):
    point_buffers = [Point(lon, lat).buffer(pad) for lon, lat in points]
    expanded = unary_union([polygon, *point_buffers])
    return expanded


def extract_cities(pbf_file: Path, cities: Iterable[str], output_dir: Path, batch_size: int = 2):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_wsl = to_wsl_path(output_dir)
    pbf_wsl = to_wsl_path(pbf_file)
    processed_files = []

    for i, city_batch in enumerate(batch_list(cities, batch_size)):

        extracts = []
        batch_temp_files = []
        batch_results = []

        for city in city_batch:
            geom = get_city_geojson_expanded(city)

            safe_name = city.lower().replace(" ", "_").replace(",", "")

            tmp_path = Path(TMP_DIR) / f"tmp_{safe_name}.geojson"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": geom, "properties": {}}]
                }, f)

            batch_temp_files.append(tmp_path)

            out_file_wsl = f"{output_dir_wsl}/{safe_name}.osm.pbf"
            batch_results.append(out_file_wsl)

            extracts.append({
                "output": out_file_wsl,
                "polygon": {"file_name": to_wsl_path(tmp_path)},
            })

        cfg_path = Path(TMP_DIR) / f"osmium_batch_{uuid.uuid4()}.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"extracts": extracts}, f)

        cfg_wsl = to_wsl_path(cfg_path)
        res = None
        try:
            res = subprocess.run(
                [
                    "wsl", "bash", "-lc",
                    f"osmium extract --config '{cfg_wsl}' '{pbf_wsl}' --overwrite "
                ],
                check=True,
                cwd=str(output_dir),
            )
            processed_files.extend(batch_results)

        except subprocess.CalledProcessError as e:
            print(f"Error in batch {i + 1}: {e}")
            print("STDERR:")
            if res:
                print(res.stderr)
            with open(f"tmp_{safe_name}.geojson", "w", encoding="utf-8") as f:
                json.dump({
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": geom, "properties": {}}]
                }, f)
            continue

        finally:
            if cfg_path.exists():
                cfg_path.unlink()
            for f in batch_temp_files:
                if f.exists():
                    f.unlink()

    return processed_files


def build_work_list():
    cities_by_countries = {}
    for city in get_all_cities():
        if city == "San Juan":
            cities_by_countries.setdefault("pr", []).append((city.lower(), ""))
            continue

        country, continent = get_city_info(city)
        country = country.lower()
        if country in COUNTRIES_SPECIAL_CASES:
            country = COUNTRIES_SPECIAL_CASES[country]
        cities_by_countries.setdefault(country, []).append((city.lower(), continent.lower()))

    written = {Path(p.stem).stem.replace("_", " ") for p in PBF_CITIES_DIR.glob("*.osm.pbf")}
    work = []
    for country, city_continent in cities_by_countries.items():
        continent = city_continent[0][1]
        cities_to_write = {c for c, _ in city_continent} - written
        if cities_to_write:
            work.append((country, continent, cities_to_write))
    return work


def run_pipeline(max_parallel_downloads):
    work = build_work_list()

    with ThreadPoolExecutor(max_workers=max_parallel_downloads) as dl_pool, \
            tqdm(total=len(work), desc="Countries") as pbar:

        def process(country, continent, cities):
            session = make_session()
            pbf, url = download_country_data(country, continent, session)
            try:
                r = extract_cities(pbf, cities, PBF_CITIES_DIR)
                if r:
                    os.remove(pbf)
            except Exception as e:
                print(f"Error processing {country} url={url} with {e}")
                traceback.print_exc()
                raise e
            pbar.update(1)

        futures = [
            dl_pool.submit(process, country, continent, cities)
            for country, continent, cities in work
        ]
        for f in as_completed(futures):
            f.result()


if __name__ == "__main__":
    run_pipeline(max_parallel_downloads=4)

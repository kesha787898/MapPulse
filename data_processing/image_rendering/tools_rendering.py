import math
import os
import os.path
import random
from collections import defaultdict

import geopandas as gpd
import pycountry_convert as pc
import requests
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from pyrosm import OSM
from shapely import Point, box, unary_union
from shapely.geometry import shape, mapping

from datasets import  get_pulse_city_points
from paths import CACHE_DIR, PBF_CITIES_DIR
from tools.Cache import Cache
from tools.decorators import cache_result

CELL_SIZE = 0.005
MAX_PER_CELL = 3
POI_PROB = 0.6
EXCEPTIONAL_CITIES = {
    "santiago": "Santiago, Chile",
    "valparaiso": "Valparaiso, Chile",
    "panama city": "Panama City, Panama",
    "athens": "Athens, Greece"
}


def normalize_city(city):
    temp_city = city.lower()
    temp_city = EXCEPTIONAL_CITIES.get(temp_city, None)
    if temp_city == None:
        return city
    else:
        return temp_city


def get_cell(point):
    return (
        math.floor(point.x / CELL_SIZE),
        math.floor(point.y / CELL_SIZE)
    )


style_id = "osm-bright"
base_url = f"http://localhost:8080/styles/{style_id}/static"

geolocator = Nominatim(user_agent="geo_app_V1", timeout=10)

CACHE_ADRESS_GEOJSON = Cache(filename=CACHE_DIR / "address_geojson.json")
CACHE_CITY_ADRESS = Cache(filename=CACHE_DIR / "city_address.json")

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=2,
    max_retries=3,
    error_wait_seconds=3
)


def get_image_url(lon, lat, zoom, resolution):
    url = f"{base_url}/{lon},{lat},{zoom}/{resolution[0]}x{resolution[1]}.png"
    return url


def download_image(lon, lat, zoom, resolution, save_path):
    url = get_image_url(lon, lat, zoom, resolution)
    res = requests.get(url)
    img_data = res.content

    with open(save_path, 'wb') as handler:
        handler.write(img_data)
    return


def expand_geojson_to_points(geojson, points, pad=0.01):
    polygon = shape(geojson)

    if not points:
        return geojson

    point_buffers = [
        Point(lon, lat).buffer(pad)
        for lon, lat in points
    ]

    expanded = unary_union([polygon, *point_buffers])

    return mapping(expanded)


@cache_result(CACHE_ADRESS_GEOJSON)
def get_city_geojson(city):
    city = normalize_city(city)
    location = geocode(city, exactly_one=True, geometry="geojson")
    geojson = location.raw["geojson"]
    if geojson["type"] == "Point":
        min_lat, max_lat, min_lon, max_lon = location.raw["boundingbox"]
        min_lat = float(min_lat)
        max_lat = float(max_lat)
        min_lon = float(min_lon)
        max_lon = float(max_lon)

        return {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]]
        }

    return geojson


def get_city_geojson_expanded(city):
    return expand_geojson_to_points(get_city_geojson(city), get_pulse_city_points().get(city, []))


@cache_result(CACHE_CITY_ADRESS)
def get_city_adress(city):
    city = normalize_city(city)

    location = geolocator.geocode(
        city,
        language="en",
        addressdetails=True
    )

    if not location:
        return None

    address = location.raw["address"]
    return address


def get_city_info(city_name):
    address = get_city_adress(city_name)
    country = address.get("country")
    country_code = address.get("country_code", "").upper()

    continent_code = pc.country_alpha2_to_continent_code(country_code)

    continent_names = {
        "AF": "Africa",
        "AS": "Asia",
        "EU": "Europe",
        "NA": "North America",
        "SA": "South America",
        "OC": "Oceania",
        "AN": "Antarctica",
    }

    continent = continent_names.get(continent_code)

    return country, continent


def get_country_and_continent(city: str):
    location = geolocator.geocode(city, language="en")

    if not location:
        return None, None

    address = location.raw.get("address", {})
    country = address.get("country")
    country_code = address.get("country_code")

    if not country_code:
        return country, None

    country_code = country_code.upper()

    try:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent = pc.convert_continent_code_to_continent_name(continent_code)
    except Exception:
        continent = None

    return country, continent


def get_city_polygon(city_name):
    return shape(get_city_geojson_expanded(city_name))


def normalize_city_name(city_name):
    return city_name.replace(" ", "_").lower()


def get_pois_safe(city_name):
    try:
        return get_pois_local(city_name)
    except Exception as e:
        print(f"Error getting local POIs for {city_name}: {e}. Falling back to remote.")
        return None


def get_pois_local(city_name):
    city_name = normalize_city_name(city_name)
    osm = OSM(str(PBF_CITIES_DIR / f"{city_name}.osm.pbf"))
    pois = osm.get_data_by_custom_criteria(custom_filter={
        "amenity": True,
        "shop": True,
        "tourism": True,
        "office": True}, filter_type="keep",
        keep_nodes=True,
        keep_ways=True,
        keep_relations=False,
    )
    gdf = pois[pois.geometry.type == "Point"]
    return list(gdf.geometry)


def is_civilized_area(osm, bbox, min_poi_count=10, max_nature_percent=30):

    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)

    poi = osm.get_data_by_custom_criteria(
        custom_filter={
            "amenity": True,
            "shop": True,
            "tourism": True,
            "office": True
        },
        filter_type="keep",
        keep_nodes=True,
        keep_ways=False,
        keep_relations=False
    )

    if poi.empty:
        return False

    idx = poi.sindex.query(bbox_poly, predicate="intersects")
    poi_in_bbox = poi.loc[idx]
    poi_in_bbox = poi_in_bbox[poi_in_bbox.intersects(bbox_poly)]

    if len(poi_in_bbox) < min_poi_count:
        return False

    nature = osm.get_data_by_custom_criteria(
        custom_filter={
            "natural": ["wood", "water", "scrub", "sand", "wetland", "bare_rock"],
            "landuse": ["forest", "meadow", "grass", "reservoir"]
        },
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False
    )

    if nature.empty:
        return True

    idx = nature.sindex.query(bbox_poly, predicate="intersects")
    nature_in_bbox = nature.loc[idx]
    nature_in_bbox = nature_in_bbox[nature_in_bbox.intersects(bbox_poly)]

    if nature_in_bbox.empty:
        return True

    crs_utm = poi_in_bbox.estimate_utm_crs()

    nature_proj = nature_in_bbox.to_crs(crs_utm)

    bbox_proj = gpd.GeoSeries([bbox_poly], crs=poi_in_bbox.crs).to_crs(crs_utm).iloc[0]

    nature_area = nature_proj.intersection(bbox_proj).area.sum()
    bbox_area = bbox_proj.area

    nature_ratio = nature_area / (bbox_area + 1e-9) * 100

    if nature_ratio > max_nature_percent:
        return False

    return True


def random_point_in_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p


def random_point_near_poi(poi_point, radius=0.002):
    angle = random.uniform(0, 2 * 3.14159)
    r = random.uniform(0, radius)

    dx = r * math.cos(angle)
    dy = r * math.sin(angle)

    return Point(poi_point.x + dx, poi_point.y + dy)


def is_point_good(point, city_polygon):
    if not city_polygon.contains(point):
        return False

    if city_polygon.boundary.distance(point) < 0.001:
        return False
    return True


def generate_city_patches(city_name, n, zoom, resolution, out_dir, t):
    os.makedirs(out_dir, exist_ok=True)

    city_polygon = get_city_polygon(city_name)

    pois = get_pois_safe(city_name)
    if pois is None:
        return None

    if not pois:
        raise ValueError("No POIs found for city")

    unique_pois = {}
    for poi in pois:
        cell = get_cell(poi)
        if cell not in unique_pois:
            unique_pois[cell] = poi

    pois = list(unique_pois.values())

    grid_counts = defaultdict(int)

    results = []

    i = 0
    attempts = 0
    max_attempts = n * 30

    while i < n and attempts < max_attempts:
        attempts += 1

        if random.random() < POI_PROB:
            base_poi = random.choice(pois)
            point = random_point_near_poi(base_poi)
        else:
            point = random_point_in_polygon(city_polygon)

        if not is_point_good(point, city_polygon):
            continue

        cell = get_cell(point)

        if grid_counts[cell] >= MAX_PER_CELL:
            continue

        lon, lat = point.x, point.y

        file_path = f"{out_dir}/{city_name}_{lon}_{lat}.png"

        if os.path.exists(file_path):
            continue

        try:
            download_image(
                lon,
                lat,
                zoom=zoom,
                resolution=resolution,
                save_path=file_path
            )

            if os.path.getsize(file_path) < 50 * 1024:
                os.remove(file_path)
                continue

        except Exception:
            continue

        results.append({
            "lon": lon,
            "lat": lat,
            "path": file_path
        })

        grid_counts[cell] += 1

        i += 1
        t.update(1)

    return results

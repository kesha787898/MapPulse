import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_processing.image_rendering.tools_rendering import generate_city_patches
from datasets import get_all_cities
from paths import PRETRAINING_IMAGES_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGES_PER_CITY = 1000


def get_nums_images(cities):
    return [IMAGES_PER_CITY] * len(cities)
    # log_areas = []
    # for city in cities:
    #     city_polygon = get_city_polygon(city)
    #     log_areas.append(sqrt( city_polygon.area))
    #
    # result = []
    # for l in log_areas:
    #     weight = l / sum(log_areas)
    #     n_images = min_images_per_city + (total_images - len(cities) * min_images_per_city) * weight
    #     n_images = ceil(n_images)
    #     result.append(n_images)
    # return result


n_images_list = get_nums_images(get_all_cities())
t = tqdm(desc="Rendering images", total=sum(n_images_list))
for city, n_images in zip(get_all_cities(), n_images_list):
    city_dir = PRETRAINING_IMAGES_DIR / city
    processed = len(os.listdir(city_dir)) if city_dir.exists() else 0
    t.update(processed)
    n_needed_to_process = max(0, n_images - processed)
    if n_needed_to_process == 0:
        continue
    generate_city_patches(city, n_needed_to_process, zoom=17, resolution=(512, 512), out_dir=city_dir, t=t)

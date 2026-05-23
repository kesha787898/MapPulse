from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from tqdm import tqdm

from datasets import get_pulse_df_train, get_pulse_df_test
from paths import TRAINING_IMAGES_DIR
from data_processing.image_rendering.tools_rendering import download_image


def download_if_not_exists(task):
    lon, lat, place, hash_value = task

    path = TRAINING_IMAGES_DIR / place / f"{hash_value}.png"
    os.makedirs(path.parent, exist_ok=True)
    if path.exists():
        return

    download_image(
        lon,
        lat,
        zoom=17,
        resolution=(512, 512),
        save_path=path,
    )


if __name__ == "__main__":
    coords = set()

    for df in [get_pulse_df_train(), get_pulse_df_test()]:
        coords.update(
            zip(df["long_left"], df["lat_left"], df["place_name_left"], df["place_id_left_new"])
        )
        coords.update(
            zip(df["long_right"], df["lat_right"], df["place_name_right"], df["place_id_right_new"])
        )

    print(f"Unique images: {len(coords)}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(
            tqdm(
                pool.map(download_if_not_exists, coords),
                total=len(coords),
            )
        )

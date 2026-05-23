import os

import pandas as pd

from paths import FULL_DATASET_CSV, PRETRAINING_IMAGES_DIR, FULL_DATASET_CSV_PREPARED

if __name__ == '__main__':
    full_df = pd.read_csv(FULL_DATASET_CSV)
    images = []
    for city in full_df["City"].unique():
        for i in os.listdir(PRETRAINING_IMAGES_DIR / city):
            if city in i:
                images.append({"City": city, "path": os.path.join(PRETRAINING_IMAGES_DIR / city, i)})
    full_df_image_image_indes = pd.DataFrame(images)
    full_df = pd.merge(full_df, full_df_image_image_indes, on="City", how="inner")
    full_df.to_csv(FULL_DATASET_CSV_PREPARED, index=False)

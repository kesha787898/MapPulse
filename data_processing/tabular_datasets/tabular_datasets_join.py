import os

import pandas as pd

from paths import NUMBEO_CSV, LIVEABILITY_CSV, FULL_DATASET_CSV, PRETRAINING_IMAGES_DIR

if __name__ == '__main__':
    numbeo_df = pd.read_csv(NUMBEO_CSV)
    liveability_df = pd.read_csv(LIVEABILITY_CSV)
    numbeo_df["City"] = numbeo_df["City"].str.split(",").str[0].str.replace(r"\(.*?\)", "", regex=True)
    full_df = pd.merge(numbeo_df, liveability_df, on="City", how="inner")
    full_df = full_df[
        ["City", "Safety Index", "Traffic Commute Time Index", "Pollution Index", "Culture &Environment",
         "Infrastructure"]]
    full_df.to_csv(FULL_DATASET_CSV, index=False)
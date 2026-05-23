from collections import defaultdict
from functools import lru_cache

import pandas as pd

from paths import PLACE_PULSE_DF_CSV, FULL_DATASET_CSV, \
    PREPARED_PLACE_PULSE_DATASET_CSV_TEST, PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN, FULL_DATASET_CSV_PREPARED

dict_cities = {"Rio De Janeiro": "Rio de Janeiro"}


@lru_cache(maxsize=1)
def get_pulse_df():
    df = pd.read_csv(PLACE_PULSE_DF_CSV)
    df = df.rename(columns={
        "long_right": "lat_right",
        "lat_right": "long_right",
        "long_left": "lat_left",
        "lat_left": "long_left",
    })

    df["place_name_left"] = df["place_name_left"].replace(dict_cities)
    df["place_name_right"] = df["place_name_right"].replace(dict_cities)
    return df


@lru_cache(maxsize=1)
def get_pulse_uniq_cities():
    return set(get_pulse_df()["place_name_right"].unique()).union(set(get_pulse_df()["place_name_left"].unique()))


@lru_cache(maxsize=1)
def get_full_tabular_dataset():
    dataset = pd.read_csv(FULL_DATASET_CSV)
    return dataset


@lru_cache(maxsize=1)
def get_full_tabular_dataset_uniq_cities():
    dataset = set(get_full_tabular_dataset()["City"].unique())
    return dataset


@lru_cache(maxsize=1)
def get_all_cities():
    return sorted(list(get_pulse_uniq_cities().union(get_full_tabular_dataset_uniq_cities())))


@lru_cache(maxsize=1)
def get_full_tabular_dataset_prepared():
    return pd.read_csv(FULL_DATASET_CSV_PREPARED)


@lru_cache(maxsize=1)
def get_pulse_df_train():
    df = pd.read_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN)
    df = df.dropna(axis=0, subset=["study_question"])
    return df


@lru_cache(maxsize=1)
def get_pulse_df_test():
    df = pd.read_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TEST)
    df = df.dropna(axis=0, subset=["study_question"])

    return df


@lru_cache(maxsize=1)
def get_features_pulse():
    return list(
        sorted(set(get_pulse_df_train()["study_question"].unique()).union(
            set(get_pulse_df_test()["study_question"].unique()))))

@lru_cache(maxsize=1)
def get_pulse_city_points():
    points = defaultdict(set)

    for df in [get_pulse_df()]:
        for row in df.itertuples():
            points[row.place_name_left].add(
                (row.long_left, row.lat_left)
            )
            points[row.place_name_right].add(
                (row.long_right, row.lat_right)
            )

    points = {
        city.lower(): list(points)
        for city, points in points.items()
    }
    return points


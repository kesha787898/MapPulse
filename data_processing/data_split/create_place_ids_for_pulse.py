from datasets import get_pulse_df_train, get_pulse_df_test
from paths import (
    PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN,
    PREPARED_PLACE_PULSE_DATASET_CSV_TEST,
)



def collect_all_coords(df_train, df_test):
    coords = set()

    for df in [df_train, df_test]:
        for side in ["left", "right"]:
            for _, row in df.iterrows():
                coords.add((
                    row[f"lat_{side}"],
                    row[f"long_{side}"],
                ))

    return sorted(list(coords))


def build_coord_mapping(coords):
    return {coord: f"place_{i:08d}" for i, coord in enumerate(coords)}


def apply_mapping(df, coord_map):
    def map_row(row, side):
        key = (
            row[f"lat_{side}"],
            row[f"long_{side}"],
        )
        return coord_map[key]

    df = df.copy()

    df["place_id_left_new"] = df.apply(lambda r: map_row(r, "left"), axis=1)
    df["place_id_right_new"] = df.apply(lambda r: map_row(r, "right"), axis=1)

    return df



train_df = get_pulse_df_train().copy()
test_df = get_pulse_df_test().copy()

all_coords = collect_all_coords(train_df, test_df)
coord_map = build_coord_mapping(all_coords)

print(f"Unique places: {len(coord_map)}")


train_prepared = apply_mapping(train_df, coord_map)
test_prepared = apply_mapping(test_df, coord_map)


train_prepared.to_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN, index=False)
test_prepared.to_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TEST, index=False)
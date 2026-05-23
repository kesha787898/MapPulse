import random
from datasets import get_pulse_df, \
    get_pulse_uniq_cities, get_full_tabular_dataset_uniq_cities, get_full_tabular_dataset_prepared
from paths import PREPARED_TABULAR_DATASET_CSV_TRAIN, PREPARED_TABULAR_DATASET_CSV_TEST, \
    PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN, PREPARED_PLACE_PULSE_DATASET_CSV_TEST

random.seed(42)
TRAIN_SIZE = 0.8

tabular_only = get_full_tabular_dataset_uniq_cities().difference(get_pulse_uniq_cities())
pulse_only = get_pulse_uniq_cities().difference(get_full_tabular_dataset_uniq_cities())
shared_cities = list(get_pulse_uniq_cities().intersection(get_full_tabular_dataset_uniq_cities()))

shared_test_sampled = random.sample(shared_cities, int(len(shared_cities) * (1 - TRAIN_SIZE)))
shared_test = set(shared_test_sampled)
shared_train = set(shared_cities).difference(shared_test)

train_for_tabular_cities = tabular_only.union(shared_train)
test_for_tabular_cities = shared_test

train_for_place_pulse_cities = shared_train
test_for_place_pulse_cities = pulse_only.union(shared_test)

test_for_pulse_df = get_pulse_df()[(get_pulse_df()["place_name_left"].isin(test_for_place_pulse_cities)) &
                                   (get_pulse_df()["place_name_right"].isin(test_for_place_pulse_cities))]

train_for_pules_df = get_pulse_df()[(get_pulse_df()["place_name_left"].isin(train_for_place_pulse_cities)) &
                                    (get_pulse_df()["place_name_right"].isin(train_for_place_pulse_cities))]

train_tabular_dataset = get_full_tabular_dataset_prepared()[
    get_full_tabular_dataset_prepared()["City"].isin(train_for_tabular_cities)]
test_tabular_dataset = get_full_tabular_dataset_prepared()[
    get_full_tabular_dataset_prepared()["City"].isin(test_for_tabular_cities)]

print(f"total_for_place_pulse_points={len(get_pulse_df())}")
print(f"train_for_place_pulse_points={len(train_for_pules_df)}")
print(f"test_for_place_pulse_points={len(test_for_pulse_df)}")
print(f"total_for_tabular_points={len(get_full_tabular_dataset_prepared())}")
print(f"train_for_tabular_points={len(train_tabular_dataset)}")
print(f"test_for_tabular_points={len(test_tabular_dataset)}")
train_tabular_dataset.to_csv(PREPARED_TABULAR_DATASET_CSV_TRAIN, index=False)
test_tabular_dataset.to_csv(PREPARED_TABULAR_DATASET_CSV_TEST, index=False)
train_for_pules_df.to_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN, index=False)
test_for_pulse_df.to_csv(PREPARED_PLACE_PULSE_DATASET_CSV_TEST, index=False)

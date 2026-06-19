import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
PLACE_PULSE_DIR = DATA_DIR / "place_pulse_v2"
PLACE_PULSE_DF_CSV = PLACE_PULSE_DIR / "votes_clean.csv"
PBF_FILES_DIR = DATA_DIR / "pbf"
PBF_CITIES_DIR = PBF_FILES_DIR / "city"
PBF_COUNTRIES_DIR = PBF_FILES_DIR / "country"

TABULAR_DATASETS_DIR = DATA_DIR / "tabular_datasets"

LIVEABILITY_PDF = TABULAR_DATASETS_DIR / "liveability.pdf"
LIVEABILITY_CSV = TABULAR_DATASETS_DIR / "liveability.csv"
NUMBEO_CSV = TABULAR_DATASETS_DIR / "numbeo.csv"
FULL_DATASET_CSV = TABULAR_DATASETS_DIR / "full_dataset.csv"
FULL_DATASET_CSV_PREPARED = TABULAR_DATASETS_DIR / "full_dataset_prepared.csv"
PREPARED_TABULAR_DATASET_CSV_TRAIN = TABULAR_DATASETS_DIR / "train_tabular_dataset.csv"
PREPARED_TABULAR_DATASET_CSV_TEST = TABULAR_DATASETS_DIR / "test_tabular_dataset.csv"
PREPARED_PLACE_PULSE_DATASET_CSV_TRAIN = TABULAR_DATASETS_DIR / "train_place_pulse_dataset.csv"
PREPARED_PLACE_PULSE_DATASET_CSV_TEST = TABULAR_DATASETS_DIR / "test_place_pulse_dataset.csv"
MBTILES_DIR = DATA_DIR / "raw"
MBTILES_FILES = [MBTILES_DIR / file for file in os.listdir(MBTILES_DIR) if
                 file.endswith(".mbtiles") and not file.startswith("merged")]

PRETRAINING_IMAGES_DIR = DATA_DIR / "pretraining_images"
TRAINING_IMAGES_DIR = DATA_DIR / "training_images"
TMP_DIR = DATA_DIR / "tmp"
EMBEDDING_DIR = DATA_DIR / "embeddings"


MODELS_DIR = DATA_DIR / "models"

CUSTOM_MODEL_CHKPT = MODELS_DIR / r"pretrainin\version_13\checkpoints\epoch=48-step=3871.ckpt"

MODELS_TRAIN_CHKPTS = MODELS_DIR / r"train"
MODELS_EVAL_OUT = PROJECT_ROOT / r"results"

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
TOKENIZERS_DIR = PROJ_ROOT / "nlpinitiative" / "data_preparation" / "tokenizers"
CONV_SCHEMA_DIR = PROJ_ROOT / "nlpinitiative" / "data_preparation" / "conversion_schema"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# tokenizer config
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
GENERATOR_BATCH_SIZE = 1000

# Model defaults
DEF_MODEL = "bert-base-uncased"

# dataset structure
BINARY_LABELS = ["DISCRIMINATORY", "NEUTRAL"]
CATEGORY_LABELS = ["GENDER", "RACE", "SEXUALITY", "DISABILITY", "RELIGION", "UNSPECIFIED"]
DATASET_COLS = ["TEXT"] + BINARY_LABELS + CATEGORY_LABELS
TRAIN_TEST_SPLIT = 0.3  ## Default of 0.3 is standard for most model pipelines

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

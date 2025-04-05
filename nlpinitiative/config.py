import toml
import typer
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated

from dotenv import load_dotenv
from loguru import logger

app = typer.Typer()

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

try:
    logger.info("Loading pyproject.toml...")
    with open(PROJ_ROOT / "pyproject.toml", "r") as f:
        config = toml.load(f, dict)
        f.close()

    if "repositories" not in config.keys(): # pragma: no cover
        config["repositories"] = {
            "bin_repo": "",
            "ml_repo": "",
            "ds_repo": "",
            "streamlit_repo": "",
        }
    else: # pragma: no cover
        if "bin_repo" not in config["repositories"].keys():
            config["repositories"]["bin_repo"] = ""
        if "ml_repo" not in config["repositories"].keys():
            config["repositories"]["ml_repo"] = ""
        if "ds_repo" not in config["repositories"].keys():
            config["repositories"]["ds_repo"] = ""
        if "streamlit_repo" not in config["repositories"].keys():
            config["repositories"]["streamlit_repo"] = ""

    if "project-urls" not in config.keys(): # pragma: no cover
        config["project-urls"] = {
            "hf_space_base_url": "",
            "hf_model_base_url": "",
            "hf_dataset_base_url": "",
        }
    else: # pragma: no cover
        if "hf_space_base_url" not in config["project-urls"].keys():
            config["project-urls"]["hf_space_base_url"] = ""
        if "hf_model_base_url" not in config["project-urls"].keys():
            config["project-urls"]["hf_model_base_url"] = ""
        if "hf_dataset_base_url" not in config["project-urls"].keys():
            config["project-urls"]["hf_dataset_base_url"] = ""

    logger.success("pyproject.toml loaded successfully.")
except Exception as e: # pragma: no cover
    logger.error(e)

# Dataset directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NORM_SCHEMA_DIR = DATA_DIR / "normalization_schema"

# Testing Directories
TEST_DATA_DIR = PROJ_ROOT / "test" / "test_files"

# Model directories
MODELS_DIR = PROJ_ROOT / "models"

# tokenizer config
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
GENERATOR_BATCH_SIZE = 1000

# Model defaults
DEF_MODEL = "bert-base-uncased"

# dataset structure
BINARY_LABELS = ["DISCRIMINATORY"]
CATEGORY_LABELS = ["GENDER", "RACE", "SEXUALITY", "DISABILITY", "RELIGION", "UNSPECIFIED"]
DATASET_COLS = ["TEXT"] + BINARY_LABELS + CATEGORY_LABELS
TRAIN_TEST_SPLIT = 0.3  ## Default of 0.3 is standard for most model pipelines

# HF Hub Repositories
BIN_REPO = config["repositories"]["bin_repo"]
ML_REPO = config["repositories"]["ml_repo"]
DATASET_REPO = config["repositories"]["ds_repo"]
STREAMLIT_REPO = config["repositories"]["streamlit_repo"]


@app.command()
def main(
    bin_repo: Annotated[str, typer.Option("--binary-repo", "-br")] = None,
    ml_repo: Annotated[str, typer.Option("--multilabel-regression-repo", "-mr")] = None,
    ds_repo: Annotated[str, typer.Option("--dataset-repo", "-dr")] = None,
    streamlit_repo: Annotated[str, typer.Option("--streamlit-repo", "-sr")] = None,
    hf_space_base_url: Annotated[str, typer.Option("--space-url", "-su")] = None,
    hf_model_base_url: Annotated[str, typer.Option("--model-url", "-mu")] = None,
    hf_dataset_base_url: Annotated[str, typer.Option("--dataset-url", "-du")] = None,
): # pragma: no cover
    toml_edited = False

    if bin_repo is not None and len(bin_repo) > 0:
        config["repositories"]["bin_repo"] = bin_repo
        toml_edited = True
        logger.success(f"Successfully updated binary repository to {bin_repo}.")

    if ml_repo is not None and len(ml_repo) > 0:
        config["repositories"]["ml_repo"] = ml_repo
        toml_edited = True
        logger.success(f"Successfully updated binary repository to {ml_repo}.")

    if ds_repo is not None and len(ds_repo) > 0:
        config["repositories"]["ds_repo"] = ds_repo
        toml_edited = True
        logger.success(f'Successfully updated binary repository to "{ds_repo}".')

    if streamlit_repo is not None and len(streamlit_repo) > 0:
        config["repositories"]["streamlit_repo"] = streamlit_repo
        toml_edited = True
        logger.success(f'Successfully updated Streamlit repository to "{streamlit_repo}".')

    if hf_space_base_url is not None and len(hf_space_base_url) > 0:
        config["project-urls"]["hf_space_base_url"] = hf_space_base_url
        toml_edited = True
        logger.success(f'Successfully updated HF Space base URL to "{hf_space_base_url}".')

    if hf_model_base_url is not None and len(hf_model_base_url) > 0:
        config["project-urls"]["hf_model_base_url"] = hf_model_base_url
        toml_edited = True
        logger.success(f'Successfully updated HF Model base URL to "{hf_model_base_url}".')

    if hf_dataset_base_url is not None and len(hf_dataset_base_url) > 0:
        config["project-urls"]["hf_dataset_base_url"] = hf_dataset_base_url
        toml_edited = True
        logger.success(f'Successfully updated HF Dataset base URL to "{hf_dataset_base_url}".')

    if toml_edited:
        with open(PROJ_ROOT / "pyproject.toml", "w") as f:
            toml.dump(config, f)
            f.close()


if __name__ == "__main__": # pragma: no cover
    app()

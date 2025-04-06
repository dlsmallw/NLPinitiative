import os
import toml
import typer
import validators
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated

from dotenv import load_dotenv, set_key, get_key
from loguru import logger

app = typer.Typer()


PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Load environment variables from .env file if it exists
dotenv_path = os.path.join(PROJ_ROOT, '.env')
load_dotenv(dotenv_path)

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


# Paths
## Dataset directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NORM_SCHEMA_DIR = DATA_DIR / "normalization_schema"

## Testing Directories
TEST_DATA_DIR = PROJ_ROOT / "test" / "test_files"

## Model directories
MODELS_DIR = PROJ_ROOT / "models"

## tokenizer config
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
GENERATOR_BATCH_SIZE = 1000

## Model defaults
DEF_MODEL = "bert-base-uncased"

## dataset structure
BINARY_LABELS = ["DISCRIMINATORY"]
CATEGORY_LABELS = ["GENDER", "RACE", "SEXUALITY", "DISABILITY", "RELIGION", "UNSPECIFIED"]
DATASET_COLS = ["TEXT"] + BINARY_LABELS + CATEGORY_LABELS
TRAIN_TEST_SPLIT = 0.3  ## Default of 0.3 is standard for most model pipelines

## HF Hub Repositories
BIN_REPO = config["repositories"]["bin_repo"]
ML_REPO = config["repositories"]["ml_repo"]
DATASET_REPO = config["repositories"]["ds_repo"]
STREAMLIT_REPO = config["repositories"]["streamlit_repo"]

## Tokens
HF_TOKEN = get_key(dotenv_path, "HF_TOKEN")
if HF_TOKEN is None:
    set_key((dotenv_path), "HF_TOKEN", "")
    HF_TOKEN = get_key(dotenv_path, "HF_TOKEN")


def set_repo_id(repo_type: str, repo_id: str):
    """
    Set the repository ID in the .env file.

    Parameters
    ----------
    repo_type : str
        The type of repository (bin_repo, ml_repo, ds_repo, streamlit_repo).
    repo_id : str
        The repository ID to set.
    """

    config["repositories"][repo_type] = repo_id
    logger.success(f'Successfully updated {repo_type} to "{repo_id}".')

def set_base_url(url_type: str, base_url: str):
    """
    Set the base URL in the .env file.

    Parameters
    ----------
    url_type : str
        The type of URL (hf_space_base_url, hf_model_base_url, hf_dataset_base_url).
    base_url : str
        The base URL to set.
    """

    config["project-urls"][url_type] = base_url
    logger.success(f'Successfully updated {url_type} to "{base_url}".')

def set_token(token_type: str, token: str):
    """
    Set the token in the .env file.

    Parameters
    ----------
    token_type : str
        The type of token (HF_TOKEN).
    token : str
        The token to set.
    """

    set_key(dotenv_path, token_type, token)
    logger.success(f'Successfully updated {token_type} to "{token}".')

@app.command()
def main(
    bin_repo: Annotated[str, typer.Option("--binary-repo", "-br")] = None,
    ml_repo: Annotated[str, typer.Option("--multilabel-regression-repo", "-mr")] = None,
    ds_repo: Annotated[str, typer.Option("--dataset-repo", "-dr")] = None,
    streamlit_repo: Annotated[str, typer.Option("--streamlit-repo", "-sr")] = None,
    hf_space_base_url: Annotated[str, typer.Option("--space-url", "-su")] = None,
    hf_model_base_url: Annotated[str, typer.Option("--model-url", "-mu")] = None,
    hf_dataset_base_url: Annotated[str, typer.Option("--dataset-url", "-du")] = None,
    hf_token: Annotated[str, typer.Option("--hf-token", "-ht")] = None,
): # pragma: no cover
    toml_edited = False

    repo_param_list = [
        ("bin_repo", bin_repo), 
        ("ml_repo", ml_repo), 
        ("ds_repo", ds_repo), 
        ("streamlit_repo", streamlit_repo)
    ]

    url_param_list = [
        ("hf_space_base_url", hf_space_base_url), 
        ("hf_model_base_url", hf_model_base_url), 
        ("hf_dataset_base_url", hf_dataset_base_url)
    ]

    token_param_list = [
        ("HF_TOKEN", hf_token)
    ]

    for repo_type, repo_id in repo_param_list:
        if repo_id is not None and len(repo_id) > 0:
            set_repo_id(repo_type, repo_id)
            toml_edited = True

    for url_type, base_url in url_param_list:
        if base_url is not None and len(base_url) > 0:
            if validators.url(base_url) is False:
                logger.error(f"Invalid URL for {url_type}: {base_url}. Please provide a valid URL.")
            else:
                set_base_url(url_type, base_url)
                toml_edited = True

    for token_type, token in token_param_list:
        if token is not None and len(token) > 0:
            set_token(token_type, token)

    if toml_edited:
        with open(PROJ_ROOT / "pyproject.toml", "w") as f:
            toml.dump(config, f)
            f.close()


if __name__ == "__main__": # pragma: no cover
    app()

from pathlib import Path

import typer
from typing_extensions import Annotated, Optional, Required
from loguru import logger
from tqdm import tqdm
import os
import pandas as pd
import json

from nlpinitiative.config import (
    RAW_DATA_DIR, 
    EXTERNAL_DATA_DIR, 
    INTERIM_DATA_DIR, 
    CONV_SCHEMA_DIR, 
    DATASET_COLS
)

app = typer.Typer()

def valid_filepath(path: Path):
    return os.path.exists(path)

def load_src_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_conv_schema(path: Path) -> dict[str:str]:
    return json.load(open(path, 'r'))

def convert_to_master_schema(filesrc: Path, cv_path: Path):
    src_df = load_src_file(filesrc)
    conv_scema = load_conv_schema(cv_path)
    
    master_df = pd.DataFrame(data=[], columns=DATASET_COLS)

    for index, row in src_df.iterrows():
        pass


@app.command()
def main(
    filename: Annotated[str, typer.Option('--dataset', '-d')],
    conv_schema_filename: Annotated[str, typer.Option('--conv-schema', '-cv')],
    raw_flag : bool = typer.Option(False, '--ext', '-e'),
    ext_flag : bool = typer.Option(False, '--raw', '-r')
):
    
    print(DATASET_COLS)
    
    if raw_flag and ext_flag:
        logger.error('Only one directory flag can be used')
        return
    
    if not conv_schema_filename or filename:
        logger.error('Missing conversion schema or filename argument')
        return
    
    
    
    filepath = RAW_DATA_DIR / filename if raw_flag else EXTERNAL_DATA_DIR / filename
    schema_path = CONV_SCHEMA_DIR / conv_schema_filename
    if not valid_filepath(filepath): 
        logger.error('The file specified does not exist within the chosen directory')
        return
    if not valid_filepath(schema_path):
        logger.error('The schema file specified does not exist within the conversion_schema directory')
        return
    
    try:
        convert_to_master_schema(filepath, schema_path)
    except Exception as e:
        logger.error(e)



if __name__ == "__main__":
    app()
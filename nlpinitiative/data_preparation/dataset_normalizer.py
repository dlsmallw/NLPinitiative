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
    DATASET_COLS,
    CATEGORY_LABELS
)

app = typer.Typer()

def valid_filepath(path: Path):
    return os.path.exists(path)

def load_src_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_conv_schema(path: Path) -> dict[str:str]:
    return json.load(open(path, 'r'))

def store_normalized_dataset(df: pd.DataFrame, filename: str):
    destpath = INTERIM_DATA_DIR
    ## Handles situations of duplicate filenames
    appended_num = 0
    corrected_filename = f'{filename}.csv'
    while os.path.exists(os.path.join(destpath, corrected_filename)):
        appended_num += 1
        corrected_filename = f'{filename}-{appended_num}.csv'
    df.to_csv(os.path.join(destpath, corrected_filename), index=False)

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, merge_col: str) -> pd.DataFrame:
    new_df = pd.merge(df1, df2, on=merge_col, how='left').fillna(0.0)
    return new_df

def convert_to_master_schema(files: list[Path], cv_path: Path, export_name: str):
    conv_scema = load_conv_schema(cv_path)
    data_col_name = conv_scema['data_col']
    schema_cats = conv_scema['column_mapping'].keys()

    num_files = len(files)
    src_df = None
    for i in range(0, num_files):
        df = load_src_file(files[i])
        if src_df is not None:
            src_df = merge_dataframes(src_df, df, data_col_name)
        else:
            src_df = df
    
    master_df = pd.DataFrame(data=[], columns=DATASET_COLS)
    master_df[DATASET_COLS[0]] = src_df[data_col_name]
    for cat in schema_cats:
        from_columns = conv_scema['column_mapping'][cat]
        if len(from_columns) > 0:
            if cat == DATASET_COLS[1]:
                master_df[cat] = src_df[from_columns].gt(0.0).astype(pd.Int64Dtype())
            else:
                master_df[cat] = src_df[from_columns].sum(axis=1)
        else:
            master_df[cat] = 0.0

    cols = master_df.columns
    for col in cols:
        if "unnamed" in col.lower():
            master_df.drop(col)

    indices_to_purge = []
    for index, row in master_df.iterrows():
        is_hate = row[DATASET_COLS[1]] == 1
        has_cat_values = row[CATEGORY_LABELS].sum() > 0.0

        if is_hate and not has_cat_values \
            or not is_hate and has_cat_values:
            indices_to_purge.append(index)

    master_df.drop(master_df.index[indices_to_purge], inplace=True)
    return master_df
    

@app.command()
def main(
    filenames: Annotated[list[str], typer.Option('--dataset', '-d')],
    conv_schema_filename: Annotated[str, typer.Option('--conv-schema', '-cv')],
    ext_flag : bool = typer.Option(False, '--ext', '-e'),
    new_name : Annotated[str, typer.Option('--new-name', '-n')] = None
):  
    if not conv_schema_filename or filenames:
        logger.error('Missing conversion schema or filename argument')
        return
    
    
    export_name = filename if new_name is None else new_name
    schema_path = CONV_SCHEMA_DIR / conv_schema_filename

    filepaths = []
    for filename in filenames:
        filepath = EXTERNAL_DATA_DIR / filename if ext_flag else RAW_DATA_DIR / filename
        if not valid_filepath(filepath): 
            logger.error('The file specified does not exist within the chosen directory')
            return
        filepaths.append(filepath)

    if not valid_filepath(schema_path):
        logger.error('The schema file specified does not exist within the conversion_schema directory')
        return
    
    try:
        df = convert_to_master_schema(filepath, schema_path, export_name)
        store_normalized_dataset(df, export_name)
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    app()
"""
Script file used for facillitating normalization of a third-party dataset(s) into
the format that we will utilize for training the model. This script can also handle 
merging complimentary datasets (datasets that come from the same source that may have minor
differences in labeling scheme).
"""

from typing_extensions import Annotated
from loguru import logger
from pathlib import Path
import pandas as pd
import typer
import json
import os

from nlpinitiative.config import (
    DATASET_COLS,
    CATEGORY_LABELS,
    RAW_DATA_DIR
)

app = typer.Typer()

class DataNormalizer:
    ## Handles merging complimentary datasets into a single dataset
    def _merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, merge_col: str) -> pd.DataFrame:
        new_df = pd.merge(df1, df2, on=merge_col, how='left').fillna(0.0)
        return new_df

    ## Loads a csv file into a dataframe object
    def _load_src_file(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    ## Loads the dataset conversion scheme that will be used for normalizing a dataset
    def _load_conv_schema(self, path: Path) -> dict[str:str]:
        return json.load(open(path, 'r'))

    ## Handles the process for normalizing the third-party datasets
    def normalize_datasets(self, files: list[Path], cv_path: Path):
        num_files = len(files)
        src_df = None
        for i in range(0, num_files):
            df = self._load_src_file(RAW_DATA_DIR / files[i])
            if src_df is not None:
                src_df = self._merge_dataframes(src_df, df, data_col_name)
            else:
                src_df = df

        conv_scema = self._load_conv_schema(cv_path)
        data_col_name = conv_scema['data_col']

        if conv_scema['mapping_type'] == 'many2many':
            schema_cats = conv_scema['column_mapping'].keys()

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
        else:
            source_column = conv_scema['single_column_label']
            cat_mapping = conv_scema['column_mapping']

            data = []
            for _, row in src_df.iterrows():
                row_data = []

                text = row[data_col_name]
                type_discr = row[source_column]

                row_data.append(text)
                row_data.append(1)

                groups = []
                for cat in cat_mapping.keys():
                    if type_discr in cat_mapping[cat]:
                        if cat not in groups:
                            groups.append(cat)
                try:
                    val_breakdown = 1 / len(groups)
                    
                    for cat in cat_mapping.keys():
                        val = 0.0
                        if cat in groups:
                            val = val_breakdown
                        row_data.append(val)

                    data.append(row_data)
                except:
                    pass
            master_df = pd.DataFrame(data=data, columns=DATASET_COLS)
        return master_df
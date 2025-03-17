"""
Script file used for facillitating dataset importing either from a 
local source or from a remote source.
"""

from typing_extensions import Annotated, Optional
from loguru import logger
from urllib.parse import urlparse
import os, typer
import pandas as pd
from pathlib import Path

from nlpinitiative.config import (
    DATA_DIR,
    RAW_DATA_DIR, 
    RAW_DATA_DIR
)

app = typer.Typer()
accepted_formats = ['.csv', '.xlsx', '.json']

class DataImporter:
    def __init__(self):
        pass

    ## Checks that the url provided is valid
    def _is_valid_url(self, url):
        if url:
            parsed_url = urlparse(url)
            return bool(parsed_url.scheme in ["http", "https", "ftp"])
        else:
            return False

    ## Generates a filename that will be used when importing new datasets
    def gen_import_filename(self, url: str):
        def github():
            parsed = urlparse(url)
            path = os.path.splitext(parsed.path)[0][1:]
            path_arr = path.split('/')
            return '_'.join([path_arr[0], path_arr[1], path_arr[-1]])

        if 'github' in url:
            return github()
        else:
            split_arr = url.split('/')
            return split_arr[-1]
        
    ## Converts URLs to a format that can be used for importing data from a remote source
    ## NOTE: Currently only used for converting GitHub urls to the appropriate format
    def format_url(self, url: str):
        def github():
            base_url = 'https://raw.githubusercontent.com'
            if base_url in url:
                return url
            else:
                updated_url = url.replace('https://github.com', base_url)
                updated_url = updated_url.replace('blob', 'refs/heads')
                return updated_url
            
        if 'github' in url:
            logger.info(f"Source url identified as GitHub URL, {url}")
            formatted_url = github()
            logger.info(f"URL Formatted, {formatted_url}")
            return formatted_url
        else:
            return url
        
    ## Converts a dataset to a dataframe and also properly handles csv files with atypical delimiters
    def srcdata_to_df(self, source: str, ext: str) -> pd.DataFrame:
        try:
            match ext:
                case '.csv':
                    try:
                        df = pd.read_csv(source)
                    except:
                        df = pd.read_csv(source, delimiter=';')
                case '.xlsx':
                    df = pd.read_excel(source)
                case '.json':
                    df = pd.read_json(source)
                case _:
                    df = None
            return df
        except Exception as e:
            raise Exception(f"Failed to import from source - {e}")

    ## Handles importing a dataset from the local system using the filepath
    def import_from_local_source(self, filepath, tp_src=False) -> pd.DataFrame:
        if filepath is None or filepath == '':
            raise Exception("Filepath not provided")
        if not os.path.exists(filepath):
            raise Exception(f"Invalid filepath, {filepath}")
        
        destpath = RAW_DATA_DIR if tp_src else RAW_DATA_DIR
        tail = os.path.split(filepath)[-1]
        filename, ext = os.path.splitext(tail)[-2:]
        print(filename)
        if ext not in accepted_formats:
            raise Exception("Unsupported file type")
        
        df = srcdata_to_df(filepath, ext)

        if df is None:
            raise Exception("Failed to import from local source")
        
        logger.success(f"Data from file, {filepath}, imported")
        store_data(df, filename, destpath)
        return df
            
    ## Handles importing data from an external source (i.e., GitHub)      
    def import_from_ext_source(self, ext_src):
        if not _is_valid_url(ext_src):
            raise Exception("URL not provided")
            
        url = format_url(ext_src)
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        logger.info(f"File type identified, '{ext}'")

        if ext not in accepted_formats:
            raise Exception("Unsupported file type")
        
        new_filename = gen_import_filename(ext_src)
        destpath = RAW_DATA_DIR

        df = srcdata_to_df(url, ext)

        if df is None:
            raise Exception('Failed to extract data from the given url')
        
        logger.success(f"Successfully imported the dataset from, {ext_src}")
        store_data(df, new_filename, destpath)
        return df

    ## Stores the imported dataset within the data directory
    def store_data(self, data_df: pd.DataFrame, filename: str, destpath: Path):
        ## Handles situations of duplicate filenames
        appended_num = 0
        corrected_filename = f'{filename}.csv'
        while os.path.exists(os.path.join(destpath, corrected_filename)):
            appended_num += 1
            corrected_filename = f'{filename}-{appended_num}.csv'
        data_df.to_csv(os.path.join(destpath, corrected_filename))

## Handles logic when calling the script from cmd or terminal
@app.command()
def main(
        local: Annotated[str, typer.Option("--local-path", "-l")] = None,
        external: Annotated[str, typer.Option("--ext-path", "-e")] = None,
        third_party_source: Optional[Annotated[str, typer.Option("--third-party", "-tp")]] = False
    ):

    try:
        if local or external:
            if local:
                logger.info("Import from local source")
                df = import_from_local_source(local, third_party_source)
            else:
                logger.info("Import from external source")
                df = import_from_ext_source(external)
        else:
            logger.error("No source specified")
        return df
    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    app()
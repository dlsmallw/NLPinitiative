from typing_extensions import Annotated, Optional
from loguru import logger
from urllib.parse import urlparse
import os, typer
import pandas as pd

from nlpinitiative.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
accepted_formats = ['.csv', '.xlsx', '.json']

def valid_url_scheme(url):
    if url:
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme in ["http", "https", "ftp"])
    else:
        return False
    
def gen_import_filename(url: str):
    def github():
        parsed = urlparse(url)
        path = parsed.path
        print(path)
        path_arr = path.split('/')[1:]
        print(path_arr)
        return '_'.join([path_arr[0], path_arr[1], path_arr[-1]])

    if 'github' in url:
        return github()
    else:
        split_arr = url.split('/')
        return split_arr[-1]
    
def format_url(url: str):
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
    

def srcdata_to_df(source: str, ext: str) -> pd.DataFrame:
    try:
        match ext:
            case '.csv':
                df = pd.read_csv(source)
            case '.xlsx':
                df = pd.read_excel(source)
            case '.json':
                df = pd.read_json(source)
            case _:
                df = None
        return df
    except Exception as e:
        raise Exception(f"Failed to import from source - {e}")

def import_from_local_source(filepath, tp_src=False) -> pd.DataFrame:
    if filepath is None or filepath == '':
        raise Exception("Filepath not provided")
    if not os.path.exists(filepath):
        raise Exception(f"Invalid filepath, {filepath}")
    
    destpath = EXTERNAL_DATA_DIR if tp_src else RAW_DATA_DIR
    filename, ext = os.path.splitext(filepath)[-2:].lower()
    if ext not in accepted_formats:
        raise Exception("Unsupported file type")
    
    df = srcdata_to_df(filepath, ext)

    if df is None:
        raise Exception("Failed to import from local source")
    
    logger.success(f"Data from file, {filepath}, imported")
    store_data(df, filename, destpath)
    return df
        
        
def import_from_ext_source(ext_src):
    if not valid_url_scheme(ext_src):
        raise Exception("Filepath not provided")
        
    url = format_url(ext_src)
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    logger.info(f"File type identified, '{ext}'")

    if ext not in accepted_formats:
        raise Exception("Unsupported file type")
    
    new_filename = gen_import_filename(ext_src)
    destpath = EXTERNAL_DATA_DIR

    df = srcdata_to_df(url, ext)

    if df is None:
        raise Exception('Failed to extract data from the given url')
    
    logger.success(f"Successfully imported the dataset from, {ext_src}")
    store_data(df, new_filename, destpath)
    return df

def store_data(data_df: pd.DataFrame, filename: str, destpath):
    ## Handles situations of duplicate filenames
    appended_num = 0
    corrected_filename = f'{filename}.csv'
    while os.path.exists(os.path.join(destpath, corrected_filename)):
        appended_num += 1
        corrected_filename = f'{filename}-{appended_num}'
    data_df.to_csv(os.path.join(destpath, corrected_filename))

@app.command()
def main(
        local: Annotated[str, typer.Option("--local-path", "-l")] = None,
        external: Annotated[str, typer.Option("--ext-path", "-e")] = None,
        third_party_source: Optional[Annotated[str, typer.Option("--third-party", "-tp")]] = False
    ):

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

if __name__ == "__main__":
    app()
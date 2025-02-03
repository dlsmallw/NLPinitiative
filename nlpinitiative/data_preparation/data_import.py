from typing_extensions import Annotated
from loguru import logger
from urllib.parse import urlparse
import os, shutil, requests, typer

from nlpinitiative.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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

def import_from_local_source(filepath):
    if filepath:
        if not os.path.exists(filepath):
            return "Error", "Invalid filepath"
        else:
            extension = os.path.splitext(filepath)[-1].lower()
            if extension in accepted_formats:
                try:
                    shutil.copy(filepath, RAW_DATA_DIR)
                except Exception as e:
                    return "Error", f"Failed to import from local source - {e}"
                return "Success", f"File {filepath} imported into Raw Data folder, {RAW_DATA_DIR}"
            else:
                return "Error", "Unsupported file type"
    else:
        return "Error", "Filepath not provided"
        
def import_from_ext_source(ext_src):
    if valid_url_scheme(ext_src):
        
        url = format_url(ext_src)
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        logger.info(f"File type identified, '{ext}'")

        if ext in accepted_formats:
            new_filename = gen_import_filename(ext_src)
            output_path = RAW_DATA_DIR / new_filename
            logger.info(f"Output filename, {new_filename}, generated")
            print(RAW_DATA_DIR)

            if not os.path.exists(output_path):
                response = requests.get(url)
                if response.status_code != 200:
                    return "Error", f"Failed to fetch data from {url}"
                content = response.content

                try:
                    with open(output_path, 'wb') as file:
                        file.write(content)
                        file.close()
                    return "Success", f"Successfully imported the dataset from, {ext_src}"
                except Exception as e:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return "Error", f"Failed to write to file, {e}"
            else:
                return "Error", f"File of name {new_filename}, already exists"

@app.command()
def main(
        local: Annotated[str, typer.Option("--local-path", "-l")] = None,
        external: Annotated[str, typer.Option("--ext-path", "-e")] = None
    ):

    if local or external:
        if local:
            logger.info("Import from local source")
            res, msg = import_from_local_source(local)
        else:
            logger.info("Import from external source")
            res, msg = import_from_ext_source(external)

        if res == 'Success':
            logger.success(msg)
        else:
            logger.error(msg) 
    else:
        logger.error("No source specified")
    

if __name__ == "__main__":
    app()
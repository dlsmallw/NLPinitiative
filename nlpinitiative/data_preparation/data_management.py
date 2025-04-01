import os
import pandas as pd
import huggingface_hub as hfh

from pathlib import Path
from loguru import logger
from datasets import DatasetDict
from urllib.parse import urlparse
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from nlpinitiative.data_preparation.data_normalize import DataNormalizer
from nlpinitiative.data_preparation.data_process import DataProcessor

from nlpinitiative.config import (
    DATA_DIR,
    DATASET_REPO,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    NORM_SCHEMA_DIR,
    BINARY_LABELS,
    CATEGORY_LABELS,
    DEF_MODEL,
)

ACCEPTED_FILE_FORMATS = [".csv", ".xlsx", ".json"]


class DataManager:
    """A class for handling data import, normalization and preprocessing/tokenization."""

    def __init__(self):
        """Contructor method for instantiating a DataManager object."""

        self.normalizer = DataNormalizer()
        self.processor = DataProcessor()
        self.rec_mgr = DatasetRecordManager()

    # Data Importing Functionality
    # ===================================================================================================================
    def _is_valid_url(self, url: str) -> bool:
        """Checks that a URL has a valid format.

        Parameters
        ----------
        url : str
            The url to be checked.

        Returns
        -------
        bool
            True if the URL valid or False if not.
        """

        if url:
            parsed_url = urlparse(url)
            return bool(parsed_url.scheme in ["http", "https", "ftp"])
        else:
            return False

    def _generate_import_filename(self, url: str) -> str:
        """Generates a filename that will be used when importing new datasets.

        Parameters
        ----------
        url : str
            The url of the dataset to be imported.

        Returns
        -------
        str
            The generated file name.
        """

        def github():
            """Generates a filename based on a GitHub URL.

            Returns
            -------
            str
                The generated file name.
            """

            parsed = urlparse(url)
            path = os.path.splitext(parsed.path)[0][1:]
            path_arr = path.split("/")
            return "_".join([path_arr[0], path_arr[1], path_arr[-1]])

        if "github" in url:
            return github()
        else:
            split_arr = url.split("/")
            return split_arr[-1]

    def _format_url(self, url: str) -> str:
        """Converts URLs to a format that can be used for importing data from a remote source.

        Parameters
        ----------
        url : str
            The url of the dataset to be imported.

        Returns
        -------
        str
            The reformatted URL.
        """

        def github():
            """Handles conversion of GitHub URLs.

            Returns
            -------
            str
                The reformatted GitHub URL.
            """

            base_url = "https://raw.githubusercontent.com"
            if base_url in url:
                return url
            else:
                updated_url = url.replace("https://github.com", base_url)
                updated_url = updated_url.replace("blob", "refs/heads")
                return updated_url

        if "github" in url:
            logger.info(f"Source url identified as GitHub URL, {url}")
            formatted_url = github()
            logger.info(f"URL Formatted, {formatted_url}")
            return formatted_url
        else:
            return url

    def file_to_df(self, source: str, ext: str) -> pd.DataFrame:
        """Converts a dataset to a dataframe.

        This function also handles conversion of csv files with atypical delimiters.

        Parameters
        ----------
        source : str
            The file path for a local dataset or URL for a remote dataset.
        ext : str
            The file extension of the dataset.

        Returns
        -------
        DataFrame
            The dataset as a Pandas DataFrame.
        """

        try:
            match ext:
                case ".csv":
                    try:
                        df = pd.read_csv(source)
                    except:
                        df = pd.read_csv(source, delimiter=";")
                case ".xlsx":
                    df = pd.read_excel(source)
                case ".json":
                    df = pd.read_json(source)
                case _:
                    df = None
            return df
        except Exception as e:
            err_msg = f"Failed to import from source - {e}"
            logger.error(err_msg)
            raise Exception(err_msg)

    def import_data(
        self,
        import_type: str,
        source,
        dataset_name: str,
        is_third_party: bool = True,
        local_ds_ref_url: bool = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Imports data from a local (by file path) or remote (ny URL) source.

        Parameters
        ----------
        import_type : str
            'local' if from a local source, 'external' if from remote source.
        source : str
            The file path or URL of the dataset.
        dataset_name : str
            The name used to id the dataset.
        is_third_party : bool, optional
            True if 3rd party dataset, False if not (default is True).
        local_ds_ref_url : str, optional
            Reference URL for datasets imported locally (default is None).
        overwrite : bool, optional
            True if files with the same name should be overwritten, False if not (default is False).

        Returns
        -------
        DataFrame
            The dataset as a Pandas DataFrame.
        """

        ref_url = None
        formatted_url = None

        match import_type:
            case "local":
                if source is None or not os.path.exists(source):
                    err_msg = "Dataset filepath does not exist"
                    logger.error(err_msg)
                    raise Exception(err_msg)

                if is_third_party and not self._is_valid_url(local_ds_ref_url):
                    err_msg = (
                        "Locally imported 3rd party dataset imports must include a reference URL."
                    )
                    logger.error(err_msg)
                    raise Exception(err_msg)

                ref_url = local_ds_ref_url if is_third_party else "Custom Created Dataset"

                tail = os.path.split(source)[-1]
                filename, ext = os.path.splitext(tail)[-2:]

                if ext not in ACCEPTED_FILE_FORMATS:
                    err_msg = "Unsupported file type"
                    logger.error(err_msg)
                    raise Exception(err_msg)

                src = source
            case "external":
                if not self._is_valid_url(source):
                    err_msg = "Invalid URL"
                    logger.error(err_msg)
                    raise Exception(err_msg)

                ref_url = source
                formatted_url = self._format_url(ref_url)
                filename = self._generate_import_filename(formatted_url)
                path = urlparse(formatted_url).path
                ext = os.path.splitext(path)[1]

                if ext not in ACCEPTED_FILE_FORMATS:
                    err_msg = "Unsupported file type"
                    logger.error(err_msg)
                    raise Exception(err_msg)

                src = formatted_url
            case _:
                err_msg = "Invalid import type"
                logger.error(err_msg)
                raise Exception(err_msg)

        ds_df = self.file_to_df(src, ext)

        if self.rec_mgr.dataset_src_exists(ref_url):
            return ds_df

        self._store_data(
            data_df=ds_df, filename=filename, destpath=RAW_DATA_DIR, overwrite=overwrite
        )
        self.rec_mgr.update(
            ds_id=dataset_name,
            src_url=ref_url,
            download_url=formatted_url,
            raw_ds_filename=f"{filename}.csv",
        )

        logger.success(f"Successfully imported {import_type} dataset from {source}.")
        return ds_df

    # Data Normalization Functionality
    # ===================================================================================================================

    def _valid_filepath(self, path: Path) -> bool:
        """Checks that the specified file path is valid.

        Parameters
        ----------
        path : Path
            The file path to be checked.

        Returns
        -------
        bool
            True if it exists, False if it does not.
        """

        return os.path.exists(path)

    def normalize_dataset(
        self, ds_files: list[str], conv_schema_fn: str, output_fn: str
    ) -> pd.DataFrame:
        """Handles normalization of a third-party dataset to the schema used for training our models.

        Parameters
        ----------
        ds_files : list[str]
            One or more dataset files to merge and normalize.
        conv_schema_fn : str
            The file name of the json conversion schema used.
        output_fn : str
            The output filename to be used.

        Returns
        -------
        DataFrame
            True if it exists, False if it does not.
        """

        for filename in ds_files:
            if not self._valid_filepath((RAW_DATA_DIR / filename)):
                err_msg = f"Invalid filepath for file, {filename} [{RAW_DATA_DIR / filename}]."
                logger.error(err_msg)
                raise Exception(err_msg)

        if not self._valid_filepath(NORM_SCHEMA_DIR / conv_schema_fn):
            err_msg = f"Normalization schema file, {conv_schema_fn}, does not exist."
            logger.error(err_msg)
            raise Exception(err_msg)

        if output_fn is None or not len(output_fn) > 0:
            err_msg = f"Invalid output filename"
            logger.error(err_msg)
            raise Exception(err_msg)

        try:
            normalized_df = self.normalizer.normalize_datasets(
                files=ds_files, cv_path=NORM_SCHEMA_DIR / conv_schema_fn
            )
        except Exception as e:
            err_msg = f"Failed to normalize file(s) - {e}"
            logger.error(err_msg)
            raise Exception(err_msg)

        self._store_data(normalized_df, output_fn, INTERIM_DATA_DIR)
        for filename in ds_files:
            row_vals = self.rec_mgr.get_entry_by_raw_fn(filename)
            self.rec_mgr.update(
                ds_id=row_vals[0],
                src_url=row_vals[1],
                download_url=row_vals[2],
                raw_ds_filename=row_vals[3],
                normalization_schema_filename=conv_schema_fn,
                normalized_ds_filename=f"{output_fn}.csv",
            )
        logger.success(f"Successfully normalized dataset files [{', '.join(ds_files)}]")
        return normalized_df

    # Master Dataset Creation
    # ===================================================================================================================

    def build_master_dataset(self):
        """Takes the processed datasets, merges them and then stores the resulting dataset in the data/processed directory.

        Facilitates consolidation of all imported and normalized datasets into a single, master dataset and
        stores the master dataset in the data/processed directory.
        """

        master_df = None
        for filename in os.listdir(INTERIM_DATA_DIR):
            if filename != ".gitkeep":
                _, ext = os.path.splitext(filename)
                new_df = self.file_to_df(INTERIM_DATA_DIR / filename, ext)

                if master_df is None:
                    master_df = new_df
                else:
                    master_df = pd.concat([master_df, new_df]).dropna()
        self._store_data(
            data_df=master_df,
            filename="NLPinitiative_Master_Dataset",
            destpath=PROCESSED_DATA_DIR,
            overwrite=True,
        )

    def pull_dataset_repo(self, token: str):
        """Pulls the data directory from the linked Hugging Face Dataset Repository.

        Parameters
        ----------
        token : str
            A Hugging Face token with read/write access privileges to allow importing the data.
        """

        if token is not None:
            hfh.snapshot_download(
                repo_id=DATASET_REPO, repo_type="dataset", local_dir=DATA_DIR, token=token
            )

    def push_dataset_dir(self, token: str):
        """Pushes the data directory (all dataset information) to the linked Hugging Face Dataset Repository.

        Parameters
        ----------
        token : str
            A Hugging Face token with read/write access privileges to allow importing the data.
        """

        if token is not None:
            hfh.upload_folder(
                repo_id=DATASET_REPO, repo_type="dataset", folder_path=DATA_DIR, token=token
            )

    # Data Preparation Functionality
    # ===================================================================================================================

    def prepare_and_preprocess_dataset(
        self,
        filename: str = "NLPinitiative_Master_Dataset.csv",
        srcdir: Path = PROCESSED_DATA_DIR,
        bin_model_type: str = DEF_MODEL,
        ml_model_type: str = DEF_MODEL,
    ):
        """Preprocesses and tokenizes the specified dataset.

        Parameters
        ----------
        filename : str, optional
            The file name of the file to be loaded and processed (default is 'NLPinitiative_Master_Dataset.csv').
        srcdir : Path, optional
            The source directory of the file to be processed (default is data/processed).
        bin_model_type : str, optional
            The binary classification base model type (default is the DEF_MODEL defined in the nlpinitiative/config.py file).
        ml_model_type : str, optional
            The multilabel regression base model type (default is the DEF_MODEL defined in the nlpinitiative/config.py file).

        Returns
        -------
        tuple[DatasetContainer, DatasetContainer]
            Two data container objects consisting of the raw dataset, encoded dataset, metadata and tokenizer for training binary and multilabel models.
        """

        raw_dataset = self.processor.dataset_from_file(filename, srcdir)
        bin_ds, ml_ds = self.processor.bin_ml_dataset_split(raw_dataset)

        bin_ds_metadata = self.processor.get_dataset_metadata(bin_ds)
        bin_tkzr = self.processor.get_tokenizer(bin_model_type)
        bin_encoded_ds = self.processor.preprocess(bin_ds, bin_ds_metadata["labels"], bin_tkzr)
        bin_data_obj = DatasetContainer(bin_ds, bin_encoded_ds, bin_ds_metadata, bin_tkzr)

        ml_ds_metadata = self.processor.get_dataset_metadata(bin_ds)
        ml_tkzr = self.processor.get_tokenizer(ml_model_type)
        ml_encoded_ds = self.processor.preprocess(ml_ds, ml_ds_metadata["labels"], ml_tkzr)
        ml_data_obj = DatasetContainer(ml_ds, ml_encoded_ds, ml_ds_metadata, ml_tkzr)

        return bin_data_obj, ml_data_obj

    # Misc Helper Functions
    # ===================================================================================================================

    def _store_data(
        self, data_df: pd.DataFrame, filename: str, destpath: Path, overwrite: bool = False
    ):
        """Stores the specified DataFrame as a csv dataset within the data directory.

        Parameters
        ----------
        data_df : DataFrame
            The dataset as a Pandas DataFrame object.
        filename : str
            The file name of the data to be stored.
        destpath : Path
            The path that the data is to be stored at.
        overwrite : bool, optional
            True if file with the same name should be overwritten, False if not (default is False).
        """

        ## Handles situations of duplicate filenames
        appended_num = 0
        corrected_filename = f"{filename}.csv"

        while not overwrite and os.path.exists(os.path.join(destpath, corrected_filename)):
            appended_num += 1
            corrected_filename = f"{filename}-{appended_num}.csv"
        data_df.to_csv(
            path_or_buf=os.path.join(destpath, corrected_filename), index=False, mode="w"
        )

    def get_dataset_statistics(self, ds_path: Path) -> dict:
        """Generates statistics for the specified dataset.

        Evaluates a dataset and records various details for use in assesing if the dataset is imbalanced.

        Parameters
        ----------
        ds_path : Path
            The file path to the dataset.

        Returns
        -------
        dict
            A JSON object containing the generated data.
        """

        def get_category_details():
            """Generates statistical data for the category columns of the dataset.

            Evaluations include:
                - The sum of the categories values.
                - The number of rows that a given category has a non-zero value.
                - The number of rows where the category has a value greater than 0.5.
                - The number of rows where the category had the highest value among categories.

            Returns
            -------
            dict
                A dict containing the data for all categories.
            """

            cat_dict = dict()
            for cat in CATEGORY_LABELS:
                cat_dict[cat] = {
                    "total_combined_value": dataset_df[cat].sum(),
                    "num_positive_rows": len(dataset_df.loc[dataset_df[cat] > 0]),
                    "num_gt_threshold": len(dataset_df.loc[dataset_df[cat] >= 0.5]),
                    "num_rows_as_dominant_category": 0,
                }

            for _, row in dataset_df.iterrows():
                dominant_cat = None
                dominant_val = 0
                for cat in CATEGORY_LABELS:
                    cat_val = row[cat]
                    if cat_val > dominant_val:
                        dominant_cat = cat
                        dominant_val = cat_val
                if dominant_cat is not None:
                    cat_dict[dominant_cat]["num_rows_as_dominant_category"] += 1
            return cat_dict

        dataset_df = self.file_to_df(ds_path, ".csv")

        json_obj = {
            "total_num_entries": len(dataset_df),
            "num_positive_discriminatory": len(dataset_df.loc[dataset_df[BINARY_LABELS[0]] == 1]),
            "num_negative_discriminatory": len(dataset_df.loc[dataset_df[BINARY_LABELS[0]] == 0]),
            "category_stats": get_category_details(),
        }

        return json_obj

    def remove_file(self, filename: str, path: Path):
        """Removes the specified file.

        Parameters
        ----------
        filename : str
            The name of the file.
        path : Path
            The path to the file.
        """

        if os.path.exists(path / filename):
            try:
                os.remove(path / filename)
                logger.success(f"Successfully removed file, {filename}.")
            except Exception as e:
                logger.error(f"Failed to remove file, {filename} - {e}")
        else:
            logger.info(f"File, {filename}, does not exist.")


class DatasetContainer:
    """A class used for organizing dataset information used within model training."""

    def __init__(
        self,
        dataset: DatasetDict,
        encoded_ds: DatasetDict,
        metadata: dict,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        """Constuctor for instantiating DatasetContainer.

        Parameters
        ----------
        dataset : DatasetDict
            The raw dataset (before tokenization).
        encoded_ds : DatasetDict
            The encoded dataset.
        metadata : dict
            The metadata associated with the dataset.
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
            The tokenizer used for tokenizing the dataset.
        """

        self._raw_dataset = dataset
        self._encoded_dataset = encoded_ds
        self._labels = metadata["labels"]
        self._lbl2idx = metadata["lbl2idx"]
        self._idx2lbl = metadata["idx2lbl"]
        self._tkzr = tokenizer

    @property
    def raw_dataset(self):
        """The unencoded dataset."""

        return self._raw_dataset

    @property
    def encoded_dataset(self):
        """The encoded/tokenized dataset."""

        return self._encoded_dataset

    @property
    def labels(self):
        """The labels used within the dataset."""

        return self._labels

    @property
    def lbl2idx(self):
        """Dict for mapping label indices to the label names."""

        return self._lbl2idx

    @property
    def idx2lbl(self):
        """Dict for mapping label names to the label indices."""

        return self._idx2lbl

    @property
    def tokenizer(self):
        """The tokenizer for the dataset."""

        return self._tkzr


class DatasetRecordManager:
    """A class for managing and maintaining a record of the datasets imported and used for model training."""

    def __init__(self):
        """Constructor for instantiating a DatasetRecordManager object."""
        self.rec_df: pd.DataFrame = self.load_dataset_record()

    def load_dataset_record(self) -> pd.DataFrame:
        """Loads the dataset record into a Pandas Dataframe.

        Returns
        -------
        DataFrame
            The record of datasets as a DataFrame.
        """

        if os.path.exists(DATA_DIR / "dataset_record.csv"):
            ds_rec_df = pd.read_csv(filepath_or_buffer=DATA_DIR / "dataset_record.csv")
        else:
            ds_rec_df = pd.DataFrame(
                columns=[
                    "Dataset ID",
                    "Dataset Reference URL",
                    "Dataset Download URL",
                    "Raw Dataset Filename",
                    "Conversion Schema Filename",
                    "Converted Filename",
                ]
            )
            ds_rec_df.to_csv(DATA_DIR / "dataset_record.csv", index=False)
        return ds_rec_df

    def save_ds_record(self):
        """Faciliates saving the current state of the dataset record as a csv file."""

        self.rec_df.to_csv(DATA_DIR / "dataset_record.csv", index=False)

    def dataset_src_exists(self, src_url: str) -> bool:
        """Checks if a dataset record exists with the specified URL.

        Parameters
        ----------
        src_url : str
            The source/reference URL of the dataset.

        Returns
        -------
        bool
            True if the record exists, False otherwise.
        """

        return len(self.rec_df[self.rec_df["Dataset Reference URL"] == src_url]) > 0

    def get_ds_record_copy(self) -> pd.DataFrame:
        """Returns a copy of the dataset record dataframe

        Returns
        -------
        DataFrame
            A copy of the record as a Pandas DataFrame.
        """

        return self.rec_df.copy(deep=True)

    def update(
        self,
        ds_id: str,
        src_url: str = None,
        download_url: str = None,
        raw_ds_filename: str = None,
        normalization_schema_filename: str = None,
        normalized_ds_filename: str = None,
    ):
        """Adds a new dataset record or updates an existing record.

        Parameters
        ----------
        ds_id : str
            The name/id of the dataset (used for tracking purposes).
        src_url : str, optional
            The source or reference URL for an imported dataset (default is None).
        download_url : str, optional
            The download URL of the dataset (default is None).
        raw_ds_filename : str, optional
            The file name of the raw (non-normalized) dataset (default is None).
        normalization_schema_filename : str, optional
            The file name of the conversion schema used to normalize the dataset (default is None).
        normalized_ds_filename : str, optional
            The file name of the normalized version of the dataset (default is None).
        """

        if ds_id is not None:

            new_df = pd.DataFrame(
                {
                    "Dataset ID": [ds_id],
                    "Dataset Reference URL": [src_url],
                    "Dataset Download URL": [download_url],
                    "Raw Dataset Filename": [raw_ds_filename],
                    "Conversion Schema Filename": [normalization_schema_filename],
                    "Converted Filename": [normalized_ds_filename],
                }
            )

            temp_df = pd.concat([self.rec_df, new_df]).drop_duplicates(
                subset=["Dataset ID", "Dataset Reference URL"], keep="last"
            )

            self.rec_df = temp_df
            self.save_ds_record()

    def remove_entry(self, ds_id: str):
        """Facilitates removal of a dataset record.

        Parameters
        ----------
        ds_id : str
            The name/id of the dataset to be removed.
        """

        if ds_id is not None:
            self.rec_df = self.rec_df[self.rec_df["Dataset ID"] != ds_id]
            self.save_ds_record()

    def get_entry_by_raw_fn(self, filename: str) -> tuple:
        """Retrieves the raw file name of the specified dataset record.

        Parameters
        ----------
        filename : str
            The file name of the normalized version of the dataset.

        Raises
        ------
        Exception
            If the method failed to retrieve the specified record.

        Returns
        -------
        str
            The file name of the non-normalized version of the dataset.
        """

        try:
            row = tuple(self.rec_df[self.rec_df["Raw Dataset Filename"] == filename].values[0])
            return row
        except Exception as e:
            err_msg = f"Failed to retrieve row of Dataset Record - {e}"
            logger.error(err_msg)
            raise Exception(err_msg)

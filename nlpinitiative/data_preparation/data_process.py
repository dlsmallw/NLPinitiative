"""
Script file used for facillitating dataset preparation and preprocessing
for use in model training.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset

from nlpinitiative.config import (
    PROCESSED_DATA_DIR,
    DEF_MODEL,
    DATASET_COLS,
    BINARY_LABELS,
    CATEGORY_LABELS,
    TRAIN_TEST_SPLIT,
)


class DataProcessor:
    """A class used for performing preprocessing/tokenization."""

    def dataset_from_file(self, filename: str, srcdir: Path = PROCESSED_DATA_DIR) -> DatasetDict:
        """Loads a dataset from a specified file into a Dataset object.

        Parameters
        ----------
        filename : str
            The file name of the dataset to be loaded.
        srcdir : Path, optional
            The file path to the directory that the dataset is stored (default is data/processed).

        Raises
        ------
        Exception
            If the file specified does not exist.

        Returns
        -------
        DatasetDict
            The loaded dataset as a DatasetDict object.
        """

        if filename and os.path.exists(os.path.join(srcdir, filename)):
            ext = os.path.splitext(filename)[-1]
            ext = ext.replace(".", "")
            ds = load_dataset(
                ext, data_files=os.path.join(srcdir, filename), split="train"
            ).train_test_split(test_size=TRAIN_TEST_SPLIT)
            return ds
        else:
            raise Exception("Invalid file name or file path")

    def bin_ml_dataset_split(self, dataset: Dataset) -> tuple[DatasetDict, DatasetDict]:
        """Forms train/test split for use in the binary classification and multilabel regression models training.

        Takes a loaded dataset and splits it into two separate datasets with the required corresponding
        labels (the binary model dataset will only have the DISCRIMINATORY column, while the multilabel
        regression model will only have the columns consisting of the discrimination categories). Additionally,
        the function will split the prepared datasets into a train/test split.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to be processed.

        Returns
        -------
        DatasetDict
            The prepared binary model and multilabel model dataset objects.
        """

        def get_bin_ds() -> DatasetDict:
            """Prepares the binary model dataset.

            Returns
            -------
            DatasetDict
                The prepared binary model dataset.
            """

            train = dataset["train"].remove_columns(CATEGORY_LABELS)
            test = dataset["test"].remove_columns(CATEGORY_LABELS)

            return DatasetDict(
                {
                    "train": train.rename_column("DISCRIMINATORY", "label"),
                    "test": test.rename_column("DISCRIMINATORY", "label"),
                }
            )

        def get_ml_regr_ds() -> DatasetDict:
            """Prepares the multilabel model dataset.

            Returns
            -------
            DatasetDict
                The prepared multilabel model dataset.
            """

            def combine_labels(ex_ds):
                """Consolidates the mutiple dataset columns for categories into a single
                column consisting of a list of the category data.

                Parameters
                ----------
                ex_ds : DatasetDict
                    The multilabel dataset to be corrected.

                Returns
                -------
                DatasetDict
                    The corrected dataset.
                """

                ex_ds["labels"] = [
                    float(ex_ds["GENDER"]),
                    float(ex_ds["RACE"]),
                    float(ex_ds["SEXUALITY"]),
                    float(ex_ds["DISABILITY"]),
                    float(ex_ds["RELIGION"]),
                    float(ex_ds["UNSPECIFIED"]),
                ]
                return ex_ds

            train = dataset["train"].remove_columns(BINARY_LABELS)
            train = train.map(combine_labels)
            train = train.remove_columns(CATEGORY_LABELS)
            test = dataset["test"].remove_columns(BINARY_LABELS)
            test = test.map(combine_labels)
            test = test.remove_columns(CATEGORY_LABELS)

            return DatasetDict({"train": train, "test": test})

        return get_bin_ds(), get_ml_regr_ds()

    def get_tokenizer(self, model_type: str = DEF_MODEL):
        """Generates a tokenizer for the specified model type.

        Parameters
        ----------
        model_type : str, optional
            The model type for which the tokenizer is to be created.

        Returns
        -------
        PreTrainedTokenizer | PreTrainedTokenizerFast
            The tokenizer object.
        """

        return AutoTokenizer.from_pretrained(model_type)

    def get_dataset_metadata(self, dataset: DatasetDict) -> dict:
        """Gathers metadata for the given dataset into a dict for use in model training.

        Extracts dataset metadata to include a list of the datasets labels, a dict mapping
        the labels to their respective indices and a dict mapping the indices to the respective label.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to retrieve metadata from.

        Returns
        -------
        dict
            The metadata for the dataset within a dict object.
        """

        lbls = [
            label for label in dataset["train"].features.keys() if label not in [DATASET_COLS[0]]
        ]
        lbl2idx = {lbl: idx for idx, lbl in enumerate(lbls)}
        idx2lbl = {idx: lbl for idx, lbl in enumerate(lbls)}

        return {"labels": lbls, "lbl2idx": lbl2idx, "idx2lbl": idx2lbl}

    def preprocess(self, dataset: DatasetDict, labels: list[str], tokenizer) -> DatasetDict:
        """Preprocesses and tokenizes a given dataset.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to to be preprocessed/tokenized.
        labels : list[str]
            The datasets labels.
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
            The tokenizer used for tokenizing the dataset.

        Returns
        -------
        DatasetDict
            The preprocessed dataset.
        """

        def preprocess_runner(data: DatasetDict):
            """Runner for performing tokenization.

            Parameters
            ----------
            data : DatasetDict
                The dataset to to be tokenized.

            Returns
            -------
            DatasetDict
                The tokenized dataset.
            """

            return tokenizer(
                data[DATASET_COLS[0]], padding="max_length", truncation=True, max_length=128
            )

        if not labels:
            labels = [
                label
                for label in dataset["train"].features.keys()
                if label not in [DATASET_COLS[0]]
            ]
        if not tokenizer:
            tokenizer = self.get_tokenizer()

        encoded_ds = dataset.map(preprocess_runner, batched=True)
        encoded_ds.set_format("torch")
        return encoded_ds

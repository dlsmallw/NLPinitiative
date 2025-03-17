"""
Script file used for facillitating dataset preparation and preprocessing
for use in model training.
"""

from pathlib import Path
import os

from datasets import (
    Dataset,
    DatasetDict,
    load_dataset
)

from transformers import (
    AutoTokenizer
)

from nlpinitiative.config import (
    PROCESSED_DATA_DIR, 
    DEF_MODEL,
    DATASET_COLS,
    BINARY_LABELS,
    CATEGORY_LABELS,
    TRAIN_TEST_SPLIT
)

class DataProcessor:
    ## Loads a dataset from a specified file into a Dataset object
    def dataset_from_file(self, filename: str, srcdir: Path = PROCESSED_DATA_DIR):
        if filename and os.path.exists(os.path.join(srcdir, filename)):
            ext = os.path.splitext(filename)[-1]
            ext = ext.replace('.', '')
            ds = load_dataset(ext, data_files=os.path.join(srcdir, filename), split='train').train_test_split(test_size=TRAIN_TEST_SPLIT)
            return ds
        else:
            raise Exception('Invalid file name or file path')
        
    ## Separates a dataset into a Training and Testing (evaluation) dataset pair and further
    ## also handles formatting the datasets into a format that can be used for training the 
    ## binary classification and multilabel regression models
    def bin_ml_dataset_split(self, dataset: Dataset):
        def get_bin_ds():
            train = dataset['train'].remove_columns(CATEGORY_LABELS)
            test = dataset['test'].remove_columns(CATEGORY_LABELS)

            return DatasetDict({
                'train': train.rename_column("DISCRIMINATORY", "label"),
                'test': test.rename_column("DISCRIMINATORY", "label")
            })
        
        def get_ml_regr_ds():
            def combine_labels(ex_ds):
                ex_ds['labels'] = [
                    float(ex_ds["GENDER"]),
                    float(ex_ds["RACE"]),
                    float(ex_ds["SEXUALITY"]),
                    float(ex_ds["DISABILITY"]),
                    float(ex_ds["RELIGION"]),
                    float(ex_ds["UNSPECIFIED"]),
                ]
                return ex_ds

            train = dataset['train'].remove_columns(BINARY_LABELS)
            train = train.map(combine_labels)
            train = train.remove_columns(CATEGORY_LABELS)
            test = dataset['test'].remove_columns(BINARY_LABELS)
            test = test.map(combine_labels)
            test = test.remove_columns(CATEGORY_LABELS)

            return DatasetDict({
                'train': train,
                'test': test
            })
        return get_bin_ds(), get_ml_regr_ds()
    
    ## Initializes a tokenizer object for use in preprocessing the data
    def get_tokenizer(self, model_type=DEF_MODEL):
        return AutoTokenizer.from_pretrained(model_type)

    ## Generates dicts for easily fetching label based on id or id based on lbl
    def get_dataset_metadata(self, dataset: Dataset):
        lbls = [label for label in dataset["train"].features.keys() if label not in [DATASET_COLS[0]]]
        lbl2idx = {lbl:idx for idx, lbl in enumerate(lbls)}
        idx2lbl = {idx:lbl for idx, lbl in enumerate(lbls)}

        return {
            'labels': lbls,
            'lbl2idx': lbl2idx,
            'idx2lbl': idx2lbl
        }

    ## Handles the process of preprocessing the textual data into a format that can be used in the model training
    def preprocess(self, dataset, labels, tokenizer):
        def preprocess_runner(data):
            return tokenizer(data[DATASET_COLS[0]], padding='max_length', truncation=True, max_length=128)
        
        if not labels:
            labels = [label for label in dataset["train"].features.keys() if label not in [DATASET_COLS[0]]]
        if not tokenizer:
            tokenizer = self.get_tokenizer()

        encoded_ds = dataset.map(preprocess_runner, batched=True)
        encoded_ds.set_format("torch")
        return encoded_ds
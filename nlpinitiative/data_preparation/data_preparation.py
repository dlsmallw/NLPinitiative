from typing_extensions import Annotated, Optional
from loguru import logger
from urllib.parse import urlparse
import os, shutil, requests, typer
from pathlib import Path
import pandas as pd

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import (
    Dataset, 
    DatasetDict,
    load_dataset
)

from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer
)

from nlpinitiative.config import (
    PROCESSED_DATA_DIR, 
    INTERIM_DATA_DIR, 
    TOKENIZERS_DIR,
    GENERATOR_BATCH_SIZE, 
    SPECIAL_TOKENS,
    DEF_MODEL
)

app = typer.Typer()

def is_valid_dir(dirpath: Path):
    if os.path.isdir(dirpath) and len(os.listdir(dirpath)) > 0:
        for child_path in os.listdir(dirpath):
            if not os.path.isfile(os.path.join(dirpath, child_path)):
                return False
        return True
    return False

def custom_tokenizer(tknzr_training_dataset: Dataset):
    def dataset_to_training_corpus(dataset: Dataset):
        for index in range(0, len(dataset), GENERATOR_BATCH_SIZE):
            yield dataset[index : index + GENERATOR_BATCH_SIZE]["text"]

    tokenizer = Tokenizer(models.WordPiece(unk_token=SPECIAL_TOKENS[0]))
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True, 
        lowercase=True,
        strip_accents=True
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(dataset_to_training_corpus(tknzr_training_dataset), trainer=trainer)

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    idx = 0
    adjusted_fn = 'cust_tokenizer.json'
    while os.path.exists(os.path.join(TOKENIZERS_DIR, adjusted_fn)):
        idx += 1
        adjusted_fn = f'cust_tokenizer-{idx}.json'

    tokenizer.save(f"/nlpinitiative/data_preparation/tokenizers/{adjusted_fn}")

def get_dataset_from_file(filename: str, srcdir: Path = INTERIM_DATA_DIR):
    if filename and os.path.exists(os.path.join(srcdir, filename)):
        filename, ext = os.path.splitext(filename)
        ext = ext.replace('.', '')
        return load_dataset(ext, data_files=os.path.join(srcdir, filename))
    else:
        raise Exception('Invalid file name or file path')
    
def get_tokenizer(cust_filename: str = None):
    if cust_filename:
        try:
            cust_tkzr = Tokenizer.from_file(os.path.join(TOKENIZERS_DIR, cust_filename))
            tkzr = PreTrainedTokenizerFast(
                tokenizer_object=cust_tkzr,
                unk_token=SPECIAL_TOKENS[0], 
                pad_token=SPECIAL_TOKENS[1], 
                cls_token=SPECIAL_TOKENS[2], 
                sep_token=SPECIAL_TOKENS[3], 
                mask_token=SPECIAL_TOKENS[4]
            )
        except:
            tkzr = AutoTokenizer.from_pretrained(DEF_MODEL)
    else:
        tkzr = AutoTokenizer.from_pretrained(DEF_MODEL)
    return tkzr

    
    

@app.command()
def main(
        src_name: Annotated[str, typer.Option("--data-src", "-d")]
    ):
    res, msg = None
    dataset_path = INTERIM_DATA_DIR / src_name

    if src_name and len(src_name) > 0:
        if res == 'Success':
            logger.success(msg)
        else:
            logger.error(msg) 
    else:
        logger.error("No source specified")
    

if __name__ == "__main__":
    app()
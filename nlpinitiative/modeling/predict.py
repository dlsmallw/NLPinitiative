from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from nlpinitiative.config import (
    DEF_MODEL
)

import torch

from nlpinitiative.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

class InferenceHandler:
    def __init__(self, bin_model_path: Path, ml_regr_model_path: Path):
        # Two separate tokenizers in case we utilize different base models
        # self.bin_tokenizer = AutoTokenizer.from_pretrained(bin_model_path, model_type='bert')
        # self.ml_regr_tokenizer = AutoTokenizer.from_pretrained(ml_regr_model_path, model_type='bert')

        self.bin_tokenizer = AutoTokenizer.from_pretrained(DEF_MODEL)
        self.ml_regr_tokenizer = AutoTokenizer.from_pretrained(DEF_MODEL)

        self.bin_model = AutoModelForSequenceClassification.from_pretrained(bin_model_path, model_type='bert')
        self.ml_regr_model = AutoModelForSequenceClassification.from_pretrained(ml_regr_model_path, model_type='bert')

        self.bin_model.eval()
        self.ml_regr_model.eval()

    def encode_input(self, text):
        bin_tokenized_input = self.bin_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        ml_tokenized_input = self.ml_regr_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return bin_tokenized_input, ml_tokenized_input
    
    def discriminatory_inference(self, text):
        bin_inputs, ml_inputs = self.encode_input(text)

        with torch.no_grad():
            bin_logits = self.bin_model(**bin_inputs).logits

        probs = torch.nn.functional.softmax(bin_logits, dim=-1)
        pred_class = torch.argmax(probs).item()
        bin_label_map = {0: "Non-Discriminatory", 1: "Discriminatory"}
        bin_text_pred = bin_label_map[pred_class]

        res_obj = {
            'raw_text': text,
            'overall_sentiment': bin_text_pred,
            'numerical_value': pred_class,
            'category_sentiments': {
                'Gender': None,
                'Race': None,
                'Sexuality': None,  
                'Disability': None,
                'Religion': None,  
                'Unspecified': None
            }
        }

        if pred_class == 1:
            with torch.no_grad():
                ml_outputs = self.ml_regr_model(**ml_inputs).logits
            
            ml_op_list = ml_outputs.squeeze().tolist()

            idx = 0
            for item in res_obj['category_sentiments'].keys():
                res_obj['category_sentiments'][item] = max(0.0, ml_op_list[idx])
                idx += 1
        
        return res_obj



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

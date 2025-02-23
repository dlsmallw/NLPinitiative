"""
Script file used for performing inference with an existing model.
"""

from pathlib import Path
import typer
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from nlpinitiative.config import (
    DEF_MODEL
)

app = typer.Typer()

## Class used to encapsulate and handle the logic for inference
class InferenceHandler:
    def __init__(self, bin_model_path: Path, ml_regr_model_path: Path):
        # Two separate tokenizers in case we utilize different base models
        self.bin_tokenizer = AutoTokenizer.from_pretrained(DEF_MODEL)
        self.ml_regr_tokenizer = AutoTokenizer.from_pretrained(DEF_MODEL)

        self.bin_model = AutoModelForSequenceClassification.from_pretrained(bin_model_path, model_type='bert')
        self.ml_regr_model = AutoModelForSequenceClassification.from_pretrained(ml_regr_model_path, model_type='bert')

        self.bin_model.eval()
        self.ml_regr_model.eval()

    ## Encodes the textual data to a format the model can process (to integer ids corresponding to each token/word)
    def encode_input(self, text):
        bin_tokenized_input = self.bin_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        ml_tokenized_input = self.ml_regr_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return bin_tokenized_input, ml_tokenized_input
    
    ## Handles logic for checking the binary classfication of the text and 
    ## performs the multilabel regression classification if the text is found
    ## to be discriminatory
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
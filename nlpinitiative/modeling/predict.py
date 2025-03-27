"""
Script file used for performing inference with an existing model.
"""

import torch
import json
import nltk
from nltk import sent_tokenize
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from nlpinitiative.config import MODELS_DIR


class InferenceHandler:
    """A class that handles performing inference using the trained binary classification and multilabel regression models."""

    def __init__(
        self,
        bin_model_path: Path = MODELS_DIR / "binary_classification/best_model",
        ml_regr_model_path: Path = MODELS_DIR / "multilabel_regression/best_model",
    ):
        """Constructor for instantiating an InferenceHandler object.

        Parameters
        ----------
        bin_model_path : Path, optional
            Directory path to the binary model tensor file (default is models/binary_classification/best_model).
        ml_regr_model_path : Path, optional
            Directory path to the multilabel regression model tensor file (default is models/multilabel_regression/best_model).
        """

        self.bin_tokenizer, self.bin_model = self.init_model_and_tokenizer(bin_model_path)
        self.ml_regr_tokenizer, self.ml_regr_model = self.init_model_and_tokenizer(
            ml_regr_model_path
        )
        

    def init_model_and_tokenizer(self, model_path: Path):
        """Initializes a model and tokenizer for use in inference using the models path.

        Parameters
        ----------
        model_path : Path
            Directory path to the models tensor file.

        Returns
        -------
        tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, PreTrainedModel]
            A tuple containing the tokenizer and model objects.
        """

        with open(model_path / "config.json") as config_file:
            config_json = json.load(config_file)
        model_name = config_json["_name_or_path"]
        model_type = config_json["model_type"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, model_type=model_type
        )
        model.eval()

        return tokenizer, model

    def encode_binary(self, text: str):
        """Preprocesses and tokenizes the input text for binary classification.

        Parameters
        ----------
        text : str
            The input text to be preprocessed and tokenized.

        Returns
        -------
        BatchEncoding
            The preprocessed and tokenized input text.
        """

        bin_tokenized_input = self.bin_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        return bin_tokenized_input

    def encode_multilabel(self, text: str):
        """Preprocesses and tokenizes the input text for multilabel regression.

        Parameters
        ----------
        text : str
            The input text to be preprocessed and tokenized.

        Returns
        -------
        BatchEncoding
            The preprocessed and tokenized input text.
        """

        ml_tokenized_input = self.ml_regr_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        return ml_tokenized_input

    def encode_input(self, text: str):
        """Preprocesses and tokenizes the input text sentiment classification (both models).

        Parameters
        ----------
        text : str
            The input text to be preprocessed and tokenized.

        Returns
        -------
        tuple[BatchEncoding, BatchEncoding]
            A tuple containing preprocessed and tokenized input text for both the binary and multilabel regression models.
        """

        bin_inputs = self.encode_binary(text)
        ml_inputs = self.encode_multilabel(text)
        return bin_inputs, ml_inputs

    def classify_text(self, text: str):
        """Performs inference on the input text to determine the binary classification and the multilabel regression for the categories.

        Determines whether the text is discriminatory. If it is discriminatory, it will then perform regression on the input text to determine the
        assesed percentage that each category applies.

        Parameters
        ----------
        text : str
            The input text to be classified.

        Returns
        -------
        dict[str, Any]
            The resulting classification and regression values for each category.
        """

        result = {
            'text_input': text,
            'results': []
        }

        sent_res_arr = []
        sentences = sent_tokenize(text)
        for sent in sentences:
            text_prediction, pred_class = self.discriminatory_inference(sent)

            sent_result = {
                'sentence': sent,
                'binary_classification': {
                    'classification': text_prediction,
                    'prediction_class': pred_class
                },
                'multilabel_regression': None
            }

            if pred_class == 1:
                ml_results = {
                    "Gender": None,
                    "Race": None,
                    "Sexuality": None,
                    "Disability": None,
                    "Religion": None,
                    "Unspecified": None
                }

                ml_infer_results = self.category_inference(sent)
                for idx, key in enumerate(ml_results.keys()):
                    ml_results[key] = min(max(ml_infer_results[idx], 0.0), 1.0)

                sent_result['multilabel_regression'] = ml_results
            sent_res_arr.append(sent_result)

        result['results'] = sent_res_arr
        return result

    def discriminatory_inference(self, text: str):
        """Performs inference on the input text to determine the binary classification.

        Parameters
        ----------
        text : str
            The input text to be classified.

        Returns
        -------
        tuple[str, Number]
            A tuple consisting of the string classification (Discriminatory or Non-Discriminatory) and the numeric prediction class (1 or 0).
        """

        bin_inputs = self.encode_binary(text)

        with torch.no_grad():
            bin_logits = self.bin_model(**bin_inputs).logits

        probs = torch.nn.functional.softmax(bin_logits, dim=-1)
        pred_class = torch.argmax(probs).item()
        bin_label_map = {0: "Non-Discriminatory", 1: "Discriminatory"}
        bin_text_pred = bin_label_map[pred_class]

        return bin_text_pred, pred_class

    def category_inference(self, text: str):
        """Performs inference on the input text to determine the regression values for the categories of discrimination.

        Parameters
        ----------
        text : str
            The input text to be classified.

        Returns
        -------
        list[float]
            A tuple consisting of the string classification (Discriminatory or Non-Discriminatory) and the numeric prediction class (1 or 0).
        """

        ml_inputs = self.encode_multilabel(text)

        with torch.no_grad():
            ml_outputs = self.ml_regr_model(**ml_inputs).logits

        ml_op_list = ml_outputs.squeeze().tolist()

        results = []
        for item in ml_op_list:
            results.append(min(1.0, max(0.0, item)))

        return results

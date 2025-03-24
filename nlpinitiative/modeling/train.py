"""
Script file containing the logic for training NLP models.
"""

from pathlib import Path
from scipy.stats import pearsonr
import numpy as np
import torch
import json

import huggingface_hub as hfh

from nlpinitiative.config import (
    MODELS_DIR, 
    DEF_MODEL,
    CATEGORY_LABELS,
    BIN_REPO,
    ML_REPO
)

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

from transformers import (
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)

# Class for the regression model with a custom compute_loss method due to issues with the base class failing to properly 
# compute loss for a regression model
class RegressionTrainer(Trainer):
    """A custom class for overriding the compute_loss method used in regression model training."""

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Overridden comput_loss method necessary for training the multilabel regression model.

        Parameters
        ----------
        model
            The model being trained.
        inputs
            The dataset being used to train the model.
        return_outputs : bool, optional
            True if the outputs should be returned with the losses, False otherwise (default is False).
        **kwargs
            Any other additional parameters/configurations to use.

        Returns
        -------
        tuple[Any, Any] | Any
            A tuple consisting of the loss values and the corresponding outputs (if return_outputs is True), or just the loss values (if return_outputs is False).
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure labels and logits are float
        labels = labels.to(torch.float32)
        logits = logits.to(torch.float32)

        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Function for computing metrics for evaluating binary classification training
def compute_bin_metrics(eval_predictions):
    """Computes the metrics values resulting from the evaluation of the trained binary classification model.
    
    Parameters
    ----------
    eval_predictions
        A tuple consisting of the label and prediction pair.

    Returns
    -------
    dict[str, Any]
        A dict consisting of the computed metrics values from evaluation of the trained binary classification model.
    """
    predictions, lbls = eval_predictions
    preds = predictions.argmax(axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()

    prec, recall, f1, _ = precision_recall_fscore_support(lbls, preds, average='binary')
    acc = accuracy_score(lbls, preds)

    auprc = average_precision_score(lbls, probs[:, 0])
    auroc = roc_auc_score(lbls, probs[:, 0])

    return {
        "accuracy": acc,
        "precision": prec, 
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc
    }

# Function for computing metrics for evaluating regression training
def compute_reg_metrics(eval_predictions):
    """Computes the metrics values resulting from the evaluation of the trained multilabel regression model.
    
    Parameters
    ----------
    eval_predictions
        A tuple consisting of the labels corresponding prediction pair.

    Returns
    -------
    dict[str, float]
        A dict consisting of the computed metrics values from evaluation of the trained multilabel regression model.
    """
    preds, lbls = eval_predictions

    mse = mean_squared_error(lbls, preds, multioutput='raw_values')
    sqrt_mse = np.sqrt(mse)
    mae = mean_absolute_error(lbls, preds, multioutput='raw_values')
    r2 = r2_score(lbls, preds, multioutput='raw_values')

    pear_corr = [
        pearsonr(lbls[:, i], preds[:, i])[0] if len(np.unique(lbls[:, i])) > 1 else np.nan
        for i in range(lbls.shape[1])
    ]

    mean_rmse = sqrt_mse.mean()
    mean_mae = mae.mean()
    mean_r2 = r2.mean()
    mean_pear = np.nanmean(pear_corr)

    return {
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae,
        'mean_r2': mean_r2,
        'mean_pearson': mean_pear,
        ## Per category
        'rmse_per_cat': sqrt_mse.tolist(),
        'mae_per_cat': mae.tolist(),
        'r2_per_cat': r2.tolist(),
        'pearson_per_cat': pear_corr
    }

# Base function for generating training arguments
def _train_args(
    output_dir: Path,
    eval_strat: str,
    save_strat: str,
    logging_steps: int,
    save_steps: int,
    learn_rate: float,
    batch_sz: int,
    num_train_epochs: int,
    weight_decay: float,
    best_model_at_end: bool,
    best_model_metric: str,
    greater_better: bool
):
    """Generates training arguments for use in model training.
    
    Parameters
    ----------
    output_dir : Path
        The output directory to store the trained model.
    eval_strat : str
        The evaluation strategy (i.e., steps/epochs).
    save_strat : str
        The save strategy (i.e., steps/epochs).
    logging_steps : int
        The periodicity for which logging will occur.
    save_steps : int
        The step periodicity for which a model will be saved.
    learn_rate : float
        A hyper parameter for determining model parameter adjustment during training.
    batch_sz : int
        The training data batch sizes.
    num_train_epochs : int
        The number of training epochs to be performed.
    weight_decay : float
        The weight decay to apply.
    best_model_at_end : bool
        True if the best model is to be saved at the completion of model training, False otherwise.
    best_model_metric : str
        The metric used for determining the best performing model.
    greater_better : bool
        True for if the better performing model should have the greater value (based on the specified metric), False otherwise.

    Returns
    -------
    TrainingArguments
        The training arguments object used for conducting model training.
    """

    return TrainingArguments(
        output_dir=output_dir, 
        eval_strategy=eval_strat, 
        save_strategy=save_strat,
        logging_steps=logging_steps,
        save_steps=save_steps,
        learning_rate=learn_rate,
        per_device_train_batch_size=batch_sz,
        per_device_eval_batch_size=batch_sz,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=best_model_at_end,
        metric_for_best_model=best_model_metric,
        greater_is_better=greater_better,
        overwrite_output_dir=True,
        save_on_each_node=True,
        save_total_limit=5
    )

# Generates the training arguments used for training and evaluating the binary classification model
def bin_train_args(
        output_dir: Path = MODELS_DIR / 'binary_classification',
        eval_strat: str = 'steps',
        save_strat: str = 'steps',
        logging_steps: int = 500,
        save_steps: int = 500,
        learn_rate: float = 2e-5,
        batch_sz: int = 8,
        num_train_epochs: int = 3,
        weight_decay: float = 0.01,
        best_model_at_end: bool = True,
        best_model_metric: str = 'f1',
        greater_better: bool = True
):
    """Generates training arguments for use in binary classification model training.
    
    Parameters
    ----------
    output_dir : str, optional
        The output directory to store the trained model (default is models/binary_classification).
    eval_strat : str, optional
        The evaluation strategy (default is 'steps').
    save_strat : str, optional
        The save strategy (default is 'steps').
    logging_steps : int, optional
        The periodicity for which logging will occur (default is 500).
    save_steps : int, optional
        The step periodicity for which a model will be saved (default is 500).
    learn_rate : float, optional
        A hyper parameter for determining model parameter adjustment during training (default is 2e-5).
    batch_sz : int, optional
        The training data batch sizes (default is 8).
    num_train_epochs : int, optional
        The number of training epochs to be performed (default is 3).
    weight_decay : float, optional
        The weight decay to apply (default is 0.01).
    best_model_at_end : bool, optional
        True if the best model is to be saved at the completion of model training, False otherwise (default is True).
    best_model_metric : str, optional
        The metric used for determining the best performing model (default is 'f1').
    greater_better : bool, optional
        True for if the better performing model should have the greater value (based on the specified metric), False otherwise (default is True).

    Returns
    -------
    TrainingArguments
        The training arguments object used for conducting binary classification model training.
    """
    
    return _train_args(
        output_dir,
        eval_strat,
        save_strat,
        logging_steps,
        save_steps,
        learn_rate,
        batch_sz,
        num_train_epochs,
        weight_decay,
        best_model_at_end,
        best_model_metric,
        greater_better
    )

# Generates the training arguments used for training and evaluating the multilabel regression model
def ml_regr_train_args(
        output_dir: Path = MODELS_DIR / 'multilabel_regression',
        eval_strat: str = 'steps',
        save_strat: str = 'steps',
        logging_steps: int = 500,
        save_steps: int = 500,
        learn_rate: float = 2e-5,
        batch_sz: int = 8,
        num_train_epochs: int = 3,
        weight_decay: float = 0.01,
        best_model_at_end: bool = True,
        best_model_metric: str = 'eval_mean_rmse',
        greater_better: bool = False
):
    """Generates training arguments for use in multilabel regression model training.
    
    Parameters
    ----------
    output_dir : str, optional
        The output directory to store the trained model (default is models/multilabel_regression).
    eval_strat : str, optional
        The evaluation strategy (default is 'steps').
    save_strat : str, optional
        The save strategy (default is 'steps').
    logging_steps : int, optional
        The periodicity for which logging will occur (default is 500).
    save_steps : int, optional
        The step periodicity for which a model will be saved (default is 500).
    learn_rate : float, optional
        A hyper parameter for determining model parameter adjustment during training (default is 2e-5).
    batch_sz : int, optional
        The training data batch sizes (default is 8).
    num_train_epochs : int, optional
        The number of training epochs to be performed (default is 3).
    weight_decay : float, optional
        The weight decay to apply (default is 0.01).
    best_model_at_end : bool, optional
        True if the best model is to be saved at the completion of model training, False otherwise (default is True).
    best_model_metric : str, optional
        The metric used for determining the best performing model (default is 'eval_mean_rmse').
    greater_better : bool, optional
        True for if the better performing model should have the greater value (based on the specified metric), False otherwise (default is False).

    Returns
    -------
    TrainingArguments
        The training arguments object used for conducting multilabel regression model training.
    """
    
    return _train_args(
        output_dir,
        eval_strat,
        save_strat,
        logging_steps,
        save_steps,
        learn_rate,
        batch_sz,
        num_train_epochs,
        weight_decay,
        best_model_at_end,
        best_model_metric,
        greater_better
    )

# Generates a model object for handling binary classification
def get_bin_model(model_name: str = DEF_MODEL):
    """Generates a model object to be trained for binary classification.
    
    Parameters
    ----------
    model_name: str, optional
        The name of the pretrained model to be fine-tuned (default is the DEF_MODEL specified in nlpinitiative/config.py).

    Returns
    -------
    PreTrainedModel
        The corresponding pretrained subclass model object for binary classification.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

# Generates a model obect for handling multilabel regression
def get_ml_model(model_name: str = DEF_MODEL):
    """Generates a model object to be trained for multilabel regression.
    
    Parameters
    ----------
    model_name: str, optional
        The name of the pretrained model to be fine-tuned (default is the DEF_MODEL specified in nlpinitiative/config.py).

    Returns
    -------
    PreTrainedModel
        The corresponding pretrained subclass model object for multilabel regression.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORY_LABELS)
    )

def train(bin_trainer: Trainer, ml_trainer: Trainer, token: str = None):
    """Performs training on the binary classification and multilabel regression models.
    
    Parameters
    ----------
    bin_trainer : Trainer
        The trainer object for performing binary classification model training.
    ml_trainer : Trainer
        The trainer object for performing multilabel regression model training.
    token : str, optional
            A Hugging Face token with read/write access privileges to allow exporting the trained models (default is None).

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        A tuple consisting of the dicts containing the metrics evaluation results for the binary classification and multilabel regression model training.
    """

    bin_trainer.train()
    bin_model = bin_trainer.model
    bin_model.save_pretrained(save_directory=MODELS_DIR / 'binary_classification' / 'best_model')
    
    ml_trainer.train()
    ml_model = ml_trainer.model
    ml_model.save_pretrained(save_directory=MODELS_DIR / 'multilabel_regression' / 'best_model')
    
    if token:
        upload_best_models(token=token)

    bin_eval = bin_trainer.evaluate()
    ml_eval = ml_trainer.evaluate()
    return bin_eval, ml_eval

def upload_best_models(token: str):
    """Pushes the current best performing binary classification and multilabel regression models to their respective linked Hugging Face Model Repositories.
    
    Parameters
    ----------
    token : str, optional
        A Hugging Face token with read/write access privileges to allow exporting the trained models (default is None).
    """

    def load_model(path: Path):
        """Initializes a PreTrainedModel object from the specified model path.
        
        Parameters
        ----------
        path : Path
            The file path to the model.
            
        Returns
        -------
        PreTrainedModel
            An instantiated PreTrainedModel object. 
        """
        with open(path / 'config.json') as config_file:
            config_json = json.load(config_file)
        model_type = config_json['model_type']
        return AutoModelForSequenceClassification.from_pretrained(path, model_type=model_type)
    
    bin_model = load_model(MODELS_DIR / 'binary_classification/best_model')
    ml_model = load_model(MODELS_DIR / 'multilabel_regression/best_model')

    bin_model.push_to_hub(BIN_REPO, token=token)
    hfh.upload_file(path_or_fileobj=MODELS_DIR / 'binary_classification/best_model/config.json', path_in_repo='config.json', repo_id=BIN_REPO, token=token)
    ml_model.push_to_hub(ML_REPO, token=token)
    hfh.upload_file(path_or_fileobj=MODELS_DIR / 'multilabel_regression/best_model/config.json', path_in_repo='config.json', repo_id=ML_REPO, token=token)

"""This module contains the training logic for the binary classification and multilabel regression models."""

import os
import typer
import validators
import huggingface_hub as hfh


from pathlib import Path
from loguru import logger
from scipy.stats import pearsonr
import numpy as np
import torch
import json

import huggingface_hub as hfh

from nlpinitiative.config import (
    HF_TOKEN,
    MODELS_DIR,
    DEF_MODEL,
    CATEGORY_LABELS,
    BIN_REPO,
    ML_REPO,
    BIN_OUTPUT_DIR,
    ML_OUTPUT_DIR,
)

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

app = typer.Typer()


class RegressionTrainer(Trainer):  # pragma: no cover
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


def compute_bin_metrics(eval_predictions):  # pragma: no cover
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

    prec, recall, f1, _ = precision_recall_fscore_support(lbls, preds, average="binary")
    acc = accuracy_score(lbls, preds)

    auprc = average_precision_score(lbls, probs[:, 0])
    auroc = roc_auc_score(lbls, probs[:, 0])

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
    }


def compute_reg_metrics(eval_predictions):  # pragma: no cover
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

    mse = mean_squared_error(lbls, preds, multioutput="raw_values")
    sqrt_mse = np.sqrt(mse)
    mae = mean_absolute_error(lbls, preds, multioutput="raw_values")
    r2 = r2_score(lbls, preds, multioutput="raw_values")

    pear_corr = [
        pearsonr(lbls[:, i], preds[:, i])[0] if len(np.unique(lbls[:, i])) > 1 else np.nan
        for i in range(lbls.shape[1])
    ]

    mean_rmse = sqrt_mse.mean()
    mean_mae = mae.mean()
    mean_r2 = r2.mean()
    mean_pear = np.nanmean(pear_corr)

    return {
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
        "mean_pearson": mean_pear,
        ## Per category
        "rmse_per_cat": sqrt_mse.tolist(),
        "mae_per_cat": mae.tolist(),
        "r2_per_cat": r2.tolist(),
        "pearson_per_cat": pear_corr,
    }


def bin_train_args(
    output_dir: Path = BIN_OUTPUT_DIR,
    hub_model_id: str = BIN_REPO,
    push_to_hub: bool = True,
    hub_strategy: str = "end",
    hub_token: str = None,
    eval_strat: str = "steps",
    save_strat: str = "steps",
    logging_steps: int = 500,
    save_steps: int = 500,
    learn_rate: float = 2e-5,
    batch_sz: int = 8,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    best_model_at_end: bool = True,
    best_model_metric: str = "f1",
    greater_better: bool = True,
    overwrite_output_dir: bool = True,
    save_on_each_node: bool = True,
    save_total_limit: int = 5,
):
    """Generates training arguments for use in binary classification model training.

    Parameters
    ----------
    output_dir : str, optional
        The output directory to store the trained model (default is models/binary_classification).
    hub_model_id : str, optional
        The Hugging Face model repository ID to push the model to (default is BIN_REPO).
    push_to_hub : bool, optional
        True if the model should be pushed to the Hugging Face Model Hub, False otherwise (default is True).
    hub_strategy : str, optional
        The strategy for pushing the model to the Hugging Face Model Hub (default is 'end').
    hub_token : str, optional
        A Hugging Face token with read/write access privileges to allow exporting the trained model (default is None).
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
    overwrite_output_dir : bool, optional
        True if the output directory should be overwritten, False otherwise (default is True).
    save_on_each_node : bool, optional
        True if the model should be saved on each node, False otherwise (default is True).
    save_total_limit : int, optional
        The maximum number of checkpoints to keep (default is 5).

    Returns
    -------
    TrainingArguments
        The training arguments object used for conducting binary classification model training.
    """

    if push_to_hub:  # pragma: no cover
        if hub_token is None and (HF_TOKEN is None or HF_TOKEN == ""):
            raise ValueError("No token provided. Please provide a valid Hugging Face token.")
        if hub_model_id is None or hub_model_id == "":
            raise ValueError(
                "No model repository specified. Please provide a valid Hugging Face model repository."
            )
    else:  # pragma: no cover
        hub_model_id = None
        hub_strategy = None
        hub_token = None

    return TrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        push_to_hub=push_to_hub,
        hub_strategy=hub_strategy,
        hub_token=hub_token,
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
        overwrite_output_dir=overwrite_output_dir,
        save_on_each_node=save_on_each_node,
        save_total_limit=save_total_limit,
    )


def ml_regr_train_args(
    output_dir: Path = ML_OUTPUT_DIR,
    hub_model_id: str = ML_REPO,
    push_to_hub: bool = True,
    hub_strategy: str = "end",
    hub_token: str = None,
    eval_strat: str = "steps",
    save_strat: str = "steps",
    logging_steps: int = 500,
    save_steps: int = 500,
    learn_rate: float = 2e-5,
    batch_sz: int = 8,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    best_model_at_end: bool = True,
    best_model_metric: str = "eval_mean_rmse",
    greater_better: bool = False,
    overwrite_output_dir: bool = True,
    save_on_each_node: bool = True,
    save_total_limit: int = 5,
):
    """Generates training arguments for use in multilabel regression model training.

    Parameters
    ----------
    output_dir : str, optional
        The output directory to store the trained model (default is models/multilabel_regression).
    hub_model_id : str, optional
        The Hugging Face model repository ID to push the model to (default is BIN_REPO).
    push_to_hub : bool, optional
        True if the model should be pushed to the Hugging Face Model Hub, False otherwise (default is True).
    hub_strategy : str, optional
        The strategy for pushing the model to the Hugging Face Model Hub (default is 'end').
    hub_token : str, optional
        A Hugging Face token with read/write access privileges to allow exporting the trained model (default is None).
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
    overwrite_output_dir : bool, optional
        True if the output directory should be overwritten, False otherwise (default is True).
    save_on_each_node : bool, optional
        True if the model should be saved on each node, False otherwise (default is True).
    save_total_limit : int, optional
        The maximum number of checkpoints to keep (default is 5).

    Returns
    -------
    TrainingArguments
        The training arguments object used for conducting multilabel regression model training.
    """

    if push_to_hub:  # pragma: no cover
        if hub_token is None and (HF_TOKEN is None or HF_TOKEN == ""):
            raise ValueError("No token provided. Please provide a valid Hugging Face token.")
        if hub_model_id is None or hub_model_id == "":
            raise ValueError(
                "No model repository specified. Please provide a valid Hugging Face model repository."
            )
    else:  # pragma: no cover
        hub_model_id = None
        hub_strategy = None
        hub_token = None

    return TrainingArguments(
        output_dir=output_dir,
        hub_model_id=hub_model_id,
        push_to_hub=push_to_hub,
        hub_strategy=hub_strategy,
        hub_token=hub_token,
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
        overwrite_output_dir=overwrite_output_dir,
        save_on_each_node=save_on_each_node,
        save_total_limit=save_total_limit,
    )


def get_bin_model(model_name_or_path: str | Path = DEF_MODEL):
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

    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path)


def get_ml_model(model_name_or_path: str | Path = DEF_MODEL):
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
        model_name_or_path, num_labels=len(CATEGORY_LABELS)
    )


def train_binary_model(bin_trainer: Trainer):  # pragma: no cover
    """Performs training on the binary classification model.

    Parameters
    ----------
    bin_trainer : Trainer
        The trainer object for performing binary classification model training.

    Returns
    -------
    dict[str, float]
        A dict containing the metrics evaluation results for the binary classification model training.
    """
    bin_eval = None

    if bin_trainer is not None:
        bin_trainer.train()
        bin_eval = bin_trainer.evaluate()
        bin_trainer.save_metrics(split="all", metrics=bin_eval)

    return bin_eval


def train_multilabel_model(ml_trainer: Trainer):  # pragma: no cover
    """Performs training on the multilabel regression model.

    Parameters
    ----------
    ml_trainer : Trainer
        The trainer object for performing multilabel regression model training.

    Returns
    -------
    dict[str, float]
        A dict containing the metrics evaluation results for the multilabel regression model training.
    """
    ml_eval = None

    if ml_trainer is not None:
        ml_trainer.train()
        ml_eval = ml_trainer.evaluate()
        ml_trainer.save_metrics(split="all", metrics=ml_eval)

    return ml_eval


def train(bin_trainer: Trainer | None, ml_trainer: Trainer | None):  # pragma: no cover
    """Performs training on the binary classification and multilabel regression models.

    Parameters
    ----------
    bin_trainer : Trainer
        The trainer object for performing binary classification model training.
    ml_trainer : Trainer
        The trainer object for performing multilabel regression model training.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        A tuple consisting of the dicts containing the metrics evaluation results for the binary classification and multilabel regression model training.
    """
    bin_eval = train_binary_model(bin_trainer=bin_trainer)
    ml_eval = train_multilabel_model(ml_trainer=ml_trainer)
    return bin_eval, ml_eval


def upload_best_models(token: str = None):  # pragma: no cover
    """Pushes the current best performing binary classification and multilabel regression models to their respective linked Hugging Face Model Repositories.

    Parameters
    ----------
    token : str, optional
        A Hugging Face token with read/write access privileges to allow exporting the trained models (default is None).
    """

    def load_model(model_path: Path):
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

        return AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if token is None and (HF_TOKEN is None or HF_TOKEN == ""):
        if HF_TOKEN is not None and HF_TOKEN != "":
            token = HF_TOKEN
        else:
            raise ValueError("No token provided. Please provide a valid Hugging Face token.")

    bin_model = load_model(BIN_OUTPUT_DIR)
    ml_model = load_model(ML_OUTPUT_DIR)

    bin_model.push_to_hub(BIN_REPO, token=token)
    ml_model.push_to_hub(ML_REPO, token=token)


def sync_with_model_repos():  # pragma: no cover
    """Synchronizes the model directory with the Hugging Face Model Repositories."""

    if HF_TOKEN is None or HF_TOKEN == "":
        raise ValueError("No token provided. Please provide a valid Hugging Face token.")

    try:
        logger.info(
            "Beginning synchronization with binary model repository. This may take a few minutes..."
        )
        hfh.snapshot_download(
            repo_id=BIN_REPO, repo_type="model", local_dir=BIN_OUTPUT_DIR, token=HF_TOKEN
        )
        logger.success(f"Successfully downloaded binary classification model from {BIN_REPO}.")
    except Exception as e:
        logger.error(f"Error downloading binary classification model: {e}")

    try:
        logger.info(
            "Beginning synchronization with multilabel model repository. This may take a few minutes..."
        )
        hfh.snapshot_download(
            repo_id=ML_REPO, repo_type="model", local_dir=ML_OUTPUT_DIR, token=HF_TOKEN
        )
        logger.success(f"Successfully downloaded multilabel regression model from {ML_REPO}.")
    except Exception as e:
        logger.error(f"Error downloading multilabel regression model: {e}")


@app.command()
def main():  # pragma: no cover
    """Facilitates synchronization with the remote HF model repositories."""

    try:
        sync_with_model_repos()
    except Exception as e:
        logger.error(f"Failed to sync with remote model repositories: {e}")
        raise e


if __name__ == "__main__":  # pragma: no cover
    app()

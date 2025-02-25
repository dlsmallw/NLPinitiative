"""
Script file containing the logic for training NLP models.
"""

from scipy.stats import pearsonr
import numpy as np
import torch
import typer

from nlpinitiative.config import (
    MODELS_DIR, 
    DEF_MODEL,
    CATEGORY_LABELS
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

app = typer.Typer()

# Class for the regression model with a custom compute_loss method due to issues with the base class failing to properly 
# compute loss for a regression model
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
def compute_bin_metrics(eval_predicitions):
    predictions, lbls = eval_predicitions
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
    output_dir,
    eval_strat,
    save_strat,
    learn_rate,
    batch_sz,
    num_train_epochs,
    weight_decay,
    best_model_at_end,
    best_model_metric,
    greater_better):

    return TrainingArguments(
        output_dir, 
        eval_strategy=eval_strat, 
        save_strategy=save_strat,
        logging_steps=10,
        learning_rate=learn_rate,
        per_device_train_batch_size=batch_sz,
        per_device_eval_batch_size=batch_sz,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=best_model_at_end,
        metric_for_best_model=best_model_metric,
        greater_is_better=greater_better
    )

# Generates the training arguments used for training and evaluating the binary classification model
def bin_train_args(
        output_dir = MODELS_DIR / 'binary_classification',
        eval_strat='epoch',
        save_strat='epoch',
        learn_rate=2e-5,
        batch_sz=8,
        num_train_epochs=3,
        weight_decay=0.01,
        best_model_at_end=True,
        best_model_metric='f1',
        greater_better=True):
    
    return _train_args(
        output_dir,
        eval_strat,
        save_strat,
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
        output_dir = MODELS_DIR / 'multilabel_regression',
        eval_strat='epoch',
        save_strat='epoch',
        learn_rate=2e-5,
        batch_sz=8,
        num_train_epochs=3,
        weight_decay=0.01,
        best_model_at_end=True,
        best_model_metric='eval_mean_rmse',
        greater_better=False):
    
    return _train_args(
        output_dir,
        eval_strat,
        save_strat,
        learn_rate,
        batch_sz,
        num_train_epochs,
        weight_decay,
        best_model_at_end,
        best_model_metric,
        greater_better
    )

# Generates a model object for handling binary classification
def get_bin_model(model_name=DEF_MODEL):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

# Generates a model obect for handling multilabel regression
def get_ml_model(model_name=DEF_MODEL):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORY_LABELS)
    )
from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
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

import torch
import numpy as np
from scipy.stats import pearsonr

from nlpinitiative.config import (
    MODELS_DIR, 
    DEF_MODEL,
)
from nlpinitiative.data_preparation import data_preparation

app = typer.Typer()

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

def compute_reg_metrics(eval_predicitions):
    preds, lbls = eval_predicitions

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


# def get_model(num_lbls, id2lbl_dict, lbl2id_dict, model_name_or_path=None, task_type=None):
#     if not model_name_or_path:
#         model_name_or_path = DEF_MODEL

#     return AutoModelForSequenceClassification.from_pretrained(
#         model_name_or_path,
#         problem_type=task_type,
#         num_labels=num_lbls,
#         id2label=id2lbl_dict,
#         label2id=lbl2id_dict
#     )

def get_model(model_name_or_path=None, task_type=None):
    if not model_name_or_path:
        model_name_or_path = DEF_MODEL

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        problem_type=task_type
    )

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

@app.command()
def main(
    model_path: Path = MODELS_DIR / "model.pkl",
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

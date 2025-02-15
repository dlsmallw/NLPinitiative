from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EvalPrediction
)

from sklearn.metrics import (
    accuracy_score,
    f1_score, 
    precision_score,
    roc_auc_score
)

import torch

from nlpinitiative.config import (
    MODELS_DIR, 
    DEF_MODEL,
    PROCESSED_DATA_DIR,
)
from nlpinitiative.data_preparation import data_preparation

app = typer.Typer()

def binary_metrics():
    pass

def multilabel_regression_metrics():
    pass

def training_args(
    output_dir: Path = MODELS_DIR,
    eval_strat='epoch',
    save_strat='epoch',
    learn_rate=2e-5,
    batch_sz=8,
    num_train_epochs=5,
    weight_decay=0.01,
    best_model_at_end=True,
    best_model_metric='f1'):

    return TrainingArguments(
        output_dir, 
        eval_strategy=eval_strat, 
        save_strategy=save_strat,
        learning_rate=learn_rate,
        per_device_train_batch_size=batch_sz,
        per_device_eval_batch_size=batch_sz,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=best_model_at_end,
        metric_for_best_model=best_model_metric
    )


def get_model(num_lbls, id2lbl_dict, lbl2id_dict, model_name_or_path=None, task_type=None):
    if not model_name_or_path:
        model_name_or_path = DEF_MODEL

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        problem_type=task_type,
        num_labels=num_lbls,
        id2label=id2lbl_dict,
        label2id=lbl2id_dict
    )

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

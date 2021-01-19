import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, AdamWeightDecay

from model import get_distilbert
from utils import encode_labels

CLASS_DIST = [.07, .02, .18, .73]
CLASS_WEIGHTS = {i: 1 / (c * 4) for i, c in enumerate(CLASS_DIST)}


def train_test(
    n_epochs: int,
    weight_decay: float,
    use_class_weights: bool,
    learning_rate: float,
    use_scheduler: bool,
    experiment_name: Optional[str]
):
    results_dir = Path(f'results/train-{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}-{experiment_name}')
    results_dir.mkdir(parents=True)

    _save_args(
        None, results_dir, n_epochs, weight_decay, use_class_weights,
        learning_rate, use_scheduler, experiment_name)

    train_data = encode_labels(_load_train_data()).sample(frac=1).reset_index(drop=True)
    test_data = encode_labels(_load_test_data())

    train_texts = train_data[['Headline', 'articleBody']]
    test_texts = test_data[['Headline', 'articleBody']]
    train_labels = train_data['Stance']
    test_labels = test_data['Stance']

    _train(
        16, learning_rate, use_scheduler, n_epochs, weight_decay, CLASS_WEIGHTS if use_class_weights else None,
        results_dir, train_labels, train_texts, test_labels, test_texts, False
    )


def train_val(
    seed: Optional[int],
    n_epochs: int,
    weight_decay: float,
    use_class_weights: bool,
    learning_rate: float,
    use_scheduler: bool,
    experiment_name: Optional[str]
):
    results_dir = Path(f'results/train-val-{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}-{experiment_name}')
    results_dir.mkdir(parents=True)

    _save_args(
        seed, results_dir, n_epochs, weight_decay, use_class_weights,
        learning_rate, use_scheduler, experiment_name)

    train_data = encode_labels(_load_train_data())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data[['Headline', 'articleBody']],
        train_data['Stance'],
        test_size=.2,
        random_state=seed)

    _train(
        16, learning_rate, use_scheduler, n_epochs, weight_decay, CLASS_WEIGHTS if use_class_weights else None,
        results_dir, train_labels, train_texts, val_labels, val_texts, True
    )


def _train(
    batch_size: int,
    learning_rate: float,
    use_scheduler: bool,
    n_epochs: int,
    weight_decay: float,
    class_weights: Optional[Dict[int, float]],
    results_dir: Path,
    train_labels: pd.Series,
    train_texts: pd.DataFrame,
    val_labels: pd.Series,
    val_texts: pd.DataFrame,
    eval_on_val_set: bool = True
):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    logging.info("Encoding input...")
    train_dataset = _get_dataset(tokenizer, train_texts, train_labels)
    val_dataset = _get_dataset(tokenizer, val_texts, val_labels)

    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    model = get_distilbert()
    model.compile(
        optimizer=optimizer,
        loss=model.compute_loss,
        metrics=[
            'accuracy'
        ],
    )

    callbacks = [tf.keras.callbacks.TensorBoard(results_dir / 'logs')]
    if use_scheduler:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_get_scheduler(learning_rate)))

    model.fit(
        train_dataset.batch(batch_size),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=val_dataset.batch(64) if eval_on_val_set else None,
        class_weight=class_weights
    )
    logging.info("Finished training of a model. Saving results...")
    model.save_pretrained(results_dir)
    _save_predictions(model, results_dir, val_dataset, val_labels, val_texts)
    logging.info(f"Saved predictions and model weights in: '{results_dir}'")


def _load_train_data() -> pd.DataFrame:
    return pd.merge(
        pd.read_csv('data/train_bodies.csv'),
        pd.read_csv('data/train_stances.csv'),
        on='Body ID',
        how='left'
    )


def _load_test_data() -> pd.DataFrame:
    return pd.merge(
        pd.read_csv('data/competition_test_bodies.csv'),
        pd.read_csv('data/competition_test_stances.csv'),
        on='Body ID',
        how='left'
    )


def _save_predictions(
    model: tf.keras.Model,
    results_dir: Path,
    val_dataset: tf.data.Dataset,
    val_labels: pd.Series,
    val_texts: pd.DataFrame
):
    predictions = model.predict(val_dataset.batch(64))
    result = val_texts.copy()
    result.loc[:, 'Stance'] = val_labels
    for i in range(4):
        result.loc[:, f"logit_{str(i)}"] = predictions.logits[:, i]
    result.to_csv(results_dir / "predictions.csv", index=False)


def _save_args(
    seed: Optional[int],
    results_dir: Path,
    n_epochs: int,
    weight_decay: float,
    use_class_weights: bool,
    learning_rate: float,
    use_scheduler: bool,
    experiment_name: Optional[str]
):
    with open(results_dir / 'args.json', 'w') as json_file:
        json.dump(
            {
                "seed": seed,
                "n_epochs": n_epochs,
                "weight_decay": weight_decay,
                "use_class_weights": use_class_weights,
                "learning_rate": learning_rate,
                "experiment_name": experiment_name,
                "use_scheduler": use_scheduler
            },
            json_file
        )


def _get_dataset(tokenizer: DistilBertTokenizerFast, texts: pd.DataFrame, labels: pd.Series):
    encoded_texts = tokenizer(
        texts['Headline'].tolist(),
        texts['articleBody'].tolist(),
        truncation='longest_first',
        padding='max_length')

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(encoded_texts), labels)
    )
    return train_dataset


def _get_scheduler(initial_lr: float) -> callable:
    def _scheduler(epoch: int, _: float) -> float:
        if epoch < 10:
            return initial_lr
        else:
            return initial_lr / 2
    return _scheduler

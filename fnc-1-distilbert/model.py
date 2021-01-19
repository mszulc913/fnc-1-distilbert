import logging

import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification


def _get_number_of_trainable_params(model: tf.keras.Model) -> np.ndarray:
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def get_distilbert() -> TFDistilBertForSequenceClassification:
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    total_params = _get_number_of_trainable_params(model)
    model.distilbert.trainable = False
    trainable_params = _get_number_of_trainable_params(model)

    logging.info(
        f'Initialized TFDistilBertForSequenceClassification with {trainable_params} '
        f'trainable parameters ({total_params - trainable_params} frozen).')
    return model

from typing import List

import pandas as pd

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]
LABEL_MAPPING = {
    "agree": 0,
    "disagree": 1,
    "discuss": 2,
    "unrelated": 3
}


def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    return data.replace({'Stance': LABEL_MAPPING})


def score_submission(true_labels: List[str], pred_labels: List[str]) -> float:
    score = 0.0
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        if t == p:
            score += 0.25
            if t != 'unrelated':
                score += 0.50
        if t in RELATED and p in RELATED:
            score += 0.25
    return score

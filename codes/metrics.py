from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import PreTrainedTokenizerBase


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | str]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "report": classification_report(labels, predictions),
    }


def compute_token_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    scores: list[float] = []
    for prediction, reference in zip(predictions, references, strict=True):
        prediction_tokens = set(tokenizer.encode(prediction, add_special_tokens=False))
        reference_tokens = set(tokenizer.encode(reference, add_special_tokens=False))
        true_positive = len(prediction_tokens & reference_tokens)
        false_positive = len(prediction_tokens - reference_tokens)
        false_negative = len(reference_tokens - prediction_tokens)
        if true_positive + false_positive + false_negative == 0:
            scores.append(1.0)
            continue
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        scores.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(scores)) if scores else 0.0

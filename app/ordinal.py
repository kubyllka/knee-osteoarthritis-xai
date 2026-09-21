"""Shared ordinal-BCE decoding and differentiable KL-class scores."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ORDINAL_OUTPUTS = 4
KL_CLASSES = 5


def ordinal_logits_to_class(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Use the exact validation-time rule: count P(KL > k) > 0.5."""
    if logits.shape[-1] != ORDINAL_OUTPUTS:
        raise ValueError(f"Expected {ORDINAL_OUTPUTS} ordinal logits, got {logits.shape[-1]}.")
    return (torch.sigmoid(logits) > threshold).sum(dim=1)


def ordinal_logits_to_class_numpy(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if logits.shape[-1] != ORDINAL_OUTPUTS:
        raise ValueError(f"Expected {ORDINAL_OUTPUTS} ordinal logits, got {logits.shape[-1]}.")
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return (probabilities > threshold).sum(axis=1).astype(np.int64)


def ordinal_class_log_scores(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable score for each KL0..KL4 class from four threshold logits."""
    if logits.shape[-1] != ORDINAL_OUTPUTS:
        raise ValueError(f"Expected {ORDINAL_OUTPUTS} ordinal logits, got {logits.shape[-1]}.")
    positive = F.logsigmoid(logits)
    negative = F.logsigmoid(-logits)
    scores = [
        positive[:, :class_idx].sum(dim=1) + negative[:, class_idx:].sum(dim=1)
        for class_idx in range(KL_CLASSES)
    ]
    return torch.stack(scores, dim=1)


def ordinal_class_probabilities_numpy(logits: np.ndarray) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    class_probabilities = []
    for class_idx in range(KL_CLASSES):
        lower = probabilities[:, :class_idx].prod(axis=1) if class_idx else np.ones(len(logits))
        upper = (1.0 - probabilities[:, class_idx:]).prod(axis=1) if class_idx < ORDINAL_OUTPUTS else np.ones(len(logits))
        class_probabilities.append(lower * upper)
    return np.stack(class_probabilities, axis=1)


class OrdinalClassScoreModel(nn.Module):
    """Expose five differentiable class scores while preserving the ordinal base model."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return ordinal_class_log_scores(self.base_model(inputs))

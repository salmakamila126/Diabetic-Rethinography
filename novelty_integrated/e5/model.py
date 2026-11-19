"""Five-class B1-compatible head plus an optional monotone ordinal task."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class OrdinalHead(nn.Module):
    """One latent severity and four strictly ordered learnable thresholds."""

    def __init__(self, dimension, train_counts, *, min_gap=1e-4):
        super().__init__()
        if dimension < 1 or len(train_counts) != 5 or any(n <= 0 for n in train_counts):
            raise ValueError("Require positive feature dimension and five train grade counts")
        self.severity = nn.Linear(dimension, 1)
        self.base_threshold = nn.Parameter(torch.zeros(()))
        self.raw_gaps = nn.Parameter(torch.zeros(3))
        self.min_gap = min_gap
        # Train priors initialize the auxiliary outputs, never validation labels.
        total = sum(train_counts)
        cumulative = [sum(train_counts[k + 1:]) / total for k in range(4)]
        thresholds = [-math.log(p / (1 - p)) for p in cumulative]
        gaps = [thresholds[k] - thresholds[k - 1] - min_gap for k in range(1, 4)]
        if any(g <= 0 for g in gaps):
            raise ValueError("Train priors cannot initialize ordered thresholds")
        with torch.no_grad():
            self.severity.weight.zero_()
            self.severity.bias.zero_()
            self.base_threshold.fill_(thresholds[0])
            self.raw_gaps.copy_(torch.tensor([math.log(math.expm1(g)) for g in gaps]))

    def thresholds(self):
        gaps = F.softplus(self.raw_gaps) + self.min_gap
        return torch.cat((self.base_threshold.reshape(1),
                          self.base_threshold + torch.cumsum(gaps, dim=0)))

    def forward(self, embedding):
        if embedding.ndim != 2 or embedding.shape[1] != self.severity.in_features:
            raise ValueError("Unexpected pooled ConvNeXt feature shape")
        return self.severity(embedding.float()) - self.thresholds().reshape(1, 4)


class E5Heads(nn.Module):
    """Categorical logits alone determine the reported DR grade."""

    def __init__(self, dimension, train_counts, *, ordinal):
        super().__init__()
        # Instantiate first so the same RNG seed gives the same B1 classifier
        # initialization whether the optional ordinal head is enabled or not.
        self.categorical = nn.Linear(dimension, 5)
        self.ordinal = OrdinalHead(dimension, train_counts) if ordinal else None

    def forward(self, embedding):
        features = embedding.float()
        categorical_logits = self.categorical(features)
        ordinal_logits = self.ordinal(features) if self.ordinal is not None else None
        return categorical_logits, ordinal_logits

    def predict_logits(self, embedding):
        return self.categorical(embedding.float())

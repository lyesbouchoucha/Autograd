"""Layers, losses and optimizer built on top of the Tensor engine."""

import numpy as np

from .engine import Tensor


class Module:
    """Base class holding parameters and resetting their gradients."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Affine layer: an input of shape (n, d_in) gives x W + b of shape (n, d_out).

    The bias has shape (1, d_out) and is broadcast over the n rows; the engine
    sums its gradient back over the sample axis.
    """

    def __init__(self, n_in, n_out, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        # Uniform weights of scale 1/sqrt(n_in) keep the variance of the
        # activations roughly constant from one layer to the next.
        self.W = Tensor(rng.uniform(-1.0, 1.0, size=(n_in, n_out)))
        self.b = Tensor(np.zeros((1, n_out)))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):
    """Multilayer perceptron: affine layers separated by ReLU.

    `sizes` lists the successive dimensions, for instance [2, 16, 16, 3] for
    two input variables, two hidden layers of sixteen units and three classes.
    The last layer stays linear and returns scores, which the loss turns into
    probabilities.
    """

    def __init__(self, sizes, rng=None):
        self.layers = [Linear(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:

    def __init__(self, parameters, lr=0.1):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


def mse_loss(predictions, targets):
    """Mean squared error between predictions and targets (both Tensors)."""
    diff=(predictions - targets)
    return (diff*diff).mean()


def cross_entropy(logits, labels):
  
    labels = np.array(labels, dtype=int)
    n_samples, n_classes = logits.data.shape

    shifted = logits - Tensor(logits.data.max(axis=1, keepdims=True))
    log_normalizer = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - log_normalizer

    # Multiplying by the indicator matrix of the correct classes and summing
    # keeps exactly the term -log p_y of each row.
    one_hot = Tensor(np.eye(n_classes)[labels])
    return -(one_hot * log_probs).sum() * Tensor(1.0 / n_samples)

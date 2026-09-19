"""Reverse-mode automatic differentiation on NumPy arrays."""

import numpy as np


def unbroadcast(grad, shape):
    """Sum a gradient back to `shape` on axes NumPy broadcast during the forward pass.

    Both operands always have the same number of dimensions in this engine
    (e.g. a bias of shape (1, d) added to activations of shape (n, d)), so we
    only need to sum over axes where one side has size 1 and the other does not.
    """
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array together with its gradient and the operation that built it."""

    def __init__(self, data, children=(), op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros(self.data.shape)
        # Local derivative rule of this node, set by each operation below.
        self._backward = lambda: None
        self._children = children
        self._op = op

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # A sum passes its gradient unchanged to both inputs.
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise product."""
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # d(ab)/da = b and d(ab)/db = a, entry by entry.
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix product of two 2-D tensors."""
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            # The derivative of exp is exp, already stored in out.data.
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            # d(log x)/dx = 1/x.
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            # The derivative is 1 where the input is positive, 0 elsewhere.
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum over one axis, or over all entries when axis is None."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                # Put back the summed axis so that the gradient broadcasts.
                shape = list(self.data.shape)
                shape[axis] = 1
                grad = grad.reshape(shape)
            # Every summed entry enters the result with coefficient 1.
            self.grad += grad * np.ones(self.data.shape)

        out._backward = _backward
        return out

    def mean(self):
        """Mean over all entries."""
        count = self.data.size
        return self.sum() * Tensor(1.0 / count)

    def backward(self):
        """Fill the gradient of every tensor this one was computed from.

        The nodes are first sorted so that each node comes after the nodes it
        depends on. Walking that order backwards guarantees that a node's own
        gradient is complete before its local rule is applied.
        """
        order = []
        visited = set()

        def visit(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    visit(child)
                order.append(node)

        visit(self)

        # The derivative of a quantity with respect to itself is 1.
        self.grad = np.ones(self.data.shape)
        for node in reversed(order):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def __repr__(self):
        return "Tensor(shape=%s, op='%s')" % (self.data.shape, self._op)

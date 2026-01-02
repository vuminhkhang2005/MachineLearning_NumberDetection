"""
From-scratch SVM (still SVM) for MNIST-like digit recognition.

Goals:
  - No sklearn training/inference code
  - Explicit, readable implementation (NumPy)
  - Works on Colab + easy to serialize (npz)

Model:
  - One-vs-rest (OVR) linear SVM trained with mini-batch SGD on hinge loss + L2 regularization
  - Optional Random Fourier Features (RFF) to approximate an RBF-kernel SVM while staying linear in feature space
  - Predict-proba via temperature-softmax over class decision scores (not calibrated probabilities)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


def _stable_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass
class Standardizer:
    """Standardize features with stored mean/std (like StandardScaler, but explicit)."""

    mean_: np.ndarray
    std_: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, X: np.ndarray, *, eps: float = 1e-8) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean_=mean.astype(np.float32), std_=std.astype(np.float32), eps=eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


@dataclass
class RFFMap:
    """
    Random Fourier Features for RBF kernel approximation:
      z(x) = sqrt(2/D) * cos(Wx + b)

    For RBF kernel k(x,x') = exp(-gamma ||x-x'||^2),
      sample W ~ N(0, 2*gamma I), b ~ Uniform(0, 2pi)
    """

    W: np.ndarray  # (D, in_dim)
    b: np.ndarray  # (D,)
    scale: float   # sqrt(2/D)

    @classmethod
    def create(cls, in_dim: int, *, rff_dim: int, gamma: float, seed: int = 42) -> "RFFMap":
        rng = np.random.default_rng(seed)
        W = rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(rff_dim, in_dim)).astype(np.float32)
        b = rng.uniform(0.0, 2.0 * np.pi, size=(rff_dim,)).astype(np.float32)
        scale = float(np.sqrt(2.0 / rff_dim))
        return cls(W=W, b=b, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (N, in_dim)
        proj = X @ self.W.T + self.b  # (N, D)
        return (self.scale * np.cos(proj)).astype(np.float32)


FeatureMap = Literal["identity", "rff"]


@dataclass
class ScratchSVM:
    """
    OVR SVM with optional RFF feature map.

    Attributes:
      - W_: (n_classes, n_features_mapped)
      - b_: (n_classes,)
    """

    n_classes: int = 10
    feature_map: FeatureMap = "identity"
    standardizer: Optional[Standardizer] = None
    rff: Optional[RFFMap] = None
    W_: Optional[np.ndarray] = None
    b_: Optional[np.ndarray] = None
    temperature: float = 1.0

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, D). Got shape={X.shape}")

        if self.standardizer is not None:
            X = self.standardizer.transform(X).astype(np.float32)

        if self.feature_map == "identity":
            return X
        if self.feature_map == "rff":
            if self.rff is None:
                raise ValueError("feature_map='rff' but rff is None")
            return self.rff.transform(X)
        raise ValueError(f"Unknown feature_map: {self.feature_map}")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise ValueError("Model not trained/loaded: W_ or b_ is None")
        Z = self._prepare_X(X)  # (N, F)
        return Z @ self.W_.T + self.b_  # (N, C)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        temp = float(self.temperature) if self.temperature and self.temperature > 0 else 1.0
        return _stable_softmax(scores / temp, axis=1).astype(np.float32)

    @staticmethod
    def _make_ovr_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
        # Y: (N, C) with values in {-1, +1}
        y = y.astype(np.int64)
        Y = -np.ones((y.shape[0], n_classes), dtype=np.float32)
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 15,
        batch_size: int = 1024,
        reg_lambda: float = 1e-4,
        lr: float = 0.5,
        lr_decay: float = 0.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> "ScratchSVM":
        """
        Train OVR hinge-loss SVM with mini-batch SGD.

        Loss (per batch):
          L = 0.5*reg_lambda*||W||^2 + mean(max(0, 1 - y*(w·x+b)))

        Notes:
          - Bias is NOT regularized.
          - This is explicit and reasonably fast for MNIST on Colab.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, D). Got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,). Got shape={y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")

        # Fit standardizer on raw inputs (after caller normalization, e.g. /255).
        if self.standardizer is None:
            self.standardizer = Standardizer.fit(X)

        # Prepare feature map
        Z0 = self._prepare_X(X[:1])
        n_features = int(Z0.shape[1])

        rng = np.random.default_rng(seed)
        self.W_ = np.zeros((self.n_classes, n_features), dtype=np.float32)
        self.b_ = np.zeros((self.n_classes,), dtype=np.float32)

        n = X.shape[0]
        steps = 0

        for epoch in range(epochs):
            perm = rng.permutation(n)
            Xp = X[perm]
            yp = y[perm]

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                xb = Xp[start:end]
                yb = yp[start:end]

                Zb = self._prepare_X(xb)  # (B, F)
                Yb = self._make_ovr_labels(yb, self.n_classes)  # (B, C)

                scores = Zb @ self.W_.T + self.b_  # (B, C)
                margins = Yb * scores  # (B, C)
                active = (margins < 1.0).astype(np.float32)  # (B, C)

                # Gradients
                B = float(end - start)
                coeff = (active * Yb)  # (B, C)
                grad_W = reg_lambda * self.W_ - (coeff.T @ Zb) / B  # (C, F)
                grad_b = -(coeff.sum(axis=0)) / B  # (C,)

                # LR schedule
                cur_lr = lr / (1.0 + lr_decay * steps) if lr_decay > 0 else lr
                self.W_ -= cur_lr * grad_W
                self.b_ -= cur_lr * grad_b
                steps += 1

            if verbose:
                # quick train accuracy estimate on a tiny slice (cheap signal)
                sl = slice(0, min(2048, n))
                pred = self.predict(X[sl])
                acc = float((pred == y[sl]).mean())
                print(f"[epoch {epoch+1:02d}/{epochs}] approx train-acc={acc:.4f} lr={cur_lr:.4g}")

        return self

    def save_npz(self, path: str) -> None:
        if self.W_ is None or self.b_ is None:
            raise ValueError("Cannot save: model not trained/loaded.")
        if self.standardizer is None:
            raise ValueError("Cannot save: standardizer is missing.")

        payload = {
            "format_version": np.array([1], dtype=np.int32),
            "n_classes": np.array([self.n_classes], dtype=np.int32),
            "feature_map": np.array([0 if self.feature_map == "identity" else 1], dtype=np.int32),
            "temperature": np.array([float(self.temperature)], dtype=np.float32),
            "W": self.W_.astype(np.float32),
            "b": self.b_.astype(np.float32),
            "mean": self.standardizer.mean_.astype(np.float32),
            "std": self.standardizer.std_.astype(np.float32),
        }
        if self.feature_map == "rff":
            if self.rff is None:
                raise ValueError("feature_map='rff' but rff is None")
            payload.update(
                {
                    "rff_W": self.rff.W.astype(np.float32),
                    "rff_b": self.rff.b.astype(np.float32),
                    "rff_scale": np.array([float(self.rff.scale)], dtype=np.float32),
                }
            )

        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str) -> "ScratchSVM":
        data = np.load(path, allow_pickle=False)
        feature_map_code = int(data["feature_map"][0])
        feature_map: FeatureMap = "identity" if feature_map_code == 0 else "rff"

        standardizer = Standardizer(
            mean_=data["mean"].astype(np.float32),
            std_=data["std"].astype(np.float32),
        )

        rff = None
        if feature_map == "rff":
            rff = RFFMap(
                W=data["rff_W"].astype(np.float32),
                b=data["rff_b"].astype(np.float32),
                scale=float(data["rff_scale"][0]),
            )

        model = cls(
            n_classes=int(data["n_classes"][0]),
            feature_map=feature_map,
            standardizer=standardizer,
            rff=rff,
            W_=data["W"].astype(np.float32),
            b_=data["b"].astype(np.float32),
            temperature=float(data["temperature"][0]),
        )
        return model


def ensure_2d_float32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={X.shape}")
    return X


def mnist_flatten_normalize(images: np.ndarray) -> np.ndarray:
    """
    images:
      - (N, 28, 28) uint8/float
      - or already (N, 784)
    returns float32 (N, 784) in [0,1]
    """
    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


"""
Colab-ready training script (NO sklearn) for from-scratch SVM.

How to use on Google Colab:
  1) Runtime > Change runtime type > GPU (optional; this code uses NumPy so CPU is fine)
  2) Upload this repo or just this file + `svm_scratch_model.py`
  3) Run:
       !python train_svm_scratch_colab.py --feature-map rff --rff-dim 2048 --gamma 0.05 --epochs 20

Output:
  - outputs/svm_digit_classifier_scratch.npz

Note:
  - `predict_proba` is a softmax over SVM scores (not calibrated).
"""

from __future__ import annotations

import argparse
import os
from time import time

import numpy as np

from svm_scratch_model import RFFMap, ScratchSVM, mnist_flatten_normalize


def load_mnist_keras() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Available by default on Colab.
    from tensorflow.keras.datasets import mnist  # type: ignore

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from-scratch SVM on MNIST (Colab).")
    parser.add_argument("--feature-map", choices=["identity", "rff"], default="rff")
    parser.add_argument("--rff-dim", type=int, default=2048, help="RFF dimension (only if --feature-map rff)")
    parser.add_argument("--gamma", type=float, default=0.05, help="RBF gamma for RFF (only if --feature-map rff)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=60000, help="Use fewer training samples for faster runs")
    parser.add_argument("--output", type=str, default="outputs/svm_digit_classifier_scratch.npz")
    args = parser.parse_args()

    print("=" * 70)
    print("SVM from scratch (NumPy) - MNIST training")
    print("=" * 70)
    print(f"- feature_map: {args.feature_map}")
    if args.feature_map == "rff":
        print(f"- rff_dim: {args.rff_dim}")
        print(f"- gamma: {args.gamma}")
    print(f"- epochs: {args.epochs}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- reg_lambda: {args.reg_lambda}")
    print(f"- lr: {args.lr}")
    print(f"- lr_decay: {args.lr_decay}")
    print(f"- limit_train: {args.limit_train}")
    print()

    t0 = time()
    x_train, y_train, x_test, y_test = load_mnist_keras()
    X_train = mnist_flatten_normalize(x_train)
    X_test = mnist_flatten_normalize(x_test)

    if args.limit_train and args.limit_train < X_train.shape[0]:
        X_train = X_train[: args.limit_train]
        y_train = y_train[: args.limit_train]

    print(f"Data: X_train={X_train.shape} X_test={X_test.shape}")

    model = ScratchSVM(n_classes=10, feature_map=args.feature_map)
    if args.feature_map == "rff":
        model.rff = RFFMap.create(in_dim=X_train.shape[1], rff_dim=args.rff_dim, gamma=args.gamma, seed=args.seed)

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        reg_lambda=args.reg_lambda,
        lr=args.lr,
        lr_decay=args.lr_decay,
        seed=args.seed,
        verbose=True,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = float((y_pred == y_test).mean())
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Save
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_npz(out_path)
    print(f"Saved: {out_path}")
    print(f"Done in {time() - t0:.1f}s")


if __name__ == "__main__":
    main()


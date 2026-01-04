"""
TRAINING SCRIPT (COLAB) - SVM "tự cài" (NumPy, KHÔNG dùng sklearn)
==================================================================

Mục tiêu của file này
---------------------
- Đây là script huấn luyện mô hình nhận dạng chữ số MNIST bằng **SVM viết từ đầu**
  (tức là ta tự cài phần tối ưu/huấn luyện trong `svm_scratch_model.py`).
- File này đóng vai trò "điều phối":
  - tải dữ liệu MNIST
  - tiền xử lý dạng (N, 28, 28) -> (N, 784) và chuẩn hoá về [0,1]
  - cấu hình feature map (identity hoặc RFF)
  - gọi `.fit()` để train
  - đánh giá accuracy
  - lưu model ra `.npz`

Tại sao nói "from-scratch" nhưng vẫn là SVM?
-------------------------------------------
- SVM về bản chất là tối ưu một hàm mục tiêu dạng:

  Với bài toán nhị phân: y ∈ {-1, +1}
    minimize_W,b:  0.5 * λ * ||W||^2  +  mean(max(0, 1 - y * (W·x + b)))

  Trong đó:
  - **hinge loss**: max(0, 1 - y*s) với s = W·x + b
    - Nếu mẫu được phân loại đúng và "cách biên" đủ lớn (y*s ≥ 1) => loss = 0
    - Nếu sai hoặc gần biên (y*s < 1) => bị phạt tuyến tính theo mức vi phạm margin
  - **L2 regularization**: 0.5*λ*||W||^2
    - giúp giảm overfit, ưu tiên nghiệm có norm nhỏ -> margin lớn hơn (trực giác)
- Ở đây ta train bằng **mini-batch SGD** thay vì QP solver (như SVM cổ điển).
  Điều này giúp chạy ổn trên MNIST, dễ viết/giải thích, nhưng khác với "hard-margin QP"
  trong giáo trình cổ điển.

Vì MNIST có 10 lớp => làm sao SVM (nhị phân) xử lý?
---------------------------------------------------
- Ta dùng chiến lược **One-vs-Rest (OVR)**:
  - Tạo 10 bộ phân loại nhị phân:
      lớp k: y=+1 nếu ảnh là chữ số k, y=-1 nếu không phải k
  - Khi dự đoán, ta tính score của cả 10 bộ, chọn lớp có score lớn nhất (argmax).

Vì sao có tuỳ chọn RFF (Random Fourier Features)?
-------------------------------------------------
- Linear SVM chỉ học biên tuyến tính trong không gian feature.
- Kernel SVM (RBF) học biên phi tuyến: k(x,x') = exp(-gamma ||x-x'||^2)
  nhưng kernel SVM truyền thống rất nặng với N lớn.
- **RFF** là kỹ thuật xấp xỉ kernel RBF bằng cách ánh xạ:
    z(x) = sqrt(2/D) * cos(Wx + b)
  với W ~ N(0, 2*gamma I), b ~ Uniform(0, 2π)
  Khi đó: z(x)·z(x') ≈ k(x,x')
  => ta vẫn train "linear" trong không gian z(x) nhưng mô hình mang tính phi tuyến.

Chạy trên Google Colab
----------------------
1) Runtime > Change runtime type > GPU (tuỳ chọn; code này dùng NumPy nên CPU vẫn OK)
2) Upload repo hoặc tối thiểu 2 file:
   - `train_svm_scratch_colab.py`
   - `svm_scratch_model.py`
3) Run ví dụ:
     !python train_svm_scratch_colab.py --feature-map rff --rff-dim 2048 --gamma 0.05 --epochs 20

Output
------
- Mặc định lưu ra: `outputs/svm_digit_classifier_scratch.npz`
  File `.npz` chứa W, b, thống kê chuẩn hoá (mean/std), và nếu dùng RFF thì có cả W,b,scale của RFF.

Lưu ý quan trọng khi trình bày với giảng viên
---------------------------------------------
- `predict_proba` trong mô hình scratch chỉ là **softmax(score)** (có "temperature"),
  **KHÔNG phải xác suất đã hiệu chuẩn** (calibrated probability) như Platt scaling.
  Nói cách khác: dùng để "tham khảo độ tự tin tương đối", không nên khẳng định là xác suất thật.
"""

from __future__ import annotations

import argparse
import os
from time import time

import numpy as np

from svm_scratch_model import RFFMap, ScratchSVM, mnist_flatten_normalize


def load_mnist_keras() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Trên Google Colab thường có sẵn TensorFlow, nên ta dùng luôn MNIST từ keras.
    #
    # Dữ liệu MNIST:
    # - x_train: (60000, 28, 28) ảnh grayscale
    # - y_train: (60000,) nhãn 0..9
    # - x_test : (10000, 28, 28)
    # - y_test : (10000,)
    #
    # Pixel gốc thường là uint8 trong [0..255].
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
    # Tiền xử lý tối thiểu cho MNIST:
    # - flatten: (N, 28, 28) -> (N, 784)
    # - normalize: đưa pixel về [0,1] bằng /255
    #
    # Vì sao làm vậy?
    # - Gradient descent nhạy với scale: nếu feature quá lớn, bước học (lr) khó ổn định.
    # - Chuẩn hoá về [0,1] giúp tốc độ hội tụ và ổn định số tốt hơn.
    #
    # Lưu ý: trong `ScratchSVM.fit`, ta còn fit `Standardizer` (mean/std) để z-score
    # nhằm đưa các chiều về cùng thang đo (giảm bias theo chiều có phương sai lớn).
    X_train = mnist_flatten_normalize(x_train)
    X_test = mnist_flatten_normalize(x_test)

    if args.limit_train and args.limit_train < X_train.shape[0]:
        # Dùng ít mẫu train hơn để chạy nhanh trong demo/báo cáo.
        # Khi N giảm, thời gian train giảm gần tuyến tính theo N (vì SGD qua từng batch).
        X_train = X_train[: args.limit_train]
        y_train = y_train[: args.limit_train]

    print(f"Data: X_train={X_train.shape} X_test={X_test.shape}")

    # Khởi tạo model SVM OVR.
    # - n_classes=10 cho MNIST
    # - feature_map:
    #   - "identity": dùng trực tiếp vector 784 chiều
    #   - "rff": ánh xạ sang D chiều (rff_dim) để xấp xỉ RBF-kernel
    model = ScratchSVM(n_classes=10, feature_map=args.feature_map)
    if args.feature_map == "rff":
        # Tạo RFF map với seed cố định để tái lập kết quả (reproducibility).
        #
        # Giải thích tham số:
        # - rff_dim (D): số chiều không gian đặc trưng sau ánh xạ.
        #   D càng lớn => xấp xỉ kernel càng tốt nhưng tốn RAM/CPU hơn.
        # - gamma: tham số của RBF kernel (độ "hẹp" của Gaussian).
        #   gamma lớn => kernel hẹp, biên quyết định phức tạp hơn (dễ overfit).
        #   gamma nhỏ => kernel rộng, mô hình mượt hơn (dễ underfit nếu quá nhỏ).
        model.rff = RFFMap.create(in_dim=X_train.shape[1], rff_dim=args.rff_dim, gamma=args.gamma, seed=args.seed)

    # Train bằng SGD trên hinge loss.
    #
    # Các hyperparameter cần hiểu để trình bày:
    # - epochs: số lần quét qua toàn bộ tập train.
    # - batch_size: kích thước mini-batch (trade-off: ổn định gradient vs tốc độ).
    # - reg_lambda (λ): hệ số regularization L2 trên W (bias b không regularize).
    #   λ lớn => W nhỏ hơn => biên "đơn giản" hơn => giảm overfit nhưng có thể underfit.
    # - lr: learning rate (tốc độ học) của SGD.
    # - lr_decay: nếu >0, lr giảm dần theo steps: lr_t = lr / (1 + lr_decay * step)
    #   giúp ổn định khi gần hội tụ.
    #
    # Trong `ScratchSVM.fit`:
    # - Với mỗi batch, tính score cho tất cả lớp (B, C)
    # - Tính margin = y * score, với y ∈ {-1,+1} theo OVR
    # - active = margin < 1 => các mẫu vi phạm margin đóng góp gradient (support vectors "mềm")
    # - Tính gradient theo W,b rồi cập nhật theo SGD.
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
    # Dự đoán: lấy argmax(score) qua 10 lớp.
    # Accuracy = tỉ lệ dự đoán đúng trên tập test 10k mẫu (chuẩn MNIST).
    y_pred = model.predict(X_test)
    acc = float((y_pred == y_test).mean())
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Save
    # Lưu model ra .npz để dễ dùng lại trong môi trường không có sklearn.
    #
    # Nội dung lưu (tuỳ theo feature_map):
    # - W, b: trọng số và bias của 10 bộ phân loại OVR
    # - mean, std: tham số standardizer (để tiền xử lý giống lúc train)
    # - nếu rff:
    #   - rff_W, rff_b, rff_scale: tham số ánh xạ RFF (để map input giống lúc train)
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_npz(out_path)
    print(f"Saved: {out_path}")
    print(f"Done in {time() - t0:.1f}s")


if __name__ == "__main__":
    main()


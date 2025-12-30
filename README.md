# MachineLearning_NumberDetection

## 🔢 Mô hình SVM Nhận dạng Chữ số Viết tay (MNIST)

Project này triển khai đầy đủ quy trình xây dựng mô hình SVM (Support Vector Machine) để nhận dạng chữ số viết tay sử dụng bộ dữ liệu MNIST.

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc Project](#-cấu-trúc-project)
- [Kết quả](#-kết-quả)
- [API Reference](#-api-reference)

## ✨ Tính năng

- **Tiền xử lý dữ liệu**: Tự động tải và chuẩn hóa dữ liệu MNIST
- **Huấn luyện SVM**: Hỗ trợ nhiều kernel (RBF, Linear, Polynomial, Sigmoid)
- **Tối ưu hóa**: Grid Search để tìm siêu tham số tốt nhất
- **PCA**: Tùy chọn giảm chiều với PCA
- **Đánh giá**: Classification report, Confusion matrix
- **Xuất cho Ensemble**: Xuất xác suất dự đoán để sử dụng trong hệ ensemble
- **GPU Support**: Hỗ trợ RAPIDS cuML cho GPU acceleration trên Google Colab

## 🛠 Cài đặt

### Yêu cầu

```bash
# Các thư viện cần thiết
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Cài đặt cuML cho GPU (Google Colab)

```python
# Chạy trên Google Colab với GPU runtime
!pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

## 🚀 Sử dụng

### Cách 1: Chạy Jupyter Notebook (Khuyến nghị cho Google Colab)

1. Upload file `svm_digit_recognition.ipynb` lên Google Colab
2. Chọn Runtime > Change runtime type > GPU
3. Chạy từng cell theo thứ tự

### Cách 2: Chạy Python Script

```bash
# Chạy với cấu hình mặc định
python svm_digit_recognition.py

# Chạy với tùy chỉnh
python svm_digit_recognition.py --subset-size 20000 --kernel rbf --C 10

# Sử dụng toàn bộ dữ liệu
python svm_digit_recognition.py --use-full-data

# Bỏ qua Grid Search
python svm_digit_recognition.py --skip-grid-search --kernel rbf --C 1.0

# Sử dụng PCA
python svm_digit_recognition.py --use-pca --pca-components 100
```

### Tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--subset-size` | Số mẫu train để sử dụng | 60000 |
| `--use-full-data` | Sử dụng toàn bộ dữ liệu train | True |
| `--skip-grid-search` | Bỏ qua Grid Search | False |
| `--kernel` | Loại kernel (rbf, linear, poly, sigmoid) | rbf |
| `--C` | Hệ số regularization | 10.0 |
| `--use-pca` | Sử dụng PCA giảm chiều | False |
| `--pca-components` | Số thành phần PCA | 100 |

### Script train mới (khuyến nghị)

```bash
# Train với full data và tham số tối ưu
python train_svm_model.py --samples 60000

# Train nhanh để test (5000 mẫu)
python train_svm_model.py --quick

# Train với tham số tùy chỉnh
python train_svm_model.py --samples 30000 --C 10.0 --gamma 0.01
```

## 🧪 Test Model

### Cách 1: Ứng dụng Desktop (Tkinter)

Ứng dụng desktop cho phép bạn vẽ chữ số để test model.

```bash
# Chạy ứng dụng desktop
python test_app.py
```

**Tính năng:**
- ✏️ Vẽ chữ số trực tiếp trên canvas
- 📂 Upload ảnh chữ số từ máy tính (PNG, JPG, BMP, GIF, TIFF, WebP)
- 🎲 Test với mẫu ngẫu nhiên từ MNIST
- 📊 Hiển thị biểu đồ xác suất dự đoán
- 🖼️ Xem ảnh đã xử lý (28x28)

### Cách 2: Command Line (CLI)

```bash
# Test với mẫu MNIST ngẫu nhiên (mặc định 5 mẫu)
python test_model_cli.py

# Test với nhiều mẫu hơn
python test_model_cli.py --samples 10

# Test với file ảnh
python test_model_cli.py --image path/to/your/digit.png

# Đánh giá model trên toàn bộ test set
python test_model_cli.py --evaluate

# Không hiển thị đồ thị
python test_model_cli.py --no-plot
```

## 📁 Cấu trúc Project

```
/workspace
├── README.md                      # Tài liệu hướng dẫn
├── requirements.txt               # Dependencies
├── svm_digit_recognition.ipynb    # Jupyter Notebook (Google Colab)
├── svm_digit_recognition.py       # Python script huấn luyện (cũ)
├── train_svm_model.py             # 🆕 Script train model cải tiến (KHUYẾN NGHỊ)
├── test_app.py                    # Ứng dụng desktop test (Tkinter)
├── test_model_cli.py              # CLI test model
├── svm_digit_classifier.joblib    # Mô hình đã train (copy ở root)
└── outputs/                       # Thư mục đầu ra (tự động tạo)
    └── svm_digit_classifier.joblib      # Mô hình đã train (98.30% accuracy)
```

## 📊 Kết quả

### Hiệu suất mô hình

| Số mẫu train | Accuracy | Train Time | Support Vectors |
|--------------|----------|------------|-----------------|
| 60,000 (full) | **98.30%** | ~6 phút | ~10,700 |
| 30,000 | 97.76% | ~2.5 phút | ~6,900 |
| 10,000 | ~96-97% | ~30s | ~3,000 |

### Cấu hình tối ưu cho MNIST

| Tham số | Giá trị tối ưu |
|---------|----------------|
| Kernel | RBF |
| C | 10.0 |
| Gamma | 0.01 |

*Lưu ý: Kết quả có thể thay đổi nhẹ tùy thuộc vào phần cứng*

## 🔧 API Reference

### Sử dụng mô hình đã lưu

```python
import joblib
import numpy as np

# Load mô hình
model = joblib.load('outputs/svm_digit_classifier.joblib')

# Dự đoán nhãn
image = np.random.rand(1, 784)  # Ảnh 28x28 đã flatten
predictions = model.predict(image)
print(f"Predicted digit: {predictions[0]}")

# Dự đoán xác suất
probabilities = model.predict_proba(image)
print(f"Probabilities: {probabilities}")
```

### Hàm predict_digit

```python
from svm_digit_recognition import predict_digit

# Load mô hình
model = joblib.load('outputs/svm_digit_classifier.joblib')

# Dự đoán từ ảnh (28x28 hoặc 784)
image = np.random.rand(28, 28) * 255  # Giá trị 0-255
result = predict_digit(model, image)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
```

### Load đầu ra cho Ensemble

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('outputs/svm_predictions_for_ensemble.csv')
print(df.head())

# Load numpy arrays
probabilities = np.load('outputs/svm_probabilities.npy')
predictions = np.load('outputs/svm_predictions.npy')

print(f"Probabilities shape: {probabilities.shape}")  # (n_samples, 10)
print(f"Predictions shape: {predictions.shape}")      # (n_samples,)
```

## 📚 Lộ trình xây dựng mô hình

### 1. Chuẩn bị dữ liệu và tiền xử lý

- Tải dữ liệu MNIST (60k train, 10k test, 28x28 pixels)
- Flatten ảnh thành vector 784 chiều
- Chuẩn hóa pixel về [0, 1]

### 2. Huấn luyện mô hình SVM

- Sử dụng SVC với RBF kernel
- **KHÔNG** dùng StandardScaler để tránh vấn đề không khớp khi dự đoán ảnh mới
- Chuẩn hóa đơn giản: chia 255 để đưa về [0, 1]
- Tham số tối ưu: C=10.0, gamma=0.01
- Hỗ trợ probability output

### 3. Đánh giá mô hình

- Accuracy score
- Classification report (precision, recall, F1)
- Confusion matrix

### 4. Tối ưu hóa mô hình

- GridSearchCV cho C, gamma, kernel
- PCA để giảm chiều (tùy chọn)
- Cross-validation

### 5. Xuất đầu ra cho Ensemble

- Xác suất dự đoán (predict_proba)
- Nhãn dự đoán
- Format CSV và numpy

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License

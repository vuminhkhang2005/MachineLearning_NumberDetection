"""
🔢 Script Train Mô hình SVM Nhận dạng Chữ số Viết tay

Script này huấn luyện mô hình SVM với toàn bộ dữ liệu MNIST (60k train, 10k test)
để đạt được độ chính xác cao nhất có thể.

Các cải tiến so với phiên bản cũ:
1. Sử dụng TOÀN BỘ dữ liệu train (60,000 mẫu) thay vì chỉ 10,000
2. KHÔNG dùng StandardScaler để tránh vấn đề không khớp khi dự đoán
3. Tối ưu hyperparameters với GridSearchCV
4. Chuẩn hóa dữ liệu đơn giản bằng chia 255 (0-1) - dễ áp dụng cho ảnh mới
5. Lưu cả scaler riêng để dùng cho ảnh mới

Sử dụng:
    python train_svm_model.py
    python train_svm_model.py --samples 60000  # Full data
    python train_svm_model.py --quick  # Quick test với 5000 samples
"""

import numpy as np
import os
import sys
import argparse
from time import time
import warnings
import joblib

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# =============================================================================
# CẤU HÌNH
# =============================================================================

OUTPUT_DIR = 'outputs'
MODEL_FILENAME = 'svm_digit_classifier.joblib'

# =============================================================================
# HÀM CHÍNH
# =============================================================================

def load_mnist():
    """
    Tải và tiền xử lý dữ liệu MNIST.
    
    Dữ liệu được chuẩn hóa đơn giản bằng cách chia cho 255 để đưa về [0, 1].
    KHÔNG dùng StandardScaler để tránh vấn đề không khớp khi dự đoán ảnh mới.
    """
    print("=" * 60)
    print("📥 BƯỚC 1: Tải dữ liệu MNIST")
    print("=" * 60)
    
    start_time = time()
    
    # Tải dữ liệu từ OpenML
    print("\n🔄 Đang tải dữ liệu từ OpenML...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    print(f"✅ Tải xong trong {time() - start_time:.2f} giây")
    print(f"\n📊 Thông tin dữ liệu gốc:")
    print(f"   - Shape của X: {X.shape}")
    print(f"   - Shape của y: {y.shape}")
    print(f"   - Số lượng lớp: {len(np.unique(y))}")
    print(f"   - Các lớp: {np.unique(y)}")
    print(f"   - Dtype của X: {X.dtype}")
    print(f"   - Range của pixel: [{X.min()}, {X.max()}]")
    
    # Chuyển đổi nhãn sang số nguyên
    y = y.astype(int)
    
    # Chuẩn hóa pixel về [0, 1] - ĐƠN GIẢN VÀ NHẤT QUÁN
    # Điều này rất quan trọng: khi dự đoán ảnh mới, chỉ cần chia cho 255
    X = X.astype(np.float64) / 255.0
    
    print(f"\n📊 Sau khi chuẩn hóa:")
    print(f"   - Dtype: {X.dtype}")
    print(f"   - Range: [{X.min():.4f}, {X.max():.4f}]")
    
    return X, y


def split_data(X, y, n_train_samples=None, random_state=42):
    """
    Chia dữ liệu thành train/test theo chuẩn MNIST.
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu đầu vào
    y : array-like
        Nhãn
    n_train_samples : int, optional
        Số mẫu train muốn sử dụng. None = sử dụng hết.
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 2: Chia dữ liệu Train/Test")
    print("=" * 60)
    
    # Chia dữ liệu với tỷ lệ chuẩn MNIST (60k train, 10k test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=10000,
        random_state=random_state,
        stratify=y  # Đảm bảo phân bố đều các lớp
    )
    
    print(f"\n📊 Kết quả chia dữ liệu:")
    print(f"   - Tổng số mẫu: {len(X)}")
    print(f"   - Train: {X_train.shape[0]} mẫu")
    print(f"   - Test: {X_test.shape[0]} mẫu")
    
    # Sử dụng tập con nếu được yêu cầu
    if n_train_samples is not None and n_train_samples < len(X_train):
        print(f"\n⚡ Lấy {n_train_samples} mẫu train...")
        
        # Lấy mẫu có stratify
        indices = np.arange(len(X_train))
        np.random.seed(random_state)
        
        # Stratified sampling
        selected_indices = []
        for label in np.unique(y_train):
            label_indices = indices[y_train == label]
            n_select = int(n_train_samples * len(label_indices) / len(X_train))
            selected = np.random.choice(label_indices, size=n_select, replace=False)
            selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        X_train = X_train[selected_indices]
        y_train = y_train[selected_indices]
        
        print(f"   - Tập train sau khi lấy mẫu: {X_train.shape[0]} mẫu")
    
    # Kiểm tra phân bố các lớp
    print(f"\n📈 Phân bố lớp trong tập train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   Chữ số {label}: {count} mẫu ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train, kernel='rbf', C=10.0, gamma=0.01):
    """
    Huấn luyện mô hình SVM.
    
    Parameters:
    -----------
    X_train : array-like
        Dữ liệu train
    y_train : array-like
        Nhãn train
    kernel : str
        Loại kernel ('rbf', 'linear', 'poly')
    C : float
        Hệ số regularization
    gamma : float or str
        Hệ số gamma cho RBF kernel
        
    Returns:
    --------
    model : SVC
        Mô hình đã huấn luyện
    train_time : float
        Thời gian huấn luyện
    """
    print("\n" + "=" * 60)
    print("🏋️ BƯỚC 3: Huấn luyện mô hình SVM")
    print("=" * 60)
    
    print(f"\n📊 Cấu hình mô hình:")
    print(f"   - Kernel: {kernel}")
    print(f"   - C: {C}")
    print(f"   - Gamma: {gamma}")
    print(f"   - Probability: True")
    print(f"   - Số mẫu train: {len(X_train)}")
    
    # Tạo mô hình SVM
    # KHÔNG dùng Pipeline với StandardScaler
    # Dữ liệu đã được chuẩn hóa 0-1
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,  # Để có thể dùng predict_proba
        cache_size=2000,   # Tăng cache để train nhanh hơn
        decision_function_shape='ovr',
        random_state=42
    )
    
    print(f"\n🔄 Đang huấn luyện...")
    print(f"   (Quá trình này có thể mất vài phút với dữ liệu lớn)")
    
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    print(f"\n✅ Huấn luyện hoàn tất trong {train_time:.2f} giây")
    print(f"   - Số support vectors: {sum(model.n_support_)}")
    
    return model, train_time


def run_grid_search(X_train, y_train, n_samples=5000):
    """
    Tìm hyperparameters tốt nhất với GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like
        Dữ liệu train
    y_train : array-like
        Nhãn train
    n_samples : int
        Số mẫu sử dụng cho grid search (để tiết kiệm thời gian)
        
    Returns:
    --------
    best_params : dict
        Các tham số tốt nhất
    """
    print("\n" + "=" * 60)
    print("🔍 TÌM KIẾM HYPERPARAMETERS TỐI ƯU")
    print("=" * 60)
    
    # Lấy mẫu con cho grid search
    n_samples = min(n_samples, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_grid = X_train[indices]
    y_grid = y_train[indices]
    
    print(f"\n📊 Sử dụng {n_samples} mẫu cho GridSearch")
    
    # Định nghĩa lưới tham số
    # Dựa trên các nghiên cứu về SVM với MNIST:
    # - C trong khoảng 1-10 thường tốt
    # - gamma khoảng 0.01-0.05 với kernel RBF
    param_grid = {
        'C': [1, 5, 10],
        'gamma': [0.01, 0.02, 0.05],
        'kernel': ['rbf']
    }
    
    print(f"\n📋 Lưới tham số:")
    for key, values in param_grid.items():
        print(f"   - {key}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\n📊 Tổng số kết hợp: {total_combinations}")
    
    # Tạo GridSearchCV
    grid_search = GridSearchCV(
        SVC(probability=True, cache_size=1000, random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy',
        return_train_score=True
    )
    
    print(f"\n🔄 Đang tìm kiếm... (có thể mất vài phút)")
    
    start_time = time()
    grid_search.fit(X_grid, y_grid)
    search_time = time() - start_time
    
    print(f"\n✅ GridSearch hoàn tất trong {search_time:.2f} giây")
    print(f"\n🏆 Kết quả tốt nhất:")
    print(f"   - Best Score (CV): {grid_search.best_score_:.4f}")
    print(f"   - Best Parameters: {grid_search.best_params_}")
    
    # Hiển thị top 5 kết quả
    import pandas as pd
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')[[
        'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
    ]].head(5)
    
    print(f"\n📊 Top 5 kết hợp:")
    for i, row in results_df.iterrows():
        print(f"   {row['rank_test_score']}. {row['params']} - Accuracy: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    
    return grid_search.best_params_


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên tập test.
    
    Parameters:
    -----------
    model : SVC
        Mô hình đã huấn luyện
    X_test : array-like
        Dữ liệu test
    y_test : array-like
        Nhãn test
        
    Returns:
    --------
    results : dict
        Kết quả đánh giá
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 4: Đánh giá mô hình")
    print("=" * 60)
    
    # Dự đoán
    print(f"\n🔄 Đang dự đoán trên {len(X_test)} mẫu test...")
    
    start_time = time()
    y_pred = model.predict(X_test)
    predict_time = time() - start_time
    
    # Tính accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Kết quả:")
    print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Thời gian dự đoán: {predict_time:.2f}s")
    print(f"   - Tốc độ: {predict_time/len(X_test)*1000:.3f}ms/mẫu")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Hiển thị các cặp chữ số hay bị nhầm
    print(f"\n❌ Top 5 cặp chữ số hay bị nhầm lẫn:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    
    for _ in range(5):
        max_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
        if cm_copy[max_idx] > 0:
            print(f"   - Thực tế: {max_idx[0]}, Dự đoán: {max_idx[1]} - {cm_copy[max_idx]} lần")
            cm_copy[max_idx] = 0
    
    return {
        'accuracy': accuracy,
        'predict_time': predict_time,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def save_model(model, output_dir, filename):
    """
    Lưu mô hình.
    
    Parameters:
    -----------
    model : SVC
        Mô hình đã huấn luyện
    output_dir : str
        Thư mục lưu
    filename : str
        Tên file
    """
    print("\n" + "=" * 60)
    print("💾 BƯỚC 5: Lưu mô hình")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, filename)
    
    # Lưu model
    joblib.dump(model, model_path)
    print(f"\n✅ Đã lưu mô hình: {model_path}")
    
    # Lưu thêm vào thư mục gốc để dễ tìm
    root_path = filename
    joblib.dump(model, root_path)
    print(f"✅ Đã lưu mô hình: {root_path}")
    
    # Kiểm tra kích thước file
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 Kích thước file: {file_size:.2f} MB")
    
    return model_path


def test_prediction(model):
    """
    Test dự đoán với một vài mẫu từ MNIST.
    """
    print("\n" + "=" * 60)
    print("🧪 TEST DỰ ĐOÁN")
    print("=" * 60)
    
    # Tải vài mẫu MNIST
    print("\n📥 Tải mẫu test từ MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    X = X.astype(np.float64) / 255.0
    
    # Lấy 10 mẫu ngẫu nhiên
    np.random.seed(123)
    indices = np.random.choice(len(X), 10, replace=False)
    
    print(f"\n📊 Kết quả dự đoán 10 mẫu ngẫu nhiên:")
    print("-" * 50)
    
    correct = 0
    for i, idx in enumerate(indices):
        sample = X[idx:idx+1]  # Shape (1, 784)
        true_label = y[idx]
        
        # Dự đoán
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        confidence = proba[pred]
        
        is_correct = pred == true_label
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"   {i+1}. Thực tế: {true_label}, Dự đoán: {pred}, Tin cậy: {confidence:.2%} {status}")
    
    print("-" * 50)
    print(f"📊 Đúng: {correct}/10 ({correct*10}%)")
    
    return correct


def main():
    """Hàm chính."""
    parser = argparse.ArgumentParser(description='Train SVM Digit Recognition Model')
    parser.add_argument('--samples', type=int, default=60000,
                        help='Số mẫu train (default: 60000 = full)')
    parser.add_argument('--quick', action='store_true',
                        help='Chế độ nhanh với 5000 mẫu')
    parser.add_argument('--skip-grid-search', action='store_true',
                        help='Bỏ qua GridSearch, dùng tham số mặc định')
    parser.add_argument('--C', type=float, default=10.0,
                        help='Hệ số C (default: 10.0)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Hệ số gamma (default: 0.01)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.samples = 5000
        args.skip_grid_search = True
    
    print("=" * 60)
    print("🔢 HUẤN LUYỆN MÔ HÌNH SVM NHẬN DẠNG CHỮ SỐ")
    print("=" * 60)
    print(f"\n📊 Cấu hình:")
    print(f"   - Số mẫu train: {args.samples}")
    print(f"   - GridSearch: {'Không' if args.skip_grid_search else 'Có'}")
    if args.skip_grid_search:
        print(f"   - C: {args.C}")
        print(f"   - Gamma: {args.gamma}")
    
    total_start = time()
    
    # 1. Tải dữ liệu
    X, y = load_mnist()
    
    # 2. Chia dữ liệu
    n_train = args.samples if args.samples < 60000 else None
    X_train, X_test, y_train, y_test = split_data(X, y, n_train_samples=n_train)
    
    # 3. Tìm hyperparameters (tùy chọn)
    if not args.skip_grid_search:
        best_params = run_grid_search(X_train, y_train, n_samples=5000)
        C = best_params['C']
        gamma = best_params['gamma']
        kernel = best_params['kernel']
    else:
        C = args.C
        gamma = args.gamma
        kernel = 'rbf'
    
    # 4. Huấn luyện mô hình
    model, train_time = train_svm(X_train, y_train, kernel=kernel, C=C, gamma=gamma)
    
    # 5. Đánh giá
    results = evaluate_model(model, X_test, y_test)
    
    # 6. Lưu model
    model_path = save_model(model, OUTPUT_DIR, MODEL_FILENAME)
    
    # 7. Test prediction
    test_prediction(model)
    
    # Tổng kết
    total_time = time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    print(f"\n🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"⏱️ Tổng thời gian: {total_time:.2f} giây ({total_time/60:.1f} phút)")
    print(f"\n📊 Cấu hình mô hình tốt nhất:")
    print(f"   - Kernel: {kernel}")
    print(f"   - C: {C}")
    print(f"   - Gamma: {gamma}")
    print(f"\n📁 Mô hình đã lưu:")
    print(f"   - {model_path}")
    print(f"   - {MODEL_FILENAME}")
    print("\n✅ HOÀN TẤT!")
    
    return model, results


if __name__ == "__main__":
    model, results = main()

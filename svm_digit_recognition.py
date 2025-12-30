"""
🔢 Mô hình SVM Nhận dạng Chữ số Viết tay (MNIST)

Script này triển khai đầy đủ quy trình xây dựng mô hình SVM để nhận dạng 
chữ số viết tay sử dụng bộ dữ liệu MNIST.

Các bước chính:
1. Chuẩn bị dữ liệu và tiền xử lý
2. Huấn luyện mô hình SVM
3. Đánh giá mô hình
4. Tối ưu hóa mô hình (GridSearch, PCA)
5. Xuất đầu ra cho hệ ensemble

Sử dụng:
    python svm_digit_recognition.py

Tác giả: AI Assistant
Ngày tạo: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
import argparse
import os

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import joblib

# =============================================================================
# CẤU HÌNH
# =============================================================================

class Config:
    """Cấu hình cho mô hình."""
    
    # Dữ liệu
    TEST_SIZE = 10000
    RANDOM_STATE = 42
    USE_SUBSET = False  # Sử dụng toàn bộ dữ liệu train để đạt accuracy cao nhất
    SUBSET_SIZE = 60000  # Số mẫu train (60000 = full MNIST train)
    
    # SVM - Tham số tối ưu cho MNIST
    DEFAULT_KERNEL = 'rbf'
    DEFAULT_C = 10.0  # Tối ưu cho MNIST (thay vì 1.0)
    DEFAULT_GAMMA = 0.01  # Tối ưu cho MNIST (thay vì 'scale')
    
    # GridSearch
    GRID_SEARCH_SAMPLES = 5000
    GRID_CV = 3
    
    # PCA
    USE_PCA = False
    PCA_COMPONENTS = 100
    
    # Output
    OUTPUT_DIR = 'outputs'
    MODEL_FILENAME = 'svm_digit_classifier.joblib'
    PREDICTIONS_FILENAME = 'svm_predictions_for_ensemble.csv'


# =============================================================================
# HÀM TIỆN ÍCH
# =============================================================================

def check_gpu():
    """Kiểm tra GPU và cuML."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_available = True
    except:
        gpu_available = False
    
    try:
        from cuml.svm import SVC as cuSVC
        cuml_available = True
    except ImportError:
        cuml_available = False
    
    return gpu_available, cuml_available


def load_mnist():
    """Tải và tiền xử lý dữ liệu MNIST."""
    print("📥 Đang tải dữ liệu MNIST...")
    start_time = time()
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    print(f"✅ Tải xong trong {time() - start_time:.2f} giây")
    print(f"\n📊 Thông tin dữ liệu:")
    print(f"   - Shape của X: {X.shape}")
    print(f"   - Shape của y: {y.shape}")
    print(f"   - Số lượng lớp: {len(np.unique(y))}")
    
    # Chuyển đổi
    y = y.astype(int)
    X = X.astype(np.float32) / 255.0
    
    print(f"   - Range sau chuẩn hóa: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y


def split_data(X, y, config):
    """Chia dữ liệu train/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 Chia dữ liệu:")
    print(f"   - Train: {X_train.shape[0]} mẫu")
    print(f"   - Test: {X_test.shape[0]} mẫu")
    
    # Sử dụng tập con nếu cần
    if config.USE_SUBSET:
        print(f"\n⚡ Sử dụng tập con {config.SUBSET_SIZE} mẫu...")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=config.SUBSET_SIZE, 
                                      random_state=config.RANDOM_STATE)
        for train_idx, _ in sss.split(X_train, y_train):
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
        print(f"   - Tập train subset: {X_train.shape[0]} mẫu")
    
    return X_train, X_test, y_train, y_test


def create_pipeline(kernel='rbf', C=10.0, gamma=0.01, use_pca=False, n_components=100):
    """
    Tạo pipeline cho SVM.
    
    LƯU Ý QUAN TRỌNG: KHÔNG dùng StandardScaler trong pipeline.
    Dữ liệu đã được chuẩn hóa về [0, 1] bằng cách chia cho 255.
    Điều này giúp đảm bảo tính nhất quán khi dự đoán ảnh mới.
    """
    steps = []
    
    # KHÔNG dùng StandardScaler - sử dụng chuẩn hóa 0-1 đơn giản thay thế
    # Điều này tránh vấn đề không khớp khi dự đoán ảnh mới
    
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components)))
    
    steps.append(('svc', SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        cache_size=2000,
        random_state=42
    )))
    
    # Nếu không dùng PCA, trả về SVC trực tiếp
    if len(steps) == 1:
        return steps[0][1]
    
    return Pipeline(steps)


def train_svm(X_train, y_train, kernel='rbf', C=10.0, gamma=0.01,
              use_pca=False, n_components=100):
    """Huấn luyện mô hình SVM."""
    print(f"\n🏋️ Bắt đầu huấn luyện SVM...")
    print(f"   - Kernel: {kernel}")
    print(f"   - C: {C}")
    print(f"   - Gamma: {gamma}")
    print(f"   - PCA: {use_pca} ({n_components} components)" if use_pca else f"   - PCA: {use_pca}")
    print(f"   - Số mẫu train: {len(X_train)}")
    
    model = create_pipeline(kernel, C, gamma, use_pca, n_components)
    
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    print(f"\n✅ Huấn luyện hoàn tất trong {train_time:.2f} giây")
    
    # Hiển thị số support vectors nếu có
    if hasattr(model, 'n_support_'):
        print(f"   - Số support vectors: {sum(model.n_support_)}")
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['svc'], 'n_support_'):
        print(f"   - Số support vectors: {sum(model.named_steps['svc'].n_support_)}")
    
    return model, train_time


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Đánh giá mô hình."""
    print(f"\n{'='*60}")
    print(f"📊 Đánh giá: {model_name}")
    print(f"{'='*60}")
    
    # Dự đoán
    start_time = time()
    y_pred = model.predict(X_test)
    predict_time = time() - start_time
    
    # Tính accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"⏱️ Thời gian dự đoán: {predict_time:.4f} giây")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'predict_time': predict_time,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def run_grid_search(X_train, y_train, config):
    """Thực hiện Grid Search để tìm tham số tốt nhất."""
    print("\n🔍 Bắt đầu Grid Search...")
    print("⚠️ Quá trình này có thể mất vài phút...\n")
    
    # Sử dụng SVC trực tiếp thay vì pipeline (không dùng StandardScaler)
    svc = SVC(probability=True, cache_size=2000, random_state=42)
    
    # Lưới tham số tối ưu cho MNIST
    param_grid = {
        'C': [1, 5, 10],
        'gamma': [0.01, 0.02, 0.05],
        'kernel': ['rbf']
    }
    
    # Sử dụng tập con cho GridSearch
    n_samples = min(config.GRID_SEARCH_SAMPLES, len(X_train))
    X_grid = X_train[:n_samples]
    y_grid = y_train[:n_samples]
    
    print(f"📊 Sử dụng {n_samples} mẫu cho GridSearch")
    
    grid_search = GridSearchCV(
        svc,
        param_grid,
        cv=config.GRID_CV,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy',
        return_train_score=True
    )
    
    start_time = time()
    grid_search.fit(X_grid, y_grid)
    grid_time = time() - start_time
    
    print(f"\n✅ GridSearch hoàn tất trong {grid_time:.2f} giây")
    print(f"\n📊 Kết quả GridSearch:")
    print(f"   - Best Score (CV): {grid_search.best_score_:.4f}")
    print(f"   - Best Parameters: {grid_search.best_params_}")
    
    # Chuyển đổi key để tương thích với code cũ
    best_params = {
        'svc__kernel': grid_search.best_params_['kernel'],
        'svc__C': grid_search.best_params_['C'],
        'svc__gamma': grid_search.best_params_['gamma']
    }
    
    return best_params


def plot_confusion_matrix(cm, output_path):
    """Vẽ và lưu ma trận nhầm lẫn."""
    plt.figure(figsize=(12, 10))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Tỷ lệ'})
    
    plt.title('Ma trận Nhầm lẫn (Normalized) - Mô hình SVM', fontsize=14)
    plt.xlabel('Dự đoán', fontsize=12)
    plt.ylabel('Thực tế', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {output_path}")


def save_outputs(model, X_test, y_test, results, config):
    """Lưu mô hình và đầu ra."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n💾 Lưu đầu ra...")
    
    # Lấy xác suất và dự đoán
    proba = model.predict_proba(X_test)
    pred = results['y_pred']
    
    # Lưu mô hình
    model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"✅ Đã lưu mô hình: {model_path}")
    
    # Lưu predictions cho ensemble
    ensemble_output = pd.DataFrame(proba, columns=[f'prob_digit_{i}' for i in range(10)])
    ensemble_output['predicted_label'] = pred
    ensemble_output['true_label'] = y_test
    
    csv_path = os.path.join(config.OUTPUT_DIR, config.PREDICTIONS_FILENAME)
    ensemble_output.to_csv(csv_path, index=False)
    print(f"✅ Đã lưu: {csv_path}")
    
    # Lưu numpy arrays
    np.save(os.path.join(config.OUTPUT_DIR, 'svm_probabilities.npy'), proba)
    np.save(os.path.join(config.OUTPUT_DIR, 'svm_predictions.npy'), pred)
    print(f"✅ Đã lưu numpy arrays")
    
    # Lưu confusion matrix
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], cm_path)
    
    return proba, pred


def predict_digit(model, image):
    """
    Dự đoán chữ số từ ảnh.
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Mô hình SVM đã huấn luyện
    image : array-like
        Ảnh đầu vào (28x28 hoặc 784,)
        
    Returns:
    --------
    dict : Kết quả dự đoán
    """
    # Flatten nếu cần
    if image.ndim == 2:
        image = image.reshape(1, -1)
    elif image.ndim == 1:
        image = image.reshape(1, -1)
    
    # Chuẩn hóa nếu cần
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    
    # Dự đoán
    pred = model.predict(image)[0]
    proba = model.predict_proba(image)[0]
    
    return {
        'prediction': pred,
        'confidence': proba[pred],
        'probabilities': proba
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Hàm chính."""
    parser = argparse.ArgumentParser(description='SVM Digit Recognition')
    parser.add_argument('--subset-size', type=int, default=10000,
                        help='Số mẫu train để sử dụng (default: 10000)')
    parser.add_argument('--use-full-data', action='store_true',
                        help='Sử dụng toàn bộ dữ liệu train')
    parser.add_argument('--skip-grid-search', action='store_true',
                        help='Bỏ qua Grid Search')
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['rbf', 'linear', 'poly', 'sigmoid'],
                        help='Loại kernel SVM (default: rbf)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Hệ số regularization C (default: 1.0)')
    parser.add_argument('--use-pca', action='store_true',
                        help='Sử dụng PCA giảm chiều')
    parser.add_argument('--pca-components', type=int, default=100,
                        help='Số thành phần PCA (default: 100)')
    
    args = parser.parse_args()
    
    # Cấu hình
    config = Config()
    config.SUBSET_SIZE = args.subset_size
    config.USE_SUBSET = not args.use_full_data
    config.USE_PCA = args.use_pca
    config.PCA_COMPONENTS = args.pca_components
    
    print("="*60)
    print("🔢 MÔ HÌNH SVM NHẬN DẠNG CHỮ SỐ VIẾT TAY")
    print("="*60)
    
    # Kiểm tra GPU
    gpu_available, cuml_available = check_gpu()
    print(f"\n📊 Cấu hình:")
    print(f"   - GPU Available: {gpu_available}")
    print(f"   - cuML Available: {cuml_available}")
    
    # 1. Tải dữ liệu
    print("\n" + "="*60)
    print("📦 BƯỚC 1: Chuẩn bị dữ liệu")
    print("="*60)
    
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # 2. Grid Search (tùy chọn)
    if not args.skip_grid_search:
        print("\n" + "="*60)
        print("⚙️ BƯỚC 2: Tối ưu hóa siêu tham số")
        print("="*60)
        
        best_params = run_grid_search(X_train, y_train, config)
        kernel = best_params['svc__kernel']
        C = best_params['svc__C']
        gamma = best_params.get('svc__gamma', 'scale')
    else:
        kernel = args.kernel
        C = args.C
        gamma = config.DEFAULT_GAMMA
    
    # 3. Huấn luyện mô hình cuối cùng
    print("\n" + "="*60)
    print("🏋️ BƯỚC 3: Huấn luyện mô hình cuối cùng")
    print("="*60)
    
    final_model, train_time = train_svm(
        X_train, y_train,
        kernel=kernel,
        C=C,
        gamma=gamma,
        use_pca=config.USE_PCA,
        n_components=config.PCA_COMPONENTS
    )
    
    # 4. Đánh giá
    print("\n" + "="*60)
    print("📊 BƯỚC 4: Đánh giá mô hình")
    print("="*60)
    
    results = evaluate_model(final_model, X_test, y_test, "Mô hình Cuối cùng")
    
    # 5. Lưu đầu ra
    print("\n" + "="*60)
    print("💾 BƯỚC 5: Lưu đầu ra")
    print("="*60)
    
    save_outputs(final_model, X_test, y_test, results, config)
    
    # Tổng kết
    print("\n" + "="*60)
    print("📊 TỔNG KẾT")
    print("="*60)
    print(f"\n🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"⏱️ Thời gian huấn luyện: {train_time:.2f} giây")
    print(f"\n📁 Các file đã lưu trong thư mục '{config.OUTPUT_DIR}/':")
    print(f"   - {config.MODEL_FILENAME}")
    print(f"   - {config.PREDICTIONS_FILENAME}")
    print(f"   - svm_probabilities.npy")
    print(f"   - svm_predictions.npy")
    print(f"   - confusion_matrix.png")
    print("\n✅ Hoàn tất!")
    
    return final_model, results


if __name__ == "__main__":
    model, results = main()

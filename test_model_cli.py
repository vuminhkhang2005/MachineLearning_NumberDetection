"""
🔢 Ứng dụng Test Model CLI (Command Line Interface)

Script này cho phép test model nhận dạng chữ số qua command line.

Sử dụng:
    # Test với mẫu MNIST ngẫu nhiên
    python test_model_cli.py
    
    # Test với file ảnh
    python test_model_cli.py --image path/to/image.png
    
    # Test nhiều mẫu MNIST
    python test_model_cli.py --samples 10
    
    # Hiển thị accuracy trên toàn bộ test set
    python test_model_cli.py --evaluate
"""

import argparse
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from time import time

# Đường dẫn model
MODEL_PATH = 'outputs/svm_digit_classifier.joblib'
FALLBACK_MODEL_PATH = 'svm_digit_classifier.joblib'


def load_model():
    """Tải model đã train."""
    if os.path.exists(MODEL_PATH):
        print(f"📥 Đang tải model từ {MODEL_PATH}...")
        return joblib.load(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        print(f"📥 Đang tải model từ {FALLBACK_MODEL_PATH}...")
        return joblib.load(FALLBACK_MODEL_PATH)
    else:
        print("⚠️ Không tìm thấy model đã train. Đang huấn luyện model mới...")
        return train_new_model()


def train_new_model():
    """Huấn luyện model mới nếu chưa có."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    
    print("📥 Đang tải dữ liệu MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    X = X.astype(np.float32) / 255.0
    
    # Sử dụng tập con để train nhanh
    X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, random_state=42, stratify=y)
    
    print("🏋️ Đang huấn luyện model SVM...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, cache_size=1000))
    ])
    model.fit(X_train, y_train)
    
    # Lưu model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Đã lưu model tại {MODEL_PATH}")
    
    return model


def load_and_preprocess_image(image_path):
    """Tải và tiền xử lý ảnh từ file."""
    from PIL import Image
    
    # Đọc ảnh
    img = Image.open(image_path).convert('L')
    
    # Resize về 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Chuyển sang numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Đảo ngược màu nếu cần (MNIST có nền đen, chữ trắng)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Chuẩn hóa
    img_array = img_array / 255.0
    
    return img_array


def predict_single(model, image, true_label=None, show_plot=True):
    """Dự đoán một ảnh và hiển thị kết quả."""
    # Flatten
    img_flat = image.reshape(1, -1)
    
    # Dự đoán
    prediction = model.predict(img_flat)[0]
    probabilities = model.predict_proba(img_flat)[0]
    confidence = probabilities[prediction]
    
    # In kết quả
    print(f"\n{'='*50}")
    print(f"🎯 Dự đoán: {prediction}")
    print(f"📊 Độ tin cậy: {confidence:.2%}")
    
    if true_label is not None:
        correct = prediction == true_label
        print(f"✅ Nhãn thực tế: {true_label}")
        print(f"{'✅ ĐÚNG!' if correct else '❌ SAI!'}")
    
    # Top 3 dự đoán
    print(f"\n📈 Top 3 dự đoán:")
    top3_idx = np.argsort(probabilities)[::-1][:3]
    for i, idx in enumerate(top3_idx):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"   {emoji} Chữ số {idx}: {probabilities[idx]:.2%}")
    
    # Hiển thị plot
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Ảnh
        axes[0].imshow(image.reshape(28, 28), cmap='gray')
        title = f'Dự đoán: {prediction}'
        if true_label is not None:
            title += f' (Thực tế: {true_label})'
        axes[0].set_title(title)
        axes[0].axis('off')
        
        # Biểu đồ xác suất
        colors = ['#e74c3c' if i == prediction else '#3498db' for i in range(10)]
        axes[1].bar(range(10), probabilities, color=colors)
        axes[1].set_xlabel('Chữ số')
        axes[1].set_ylabel('Xác suất')
        axes[1].set_title('Phân bố xác suất')
        axes[1].set_xticks(range(10))
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    return prediction, confidence


def test_random_samples(model, n_samples=5):
    """Test với các mẫu ngẫu nhiên từ MNIST."""
    from sklearn.datasets import fetch_openml
    
    print(f"\n📥 Đang tải dữ liệu MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    X = X.astype(np.float32) / 255.0
    
    # Lấy n mẫu ngẫu nhiên
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    correct = 0
    print(f"\n{'='*60}")
    print(f"🎲 Test với {n_samples} mẫu ngẫu nhiên từ MNIST")
    print(f"{'='*60}")
    
    # Hiển thị tất cả mẫu cùng lúc
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if n_samples == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        image = X[idx]
        true_label = y[idx]
        
        # Dự đoán
        img_flat = image.reshape(1, -1)
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]
        confidence = probabilities[prediction]
        
        is_correct = prediction == true_label
        if is_correct:
            correct += 1
        
        # In kết quả
        status = "✅" if is_correct else "❌"
        print(f"\nMẫu {i+1}: Thực tế={true_label}, Dự đoán={prediction} ({confidence:.1%}) {status}")
        
        # Hiển thị ảnh
        row, col = i // cols, i % cols
        axes[row, col].imshow(image.reshape(28, 28), cmap='gray')
        color = 'green' if is_correct else 'red'
        axes[row, col].set_title(f'Thực: {true_label}\nDự đoán: {prediction}', color=color)
        axes[row, col].axis('off')
    
    # Ẩn các subplot không dùng
    for i in range(n_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    accuracy = correct / n_samples
    print(f"\n{'='*60}")
    print(f"📊 Kết quả: {correct}/{n_samples} đúng ({accuracy:.1%})")
    print(f"{'='*60}")
    
    return accuracy


def evaluate_model(model):
    """Đánh giá model trên toàn bộ test set."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    
    print("\n📥 Đang tải dữ liệu MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    X = X.astype(np.float32) / 255.0
    
    # Chia dữ liệu
    _, X_test, _, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
    
    print(f"\n🔍 Đánh giá model trên {len(X_test)} mẫu test...")
    
    # Dự đoán
    start_time = time()
    y_pred = model.predict(X_test)
    predict_time = time() - start_time
    
    # Tính accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*60}")
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"⏱️ Thời gian dự đoán: {predict_time:.2f}s ({predict_time/len(X_test)*1000:.3f}ms/mẫu)")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Ma trận Nhầm lẫn (Normalized)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.show()
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test Model Nhận dạng Chữ số')
    parser.add_argument('--image', type=str, help='Đường dẫn đến file ảnh để test')
    parser.add_argument('--samples', type=int, default=5, help='Số mẫu MNIST ngẫu nhiên để test (default: 5)')
    parser.add_argument('--evaluate', action='store_true', help='Đánh giá model trên toàn bộ test set')
    parser.add_argument('--no-plot', action='store_true', help='Không hiển thị đồ thị')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔢 TEST MODEL NHẬN DẠNG CHỮ SỐ VIẾT TAY")
    print("="*60)
    
    # Tải model
    model = load_model()
    print("✅ Model đã sẵn sàng!")
    
    if args.image:
        # Test với file ảnh
        if not os.path.exists(args.image):
            print(f"❌ Không tìm thấy file: {args.image}")
            return
        
        print(f"\n📂 Đang tải ảnh: {args.image}")
        image = load_and_preprocess_image(args.image)
        predict_single(model, image, show_plot=not args.no_plot)
        
    elif args.evaluate:
        # Đánh giá trên test set
        evaluate_model(model)
        
    else:
        # Test với mẫu MNIST ngẫu nhiên
        test_random_samples(model, args.samples)


if __name__ == "__main__":
    main()

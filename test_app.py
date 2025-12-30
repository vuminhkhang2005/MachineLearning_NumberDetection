"""
🔢 Ứng dụng Test Model Nhận dạng Chữ số Viết tay

Ứng dụng này cho phép bạn:
1. Vẽ chữ số trực tiếp trên canvas
2. Upload ảnh chữ số
3. Xem kết quả dự đoán và xác suất

Sử dụng:
    python test_app.py

Sau đó mở trình duyệt và truy cập http://localhost:7860
"""

import gradio as gr
import numpy as np
import os
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
    from sklearn.svm import SVC
    
    print("📥 Đang tải dữ liệu MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y = y.astype(int)
    # Chuẩn hóa đơn giản về [0, 1] - KHÔNG dùng StandardScaler
    X = X.astype(np.float64) / 255.0
    
    # Sử dụng 30000 mẫu để train (cân bằng giữa tốc độ và độ chính xác)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=30000, random_state=42, stratify=y)
    
    print("🏋️ Đang huấn luyện model SVM...")
    print("   (Quá trình này có thể mất vài phút...)")
    
    # KHÔNG dùng Pipeline với StandardScaler - tránh vấn đề không khớp khi dự đoán
    model = SVC(
        kernel='rbf', 
        C=10.0,  # Tối ưu cho MNIST
        gamma=0.01,  # Tối ưu cho MNIST
        probability=True, 
        cache_size=2000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Lưu model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Đã lưu model tại {MODEL_PATH}")
    
    return model


def preprocess_image(image):
    """
    Tiền xử lý ảnh đầu vào để phù hợp với model MNIST.
    
    QUAN TRỌNG: MNIST có các đặc điểm sau:
    - Kích thước 28x28 pixels
    - Nền đen (0), chữ trắng (255)
    - Chữ số được căn giữa với bounding box
    - Giá trị pixel đã chuẩn hóa về [0, 1]
    
    Parameters:
    -----------
    image : PIL Image or numpy array
        Ảnh đầu vào
        
    Returns:
    --------
    numpy array : Ảnh đã xử lý (1, 784)
    """
    if image is None:
        return None
    
    # Chuyển sang PIL Image nếu cần
    if isinstance(image, np.ndarray):
        # Nếu là ảnh từ canvas (có thể là RGBA)
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Lấy alpha channel (nét vẽ nằm trong alpha)
                alpha = image[:, :, 3]
                # Hoặc chuyển sang grayscale
                img = Image.fromarray(image).convert('L')
                img_array_temp = np.array(img)
                # Kết hợp với alpha để lấy nét vẽ
                if alpha.max() > 0:
                    img_array_temp = alpha
                img = Image.fromarray(img_array_temp.astype(np.uint8))
            else:  # RGB
                img = Image.fromarray(image).convert('L')
        else:  # Grayscale
            img = Image.fromarray(image.astype(np.uint8))
    else:
        img = image.convert('L')
    
    # Chuyển sang numpy array
    img_array = np.array(img, dtype=np.float64)
    
    # Xác định xem nền sáng hay tối
    # MNIST có nền đen (0), chữ trắng (255)
    # Nếu giá trị trung bình của ảnh > 127, nghĩa là nền sáng -> đảo màu
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Tìm bounding box của chữ số và căn giữa (giống MNIST)
    # Điều này rất quan trọng để ảnh vẽ tay khớp với MNIST
    threshold = 20  # Ngưỡng để xác định pixel thuộc chữ
    coords = np.where(img_array > threshold)
    
    if len(coords[0]) > 0 and len(coords[1]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Cắt vùng chứa chữ số
        digit_region = img_array[y_min:y_max+1, x_min:x_max+1]
        
        # Resize digit region về 20x20 (MNIST để margin 4 pixel mỗi bên)
        digit_img = Image.fromarray(digit_region.astype(np.uint8))
        
        # Giữ tỷ lệ khung hình
        aspect = digit_region.shape[1] / digit_region.shape[0]
        if aspect > 1:  # Rộng hơn cao
            new_width = 20
            new_height = max(1, int(20 / aspect))
        else:  # Cao hơn rộng
            new_height = 20
            new_width = max(1, int(20 * aspect))
        
        digit_img = digit_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Tạo ảnh 28x28 với nền đen và đặt chữ số vào giữa
        final_array = np.zeros((28, 28), dtype=np.float64)
        
        # Tính vị trí để căn giữa
        y_offset = (28 - new_height) // 2
        x_offset = (28 - new_width) // 2
        
        # Đặt chữ số vào giữa
        final_array[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = np.array(digit_img)
        
        img_array = final_array
    else:
        # Nếu không tìm thấy chữ, resize đơn giản
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float64)
    
    # Chuẩn hóa về [0, 1] - QUAN TRỌNG: giống với cách train
    img_array = img_array / 255.0
    
    # Flatten thành vector 784 chiều
    img_flat = img_array.reshape(1, -1)
    
    return img_flat, img_array


def create_probability_chart(probabilities):
    """Tạo biểu đồ xác suất."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#3498db' if p < max(probabilities) else '#e74c3c' for p in probabilities]
    bars = ax.bar(range(10), probabilities, color=colors)
    
    ax.set_xlabel('Chữ số', fontsize=12)
    ax.set_ylabel('Xác suất', fontsize=12)
    ax.set_title('Phân bố xác suất dự đoán', fontsize=14)
    ax.set_xticks(range(10))
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên thanh
    for bar, prob in zip(bars, probabilities):
        if prob > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def predict_digit(image):
    """
    Dự đoán chữ số từ ảnh.
    
    Parameters:
    -----------
    image : PIL Image or numpy array
        Ảnh đầu vào
        
    Returns:
    --------
    tuple : (kết quả dự đoán, biểu đồ xác suất, ảnh đã xử lý)
    """
    if image is None:
        return "⚠️ Vui lòng vẽ hoặc upload một ảnh chữ số!", None, None
    
    try:
        # Tiền xử lý ảnh
        result = preprocess_image(image)
        if result is None:
            return "⚠️ Không thể xử lý ảnh!", None, None
            
        img_flat, img_display = result
        
        # Dự đoán
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]
        confidence = probabilities[prediction]
        
        # Tạo kết quả
        result_text = f"""
## 🎯 Kết quả Dự đoán

### Chữ số được nhận dạng: **{prediction}**

### Độ tin cậy: **{confidence:.1%}**

---

### Top 3 dự đoán:
"""
        # Lấy top 3
        top3_idx = np.argsort(probabilities)[::-1][:3]
        for i, idx in enumerate(top3_idx):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            result_text += f"\n{emoji} Chữ số **{idx}**: {probabilities[idx]:.1%}"
        
        # Tạo biểu đồ
        prob_chart = create_probability_chart(probabilities)
        
        # Tạo ảnh đã xử lý để hiển thị
        fig_processed, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img_display, cmap='gray')
        ax.set_title('Ảnh sau xử lý (28x28)')
        ax.axis('off')
        plt.tight_layout()
        
        return result_text, prob_chart, fig_processed
        
    except Exception as e:
        return f"❌ Lỗi: {str(e)}", None, None


def predict_from_canvas(canvas_data):
    """Xử lý dữ liệu từ canvas vẽ."""
    if canvas_data is None:
        return "⚠️ Vui lòng vẽ một chữ số!", None, None
    
    # Canvas data có thể là dict với key 'composite' hoặc trực tiếp là image
    if isinstance(canvas_data, dict):
        image = canvas_data.get('composite', None)
        if image is None:
            image = canvas_data.get('image', None)
    else:
        image = canvas_data
    
    return predict_digit(image)


def predict_from_upload(image):
    """Xử lý ảnh upload."""
    return predict_digit(image)


def test_with_mnist_sample():
    """Test với một mẫu từ MNIST."""
    from sklearn.datasets import fetch_openml
    
    print("📥 Đang tải một mẫu từ MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Lấy ngẫu nhiên một mẫu
    idx = np.random.randint(0, len(X))
    sample = X[idx].reshape(28, 28)
    true_label = int(y[idx])
    
    # Dự đoán
    img_flat = X[idx].reshape(1, -1).astype(np.float32) / 255.0
    prediction = model.predict(img_flat)[0]
    probabilities = model.predict_proba(img_flat)[0]
    confidence = probabilities[prediction]
    
    result_text = f"""
## 🎯 Test với mẫu MNIST

### Nhãn thực tế: **{true_label}**
### Dự đoán: **{prediction}**
### Độ tin cậy: **{confidence:.1%}**
### Kết quả: **{'✅ Đúng!' if prediction == true_label else '❌ Sai!'}**
"""
    
    # Tạo biểu đồ
    prob_chart = create_probability_chart(probabilities)
    
    # Tạo ảnh mẫu
    fig_sample, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(sample, cmap='gray')
    ax.set_title(f'Mẫu MNIST (Label: {true_label})')
    ax.axis('off')
    plt.tight_layout()
    
    return result_text, prob_chart, fig_sample


# ============================================================================
# TẢI MODEL
# ============================================================================

print("="*60)
print("🔢 ỨNG DỤNG TEST NHẬN DẠNG CHỮ SỐ VIẾT TAY")
print("="*60)

model = load_model()
print("✅ Model đã sẵn sàng!")


# ============================================================================
# TẠO GIAO DIỆN GRADIO
# ============================================================================

# CSS tùy chỉnh
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-title {
    text-align: center;
    color: #2c3e50;
}
"""

# Tạo giao diện
with gr.Blocks(css=custom_css, title="🔢 Test Nhận dạng Chữ số") as demo:
    gr.Markdown("""
    # 🔢 Ứng dụng Test Nhận dạng Chữ số Viết tay
    
    Ứng dụng sử dụng **mô hình SVM** được huấn luyện trên bộ dữ liệu **MNIST** để nhận dạng chữ số viết tay từ 0-9.
    
    ---
    """)
    
    with gr.Tabs():
        # Tab 1: Vẽ chữ số
        with gr.TabItem("✏️ Vẽ chữ số"):
            gr.Markdown("### Vẽ một chữ số (0-9) trên canvas bên dưới")
            
            with gr.Row():
                with gr.Column(scale=1):
                    canvas = gr.Sketchpad(
                        label="Vẽ chữ số tại đây",
                        brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
                        canvas_size=(280, 280),
                        type="numpy"
                    )
                    draw_btn = gr.Button("🔍 Nhận dạng", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(canvas, value="🗑️ Xóa")
                
                with gr.Column(scale=1):
                    draw_result = gr.Markdown(label="Kết quả")
                    draw_chart = gr.Plot(label="Biểu đồ xác suất")
                    draw_processed = gr.Plot(label="Ảnh đã xử lý")
            
            draw_btn.click(
                fn=predict_from_canvas,
                inputs=[canvas],
                outputs=[draw_result, draw_chart, draw_processed]
            )
        
        # Tab 2: Upload ảnh
        with gr.TabItem("📤 Upload ảnh"):
            gr.Markdown("### Upload một ảnh chữ số viết tay")
            gr.Markdown("*Lưu ý: Ảnh nên có nền sáng và chữ tối, hoặc ngược lại*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    upload_image = gr.Image(
                        label="Upload ảnh",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    upload_btn = gr.Button("🔍 Nhận dạng", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    upload_result = gr.Markdown(label="Kết quả")
                    upload_chart = gr.Plot(label="Biểu đồ xác suất")
                    upload_processed = gr.Plot(label="Ảnh đã xử lý")
            
            upload_btn.click(
                fn=predict_from_upload,
                inputs=[upload_image],
                outputs=[upload_result, upload_chart, upload_processed]
            )
        
        # Tab 3: Test với MNIST
        with gr.TabItem("🎲 Test với MNIST"):
            gr.Markdown("### Test với một mẫu ngẫu nhiên từ bộ dữ liệu MNIST")
            
            with gr.Row():
                with gr.Column(scale=1):
                    mnist_btn = gr.Button("🎲 Lấy mẫu ngẫu nhiên", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    mnist_result = gr.Markdown(label="Kết quả")
            
            with gr.Row():
                mnist_sample = gr.Plot(label="Mẫu MNIST")
                mnist_chart = gr.Plot(label="Biểu đồ xác suất")
            
            mnist_btn.click(
                fn=test_with_mnist_sample,
                inputs=[],
                outputs=[mnist_result, mnist_chart, mnist_sample]
            )
    
    gr.Markdown("""
    ---
    ### 📖 Hướng dẫn sử dụng:
    
    1. **Vẽ chữ số**: Sử dụng chuột để vẽ một chữ số trên canvas, sau đó nhấn "Nhận dạng"
    2. **Upload ảnh**: Tải lên một ảnh chữ số viết tay để nhận dạng
    3. **Test với MNIST**: Nhấn nút để test model với một mẫu ngẫu nhiên từ tập dữ liệu MNIST
    
    ### 📊 Thông tin model:
    - **Thuật toán**: Support Vector Machine (SVM)
    - **Kernel**: RBF (Radial Basis Function)
    - **Dữ liệu huấn luyện**: MNIST (60,000 ảnh chữ số viết tay)
    """)


# ============================================================================
# CHẠY ỨNG DỤNG
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Khởi động ứng dụng...")
    print("📍 Truy cập: http://localhost:7860")
    print("📍 Hoặc: http://0.0.0.0:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

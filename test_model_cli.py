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


def preprocess_digit_image(image_array, dilate_iterations=3, thin_stroke_mode=True, 
                           contrast_factor=1.5, debug=False):
    """
    Tiền xử lý ảnh chữ số viết tay (từ numpy array) để phù hợp với MNIST.
    
    ĐẶC BIỆT TỐI ƯU CHO NÉT BÚT MỎNG TRÊN GIẤY TRẮNG!
    
    Hàm này có thể được import và sử dụng từ các module khác:
        from test_model_cli import preprocess_digit_image, load_model
        
        model = load_model()
        processed = preprocess_digit_image(my_image_array)
        prediction = model.predict(processed.reshape(1, -1))[0]
    
    Parameters:
    -----------
    image_array : np.ndarray
        Ảnh đầu vào dạng numpy array (grayscale, bất kỳ kích thước)
    dilate_iterations : int
        Số lần làm dày nét chữ (mặc định 3, tăng lên 4-6 nếu nét rất mỏng)
    thin_stroke_mode : bool
        Bật chế độ xử lý nét mỏng đặc biệt (mặc định True)
    contrast_factor : float
        Hệ số tăng cường độ tương phản (mặc định 1.5, tăng nếu nét nhạt)
    debug : bool
        Hiển thị thông tin debug
        
    Returns:
    --------
    np.ndarray : Ảnh 28x28 đã xử lý, chuẩn hóa về [0, 1], dạng (28, 28)
    """
    from PIL import Image, ImageFilter, ImageOps, ImageEnhance
    
    # Đảm bảo là float64
    img_array = image_array.astype(np.float64)
    
    # Nếu có 3 kênh màu, chuyển sang grayscale
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    original_shape = img_array.shape
    
    if debug:
        print(f"📷 Kích thước ảnh gốc: {img_array.shape}")
        print(f"📊 Min/Max pixel: {img_array.min():.0f}/{img_array.max():.0f}")
        print(f"🔧 Chế độ nét mỏng: {'BẬT' if thin_stroke_mode else 'TẮT'}")
    
    # BƯỚC 1: TĂNG CƯỜNG ĐỘ TƯƠNG PHẢN
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    cutoff = 1 if thin_stroke_mode else 2
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    
    if contrast_factor > 1.0:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_factor)
    
    img_array = np.array(img_pil, dtype=np.float64)
    
    # BƯỚC 2: ĐẢO NGƯỢC MÀU NẾU NỀN SÁNG
    h, w = img_array.shape
    sample_points = [
        img_array[0, 0], img_array[0, -1], 
        img_array[-1, 0], img_array[-1, -1],
        img_array[0, w//2], img_array[-1, w//2],
        img_array[h//2, 0], img_array[h//2, -1]
    ]
    background_value = np.median(sample_points)
    
    if background_value > 127:
        img_array = 255 - img_array
        if debug:
            print("🔄 Đã đảo ngược màu (nền sáng -> nền đen)")
    
    # BƯỚC 3: XỬ LÝ NHIỄU
    if thin_stroke_mode:
        noise_threshold = 15
        img_array[img_array < noise_threshold] = 0
    
    # BƯỚC 4: LÀM MỊN VÀ KẾT NỐI NÉT ĐỨT
    if thin_stroke_mode:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
        img_array = np.array(img_pil, dtype=np.float64)
    
    # BƯỚC 5: LÀM DÀY NÉT CHỮ
    if dilate_iterations > 0:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        scale_factor = max(original_shape) / 200.0
        adjusted_iterations = max(dilate_iterations, int(dilate_iterations * scale_factor))
        adjusted_iterations = min(adjusted_iterations, 8)
        
        for _ in range(adjusted_iterations):
            img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        
        img_array = np.array(img_pil, dtype=np.float64)
        
        if debug:
            print(f"✏️ Đã làm dày nét {adjusted_iterations} lần")
    
    # BƯỚC 6: MORPHOLOGICAL CLOSING
    if thin_stroke_mode:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.MinFilter(size=3))
        img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        img_array = np.array(img_pil, dtype=np.float64)
    
    # BƯỚC 7: NGƯỠNG ĐỘNG
    if img_array.max() > 0:
        non_zero_pixels = img_array[img_array > 5]
        if len(non_zero_pixels) > 0:
            percentile = 20 if thin_stroke_mode else 30
            threshold = max(20, np.percentile(non_zero_pixels, percentile))
        else:
            threshold = 20
    else:
        threshold = 20
    
    # BƯỚC 8: TÌM BOUNDING BOX VÀ CĂN GIỮA
    coords = np.where(img_array > threshold)
    
    if len(coords[0]) > 0 and len(coords[1]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        padding = 8 if thin_stroke_mode else 5
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0] - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1] - 1, x_max + padding)
        
        digit_region = img_array[y_min:y_max+1, x_min:x_max+1]
        
        digit_img = Image.fromarray(digit_region.astype(np.uint8))
        
        h, w = digit_region.shape
        aspect = w / h
        if aspect > 1:
            new_width = 20
            new_height = max(1, int(20 / aspect))
        else:
            new_height = 20
            new_width = max(1, int(20 * aspect))
        
        digit_img = digit_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        final_array = np.zeros((28, 28), dtype=np.float64)
        y_offset = (28 - new_height) // 2
        x_offset = (28 - new_width) // 2
        
        resized_digit = np.array(digit_img, dtype=np.float64)
        final_array[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_digit
        
        img_array = final_array
    else:
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float64)
    
    # BƯỚC 9: LÀM DÀY THÊM SAU KHI RESIZE
    if thin_stroke_mode:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        img_array = np.array(img_pil, dtype=np.float64)
    
    # BƯỚC 10: CHUẨN HÓA VỀ [0, 1]
    if img_array.max() > 0:
        img_array = img_array / 255.0
    
    img_array = np.clip(img_array, 0, 1)
    
    if debug:
        print(f"✅ Ảnh cuối cùng: {img_array.shape}, range [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    return img_array


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


def load_and_preprocess_image(image_path, dilate_iterations=3, debug=False, 
                               thin_stroke_mode=True, contrast_factor=1.5):
    """
    Tải và tiền xử lý ảnh từ file để phù hợp với MNIST.
    
    ĐẶC BIỆT TỐI ƯU CHO NÉT BÚT MỎNG TRÊN GIẤY TRẮNG!
    
    QUAN TRỌNG: MNIST có các đặc điểm sau:
    - Kích thước 28x28 pixels
    - Nền đen (0), chữ trắng (255)
    - Chữ số được căn giữa với bounding box
    - Giá trị pixel đã chuẩn hóa về [0, 1]
    - NÉT CHỮ TƯƠNG ĐỐI DÀY (2-4 pixels trong 28x28)
    
    Parameters:
    -----------
    image_path : str
        Đường dẫn đến file ảnh
    dilate_iterations : int
        Số lần làm dày nét chữ (mặc định 3, tăng lên 4-5 nếu nét rất mỏng)
    debug : bool
        Hiển thị ảnh trung gian để debug
    thin_stroke_mode : bool
        Bật chế độ xử lý nét mỏng đặc biệt (mặc định True)
    contrast_factor : float
        Hệ số tăng cường độ tương phản (mặc định 1.5, tăng nếu nét nhạt)
        
    Returns:
    --------
    np.ndarray : Ảnh 28x28 đã xử lý, chuẩn hóa về [0, 1]
    """
    from PIL import Image, ImageFilter, ImageOps, ImageEnhance
    
    # Đọc ảnh và chuyển sang grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    original_shape = img_array.shape
    
    if debug:
        print(f"📷 Kích thước ảnh gốc: {img_array.shape}")
        print(f"📊 Min/Max pixel: {img_array.min():.0f}/{img_array.max():.0f}")
        print(f"📊 Mean pixel: {img_array.mean():.1f}")
        print(f"🔧 Chế độ nét mỏng: {'BẬT' if thin_stroke_mode else 'TẮT'}")
    
    # =====================================================================
    # BƯỚC 1: TĂNG CƯỜNG ĐỘ TƯƠNG PHẢN (ĐẶC BIỆT QUAN TRỌNG CHO NÉT MỎNG)
    # =====================================================================
    
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    
    # AutoContrast mạnh hơn cho nét mỏng
    cutoff = 1 if thin_stroke_mode else 2
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    
    # Tăng contrast thêm nếu cần (cho nét nhạt)
    if contrast_factor > 1.0:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_factor)
    
    img_array = np.array(img_pil, dtype=np.float64)
    
    if debug:
        print(f"📊 Sau tăng contrast (factor={contrast_factor}) - Min/Max: {img_array.min():.0f}/{img_array.max():.0f}")
    
    # =====================================================================
    # BƯỚC 2: ĐẢO NGƯỢC MÀU NẾU NỀN SÁNG (MNIST CÓ NỀN ĐEN)
    # =====================================================================
    
    # Kiểm tra nền: lấy giá trị ở các góc và cạnh
    h, w = img_array.shape
    sample_points = [
        img_array[0, 0], img_array[0, -1], 
        img_array[-1, 0], img_array[-1, -1],
        img_array[0, w//2], img_array[-1, w//2],
        img_array[h//2, 0], img_array[h//2, -1],
        # Thêm các điểm ở gần góc để chính xác hơn
        img_array[min(10, h-1), min(10, w-1)],
        img_array[min(10, h-1), max(0, w-11)],
        img_array[max(0, h-11), min(10, w-1)],
        img_array[max(0, h-11), max(0, w-11)]
    ]
    background_value = np.median(sample_points)
    
    if debug:
        print(f"📊 Giá trị nền (median mẫu): {background_value:.0f}")
    
    # Đảo màu nếu nền sáng (viết trên giấy trắng)
    is_light_background = background_value > 127
    if is_light_background:
        img_array = 255 - img_array
        if debug:
            print("🔄 Đã đảo ngược màu (nền sáng -> nền đen)")
    
    # =====================================================================
    # BƯỚC 3: XỬ LÝ NHIỄU VÀ CHUẨN BỊ CHO NÉT MỎNG
    # =====================================================================
    
    if thin_stroke_mode:
        # Với nét mỏng, ta cần loại bỏ nhiễu từ giấy nhưng giữ lại nét
        # Dùng ngưỡng thấp hơn để giữ nét mỏng
        noise_threshold = 15
        img_array[img_array < noise_threshold] = 0
        
        if debug:
            print(f"🧹 Đã loại bỏ nhiễu < {noise_threshold}")
    
    # =====================================================================
    # BƯỚC 4: LÀM MỊN VÀ KẾT NỐI NÉT ĐỨT (MORPHOLOGICAL CLOSING)
    # Rất quan trọng cho nét bút mỏng bị đứt đoạn!
    # =====================================================================
    
    if thin_stroke_mode:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        # Làm mờ nhẹ để kết nối các nét đứt gần nhau
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        img_array = np.array(img_pil, dtype=np.float64)
        
        if debug:
            print("🔗 Đã làm mờ nhẹ để kết nối nét đứt")
    
    # =====================================================================
    # BƯỚC 5: LÀM DÀY NÉT CHỮ (MORPHOLOGICAL DILATION)
    # Đây là bước QUAN TRỌNG NHẤT cho nét bút mỏng!
    # =====================================================================
    
    if dilate_iterations > 0:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        # Tính số lần dilate dựa trên kích thước ảnh
        # Ảnh lớn hơn cần dilate nhiều hơn
        scale_factor = max(original_shape) / 200.0  # 200 là kích thước tham chiếu
        adjusted_iterations = max(dilate_iterations, int(dilate_iterations * scale_factor))
        adjusted_iterations = min(adjusted_iterations, 8)  # Giới hạn tối đa
        
        if debug:
            print(f"✏️ Số lần dilate điều chỉnh: {adjusted_iterations} (gốc: {dilate_iterations})")
        
        # Dùng MaxFilter để làm dày nét (tương tự dilation)
        for i in range(adjusted_iterations):
            img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        
        img_array = np.array(img_pil, dtype=np.float64)
        
        if debug:
            print(f"✏️ Đã làm dày nét {adjusted_iterations} lần")
    
    # =====================================================================
    # BƯỚC 6: MORPHOLOGICAL CLOSING ĐỂ ĐÓNG CÁC LỖ NHỎ
    # Giúp tránh nhận nhầm thành số 8 (8 có nhiều lỗ/vòng tròn)
    # =====================================================================
    
    if thin_stroke_mode:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        # MinFilter sau MaxFilter = Closing operation
        # Giúp làm mịn cạnh và đóng các lỗ nhỏ
        img_pil = img_pil.filter(ImageFilter.MinFilter(size=3))
        img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        
        img_array = np.array(img_pil, dtype=np.float64)
        
        if debug:
            print("🔲 Đã áp dụng morphological closing")
    
    # =====================================================================
    # BƯỚC 7: NHẬN DIỆN NGƯỠNG VÀ LỌC NHIỄU CUỐI CÙNG
    # =====================================================================
    
    # Tính ngưỡng động dựa trên histogram
    if img_array.max() > 0:
        non_zero_pixels = img_array[img_array > 5]
        if len(non_zero_pixels) > 0:
            # Dùng ngưỡng thấp hơn cho nét mỏng
            percentile = 20 if thin_stroke_mode else 30
            threshold = max(20, np.percentile(non_zero_pixels, percentile))
        else:
            threshold = 20
    else:
        threshold = 20
    
    if debug:
        print(f"📊 Ngưỡng động: {threshold:.0f}")
    
    # =====================================================================
    # BƯỚC 8: TÌM BOUNDING BOX VÀ CĂN GIỮA
    # =====================================================================
    
    coords = np.where(img_array > threshold)
    
    if len(coords[0]) > 0 and len(coords[1]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Thêm padding nhỏ
        padding = 8 if thin_stroke_mode else 5
        y_min = max(0, y_min - padding)
        y_max = min(img_array.shape[0] - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img_array.shape[1] - 1, x_max + padding)
        
        # Cắt vùng chứa chữ số
        digit_region = img_array[y_min:y_max+1, x_min:x_max+1]
        
        if debug:
            print(f"📦 Bounding box: ({x_min}, {y_min}) -> ({x_max}, {y_max})")
            print(f"📦 Kích thước vùng chữ số: {digit_region.shape}")
        
        # Resize về 20x20 (MNIST để margin 4 pixel mỗi bên)
        digit_img = Image.fromarray(digit_region.astype(np.uint8))
        
        # Giữ tỷ lệ khung hình
        h, w = digit_region.shape
        aspect = w / h
        if aspect > 1:
            new_width = 20
            new_height = max(1, int(20 / aspect))
        else:
            new_height = 20
            new_width = max(1, int(20 * aspect))
        
        # Dùng LANCZOS cho chất lượng tốt
        digit_img = digit_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Tạo ảnh 28x28 với nền đen và đặt chữ số vào giữa
        final_array = np.zeros((28, 28), dtype=np.float64)
        
        y_offset = (28 - new_height) // 2
        x_offset = (28 - new_width) // 2
        
        resized_digit = np.array(digit_img, dtype=np.float64)
        final_array[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_digit
        
        img_array = final_array
    else:
        if debug:
            print("⚠️ Không tìm thấy chữ số, resize toàn bộ ảnh")
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float64)
    
    # =====================================================================
    # BƯỚC 9: LÀM DÀY THÊM SAU KHI RESIZE (CHO NÉT MỎNG)
    # =====================================================================
    
    if thin_stroke_mode:
        # Sau khi resize, nét có thể mỏng đi. Dilate thêm 1 lần
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.MaxFilter(size=3))
        img_array = np.array(img_pil, dtype=np.float64)
        
        if debug:
            print("✏️ Đã làm dày thêm sau resize")
    
    # =====================================================================
    # BƯỚC 10: CHUẨN HÓA VỀ [0, 1]
    # =====================================================================
    
    # Normalize về [0, 1]
    if img_array.max() > 0:
        img_array = img_array / 255.0
    
    # Clip để đảm bảo trong khoảng [0, 1]
    img_array = np.clip(img_array, 0, 1)
    
    if debug:
        print(f"✅ Ảnh cuối cùng: {img_array.shape}, range [{img_array.min():.3f}, {img_array.max():.3f}]")
        # Hiển thị mật độ pixel
        non_zero = np.count_nonzero(img_array > 0.1)
        print(f"📊 Mật độ nét: {non_zero}/{28*28} pixels ({non_zero/(28*28)*100:.1f}%)")
    
    return img_array


def predict_single(model, image, true_label=None, show_plot=True, original_image=None):
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
        # Nếu có ảnh gốc, hiển thị 3 panel
        if original_image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            
            # Ảnh gốc
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Ảnh gốc')
            axes[0].axis('off')
            
            # Ảnh đã xử lý
            axes[1].imshow(image.reshape(28, 28), cmap='gray')
            title = f'Sau xử lý → Dự đoán: {prediction}'
            if true_label is not None:
                title += f' (Thực tế: {true_label})'
            axes[1].set_title(title)
            axes[1].axis('off')
            
            # Biểu đồ xác suất
            colors = ['#e74c3c' if i == prediction else '#3498db' for i in range(10)]
            axes[2].bar(range(10), probabilities, color=colors)
            axes[2].set_xlabel('Chữ số')
            axes[2].set_ylabel('Xác suất')
            axes[2].set_title('Phân bố xác suất')
            axes[2].set_xticks(range(10))
            axes[2].set_ylim([0, 1])
        else:
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
    parser = argparse.ArgumentParser(
        description='Test Model Nhận dạng Chữ số',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test với ảnh viết tay nét bút mỏng trên giấy trắng (MẶC ĐỊNH)
  python test_model_cli.py --image my_digit.png
  
  # Nếu kết quả vẫn sai, tăng dilate và contrast
  python test_model_cli.py --image my_digit.png --dilate 5 --contrast 2.0
  
  # Tắt chế độ nét mỏng (cho ảnh nét đậm sẵn)
  python test_model_cli.py --image my_digit.png --no-thin-mode
  
  # Debug xem quá trình xử lý ảnh
  python test_model_cli.py --image my_digit.png --debug
  
  # Test với MNIST
  python test_model_cli.py --samples 10

LƯU Ý VỀ NÉT BÚT MỎNG:
  - Mặc định đã bật chế độ tối ưu cho nét mỏng trên giấy trắng
  - Nếu vẫn nhận sai (hay bị nhầm thành 8), thử:
    + Tăng --dilate lên 4-6
    + Tăng --contrast lên 1.8-2.5
    + Chụp ảnh rõ hơn, đủ sáng
        """
    )
    parser.add_argument('--image', type=str, help='Đường dẫn đến file ảnh để test')
    parser.add_argument('--samples', type=int, default=5, help='Số mẫu MNIST ngẫu nhiên để test (default: 5)')
    parser.add_argument('--evaluate', action='store_true', help='Đánh giá model trên toàn bộ test set')
    parser.add_argument('--no-plot', action='store_true', help='Không hiển thị đồ thị')
    parser.add_argument('--dilate', type=int, default=3, 
                        help='Số lần làm dày nét chữ (default: 3). Tăng lên 4-6 nếu nét bút RẤT mỏng')
    parser.add_argument('--contrast', type=float, default=1.5,
                        help='Hệ số tăng contrast (default: 1.5). Tăng lên 1.8-2.5 cho nét nhạt')
    parser.add_argument('--no-thin-mode', action='store_true',
                        help='Tắt chế độ xử lý nét mỏng (dùng cho ảnh nét đậm sẵn)')
    parser.add_argument('--debug', action='store_true', 
                        help='Hiển thị thông tin debug về quá trình xử lý ảnh')
    
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
        
        # Hiển thị cấu hình
        thin_mode = not args.no_thin_mode
        print(f"🔧 Cấu hình:")
        print(f"   - Chế độ nét mỏng: {'BẬT' if thin_mode else 'TẮT'}")
        print(f"   - Dilate iterations: {args.dilate}")
        print(f"   - Contrast factor: {args.contrast}")
        
        # Đọc ảnh gốc để hiển thị so sánh
        from PIL import Image
        original_img = Image.open(args.image).convert('L')
        original_array = np.array(original_img)
        
        # Tiền xử lý với các tham số mới
        image = load_and_preprocess_image(
            args.image, 
            dilate_iterations=args.dilate, 
            debug=args.debug,
            thin_stroke_mode=thin_mode,
            contrast_factor=args.contrast
        )
        
        # Dự đoán
        predict_single(model, image, show_plot=not args.no_plot, original_image=original_array)
        
        # Gợi ý cụ thể hơn
        print(f"\n💡 Gợi ý nếu kết quả sai:")
        print(f"   1. Tăng độ dày nét: --dilate 5 hoặc --dilate 6")
        print(f"   2. Tăng độ tương phản: --contrast 2.0 hoặc --contrast 2.5")
        print(f"   3. Kết hợp cả hai: --dilate 5 --contrast 2.0")
        print(f"   4. Dùng --debug để xem quá trình xử lý ảnh")
        
    elif args.evaluate:
        # Đánh giá trên test set
        evaluate_model(model)
        
    else:
        # Test với mẫu MNIST ngẫu nhiên
        test_random_samples(model, args.samples)


if __name__ == "__main__":
    main()

"""
🔢 Ứng dụng Desktop Test Model Nhận dạng Chữ số Viết tay

Ứng dụng này cho phép bạn:
1. Vẽ chữ số trực tiếp trên canvas
2. Upload ảnh chữ số từ máy tính
3. Xem kết quả dự đoán và xác suất

Sử dụng:
    python test_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import joblib
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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


class DigitRecognitionApp:
    """Ứng dụng Desktop nhận dạng chữ số viết tay."""
    
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("🔢 Nhận dạng Chữ số Viết tay")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        
        # Canvas size
        self.canvas_size = 280
        self.brush_size = 20
        
        # Số lần làm dày nét (dilation) - quan trọng cho nét bút mỏng
        # Mặc định 3 để xử lý tốt hơn nét mỏng trên giấy trắng
        self.dilate_iterations = tk.IntVar(value=3)
        
        # Image để vẽ (nền đen)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Biến lưu vị trí chuột trước đó
        self.last_x = None
        self.last_y = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Thiết lập giao diện."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔢 Nhận dạng Chữ số Viết tay", 
                                font=('Segoe UI', 18, 'bold'))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                   text="Vẽ một chữ số (0-9) trên canvas bên trái, sau đó nhấn 'Nhận dạng'",
                                   font=('Segoe UI', 10))
        subtitle_label.pack(pady=(0, 10))
        
        # Content frame (chứa canvas và kết quả)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame - Canvas vẽ
        left_frame = ttk.LabelFrame(content_frame, text="✏️ Vẽ chữ số", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Canvas để vẽ
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size,
                                bg='black', cursor='cross', highlightthickness=2,
                                highlightbackground='#3498db')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_on_canvas)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Buttons frame
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10, fill=tk.X)
        
        # Style cho buttons
        style = ttk.Style()
        style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'))
        style.configure('Secondary.TButton', font=('Segoe UI', 10))
        
        predict_btn = ttk.Button(btn_frame, text="🔍 Nhận dạng", 
                                 command=self.predict, style='Primary.TButton')
        predict_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        clear_btn = ttk.Button(btn_frame, text="🗑️ Xóa", 
                               command=self.clear_canvas, style='Secondary.TButton')
        clear_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Thêm hàng nút thứ hai
        btn_frame2 = ttk.Frame(left_frame)
        btn_frame2.pack(pady=5, fill=tk.X)
        
        upload_btn = ttk.Button(btn_frame2, text="📂 Tải ảnh lên", 
                                command=self.upload_image, style='Secondary.TButton')
        upload_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        test_btn = ttk.Button(btn_frame2, text="🎲 Test MNIST", 
                              command=self.test_mnist_sample, style='Secondary.TButton')
        test_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Slider để điều chỉnh độ dày nét (dilation) - quan trọng cho nét bút mỏng
        dilate_frame = ttk.LabelFrame(left_frame, text="✏️ Độ dày nét (cho ảnh upload)", padding="5")
        dilate_frame.pack(pady=5, fill=tk.X)
        
        dilate_label = ttk.Label(dilate_frame, 
                                 text="Tăng lên 4-6 nếu nét bút MỎNG trên giấy trắng:")
        dilate_label.pack()
        
        dilate_slider = ttk.Scale(dilate_frame, from_=0, to=8, 
                                  variable=self.dilate_iterations, 
                                  orient=tk.HORIZONTAL)
        dilate_slider.pack(fill=tk.X, padx=5)
        
        self.dilate_value_label = ttk.Label(dilate_frame, text="Mức: 3 (mặc định)")
        self.dilate_value_label.pack()
        
        def update_dilate_label(*args):
            val = self.dilate_iterations.get()
            hint = ""
            if val <= 2:
                hint = " (nét đậm)"
            elif val <= 4:
                hint = " (bình thường)"
            else:
                hint = " (nét rất mỏng)"
            self.dilate_value_label.config(text=f"Mức: {val}{hint}")
        
        self.dilate_iterations.trace_add("write", update_dilate_label)
        
        # Right frame - Kết quả
        right_frame = ttk.LabelFrame(content_frame, text="📊 Kết quả", padding="10")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Result text
        self.result_label = ttk.Label(right_frame, text="Vẽ một chữ số và nhấn 'Nhận dạng'",
                                      font=('Segoe UI', 12), wraplength=400)
        self.result_label.pack(pady=(0, 10))
        
        # Prediction display
        self.prediction_frame = ttk.Frame(right_frame)
        self.prediction_frame.pack(pady=10)
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="?", 
                                          font=('Segoe UI', 72, 'bold'),
                                          foreground='#3498db')
        self.prediction_label.pack()
        
        self.confidence_label = ttk.Label(self.prediction_frame, text="",
                                          font=('Segoe UI', 14))
        self.confidence_label.pack()
        
        # Chart frame
        self.chart_frame = ttk.Frame(right_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Processed image frame
        processed_frame = ttk.LabelFrame(left_frame, text="Ảnh sau xử lý (28x28)", padding="5")
        processed_frame.pack(pady=10)
        
        self.processed_label = ttk.Label(processed_frame)
        self.processed_label.pack()
    
    def start_draw(self, event):
        """Bắt đầu vẽ."""
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        """Vẽ trên canvas."""
        if self.last_x and self.last_y:
            # Vẽ trên Tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill='white', width=self.brush_size, 
                                    capstyle=tk.ROUND, smooth=True)
            
            # Vẽ trên PIL Image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=255, width=self.brush_size)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, event):
        """Dừng vẽ."""
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """Xóa canvas."""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Reset kết quả
        self.prediction_label.config(text="?", foreground='#3498db')
        self.confidence_label.config(text="")
        self.result_label.config(text="Vẽ một chữ số và nhấn 'Nhận dạng'")
        
        # Clear chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Clear processed image
        self.processed_label.config(image='')
    
    def preprocess_image(self, img_array):
        """Tiền xử lý ảnh để khớp với MNIST."""
        # Tìm bounding box của chữ số
        threshold = 20
        coords = np.where(img_array > threshold)
        
        if len(coords[0]) > 0 and len(coords[1]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Cắt vùng chứa chữ số
            digit_region = img_array[y_min:y_max+1, x_min:x_max+1]
            
            # Resize về 20x20
            digit_img = Image.fromarray(digit_region.astype(np.uint8))
            
            # Giữ tỷ lệ
            aspect = digit_region.shape[1] / digit_region.shape[0]
            if aspect > 1:
                new_width = 20
                new_height = max(1, int(20 / aspect))
            else:
                new_height = 20
                new_width = max(1, int(20 * aspect))
            
            digit_img = digit_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Tạo ảnh 28x28 với nền đen và đặt chữ số vào giữa
            final_array = np.zeros((28, 28), dtype=np.float64)
            
            y_offset = (28 - new_height) // 2
            x_offset = (28 - new_width) // 2
            
            final_array[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = np.array(digit_img)
            
            return final_array
        else:
            # Resize đơn giản
            img = Image.fromarray(img_array.astype(np.uint8))
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            return np.array(img, dtype=np.float64)
    
    def predict(self):
        """Dự đoán chữ số."""
        # Lấy ảnh từ PIL Image
        img_array = np.array(self.image, dtype=np.float64)
        
        # Kiểm tra xem có vẽ gì không
        if img_array.max() < 10:
            messagebox.showwarning("Cảnh báo", "Vui lòng vẽ một chữ số trước!")
            return
        
        # Tiền xử lý
        processed = self.preprocess_image(img_array)
        
        # Chuẩn hóa và flatten
        img_flat = (processed / 255.0).reshape(1, -1)
        
        # Dự đoán
        prediction = self.model.predict(img_flat)[0]
        probabilities = self.model.predict_proba(img_flat)[0]
        confidence = probabilities[prediction]
        
        # Hiển thị kết quả
        self.prediction_label.config(text=str(prediction), foreground='#27ae60')
        self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1%}")
        
        # Top 3
        top3_idx = np.argsort(probabilities)[::-1][:3]
        result_text = "Top 3 dự đoán:\n"
        for i, idx in enumerate(top3_idx):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            result_text += f"{emoji} Chữ số {idx}: {probabilities[idx]:.1%}\n"
        self.result_label.config(text=result_text)
        
        # Hiển thị biểu đồ
        self.show_probability_chart(probabilities)
        
        # Hiển thị ảnh đã xử lý
        self.show_processed_image(processed)
    
    def show_probability_chart(self, probabilities):
        """Hiển thị biểu đồ xác suất."""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=80)
        
        colors = ['#3498db' if p < max(probabilities) else '#e74c3c' for p in probabilities]
        bars = ax.bar(range(10), probabilities, color=colors)
        
        ax.set_xlabel('Chữ số', fontsize=9)
        ax.set_ylabel('Xác suất', fontsize=9)
        ax.set_xticks(range(10))
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Embed vào Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.close(fig)
    
    def show_processed_image(self, processed):
        """Hiển thị ảnh đã xử lý."""
        # `processed` có thể là:
        # - uint8/float trong thang 0..255 (pipeline canvas cũ)
        # - float trong thang 0..1 (pipeline OpenCV mới cho ảnh upload)
        arr = np.asarray(processed)
        if arr.size == 0:
            return
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)

        # Scale lên để dễ nhìn
        img = Image.fromarray(arr)
        img = img.resize((84, 84), Image.Resampling.NEAREST)
        
        photo = ImageTk.PhotoImage(img)
        self.processed_label.config(image=photo)
        self.processed_label.image = photo  # Giữ reference
    
    def test_mnist_sample(self):
        """Test với mẫu ngẫu nhiên từ MNIST."""
        from sklearn.datasets import fetch_openml
        
        self.result_label.config(text="Đang tải mẫu MNIST...")
        self.root.update()
        
        try:
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
            
            # Lấy ngẫu nhiên một mẫu
            idx = np.random.randint(0, len(X))
            sample = X[idx].reshape(28, 28)
            true_label = int(y[idx])
            
            # Dự đoán
            img_flat = X[idx].reshape(1, -1).astype(np.float64) / 255.0
            prediction = self.model.predict(img_flat)[0]
            probabilities = self.model.predict_proba(img_flat)[0]
            confidence = probabilities[prediction]
            
            # Hiển thị trên canvas
            self.clear_canvas()
            
            # Scale sample lên để vẽ trên canvas
            sample_scaled = Image.fromarray(sample.astype(np.uint8))
            sample_scaled = sample_scaled.resize((self.canvas_size, self.canvas_size), 
                                                  Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(sample_scaled)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Giữ reference
            
            # Hiển thị kết quả
            is_correct = prediction == true_label
            color = '#27ae60' if is_correct else '#e74c3c'
            self.prediction_label.config(text=str(prediction), foreground=color)
            self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1%}")
            
            result_text = f"Nhãn thực tế: {true_label}\n"
            result_text += f"Dự đoán: {prediction}\n"
            result_text += f"Kết quả: {'✅ Đúng!' if is_correct else '❌ Sai!'}"
            self.result_label.config(text=result_text)
            
            # Hiển thị biểu đồ
            self.show_probability_chart(probabilities)
            
            # Hiển thị ảnh đã xử lý
            self.show_processed_image(sample)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải MNIST: {str(e)}")
    
    def upload_image(self):
        """Upload và nhận dạng ảnh từ máy tính."""
        # Mở dialog chọn file
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ số",
            filetypes=file_types,
            initialdir=os.getcwd()
        )
        
        if not file_path:
            return  # Người dùng hủy
        
        try:
            self.result_label.config(text=f"Đang xử lý: {os.path.basename(file_path)}...")
            self.root.update()
            
            # Đọc và tiền xử lý ảnh (sử dụng dilate_iterations từ slider)
            processed = self.load_and_preprocess_uploaded_image(
                file_path, 
                dilate_iterations=self.dilate_iterations.get()
            )
            
            # `processed` đã là (28,28) và chuẩn hóa về [0,1] để khớp MNIST
            img_flat = processed.reshape(1, -1)
            
            # Dự đoán
            prediction = self.model.predict(img_flat)[0]
            probabilities = self.model.predict_proba(img_flat)[0]
            confidence = probabilities[prediction]
            
            # Xóa canvas và hiển thị ảnh đã upload
            self.clear_canvas()
            
            # Hiển thị ảnh gốc (scale để fit canvas)
            original_img = Image.open(file_path).convert('L')
            # Scale để fit vào canvas nhưng giữ tỷ lệ
            orig_w, orig_h = original_img.size
            scale = min(self.canvas_size / orig_w, self.canvas_size / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            original_scaled = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Tạo ảnh nền đen với ảnh ở giữa
            display_img = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
            x_offset = (self.canvas_size - new_w) // 2
            y_offset = (self.canvas_size - new_h) // 2
            display_img.paste(original_scaled, (x_offset, y_offset))
            
            photo = ImageTk.PhotoImage(display_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Giữ reference
            
            # Hiển thị kết quả
            self.prediction_label.config(text=str(prediction), foreground='#27ae60')
            self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1%}")
            
            # Top 3
            top3_idx = np.argsort(probabilities)[::-1][:3]
            result_text = f"📂 File: {os.path.basename(file_path)}\n\n"
            result_text += "Top 3 dự đoán:\n"
            for i, idx in enumerate(top3_idx):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                result_text += f"{emoji} Chữ số {idx}: {probabilities[idx]:.1%}\n"
            self.result_label.config(text=result_text)
            
            # Hiển thị biểu đồ
            self.show_probability_chart(probabilities)
            
            # Hiển thị ảnh đã xử lý (28x28)
            self.show_processed_image(processed)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")
    
    def load_and_preprocess_uploaded_image(self, image_path, dilate_iterations=3):
        """
        Tải và tiền xử lý ảnh từ file để phù hợp với MNIST.
        
        ĐẶC BIỆT TỐI ƯU CHO NÉT BÚT MỎNG TRÊN GIẤY TRẮNG!
        
        Sử dụng thuật toán mới với:
        - Otsu thresholding tự động
        - Binarization mạnh
        - Morphological operations đúng thứ tự
        
        Parameters:
        -----------
        image_path : str
            Đường dẫn đến file ảnh
        dilate_iterations : int
            Số lần làm dày nét chữ (mặc định 3, tăng lên 4-6 nếu nét rất mỏng)
        """
        # Pipeline mới (OpenCV): adaptive threshold + contour crop + deskew + center-of-mass
        try:
            from mnist_preprocessing import PreprocessParams, preprocess_digit_to_mnist

            params = PreprocessParams(
                dilate_iterations=int(dilate_iterations),
                pad_px=10 if dilate_iterations >= 4 else 8,
                adaptive_block_size=31,
                adaptive_C=10,
                deskew=True,
            )
            img = Image.open(image_path)
            img_array = np.array(img)
            return preprocess_digit_to_mnist(img_array, params=params, debug=False)
        except Exception:
            # Fallback: dùng lại hàm preprocess của CLI (PIL pipeline cũ)
            from PIL import Image
            from test_model_cli import preprocess_digit_image

            img = Image.open(image_path).convert("L")
            img_array = np.array(img, dtype=np.float64)
            processed = preprocess_digit_image(
                img_array,
                dilate_iterations=dilate_iterations,
                thin_stroke_mode=True,
                contrast_factor=1.5,
                debug=False,
            )
            return processed


# ============================================================================
# CHẠY ỨNG DỤNG
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🔢 ỨNG DỤNG DESKTOP NHẬN DẠNG CHỮ SỐ VIẾT TAY")
    print("="*60)
    
    # Tải model
    model = load_model()
    print("✅ Model đã sẵn sàng!")
    
    # Tạo và chạy ứng dụng
    print("\n🚀 Khởi động ứng dụng desktop...")
    root = tk.Tk()
    app = DigitRecognitionApp(root, model)
    root.mainloop()

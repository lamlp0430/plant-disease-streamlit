import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# --- "BỘ NÃO" AI (Load Model) ---
@st.cache_resource
def load_model():
    model_path = "best.onnx"  # <--- Đổi tên file thành .onnx
    
    if not os.path.exists(model_path):
        st.error("❌ Không tìm thấy file 'best.onnx'. Vui lòng upload file lên!")
        return None
    
    # Load model ONNX (thêm task='detect' để chắc chắn)
    model = YOLO(model_path, task="detect") 
    
    print(f"🚀 Đã load thành công model: {model_path}")
    return model

model = load_model()
# ----------------------

st.set_page_config(layout="wide")
st.title("✅ PHIÊN BẢN 40 (STREAMLIT CLOUD)!")
st.info("Code này 'triệt để' chạy trên 'Streamlit Cloud' (Có đủ RAM).")

# --- KHỞI TẠO "TRẠNG THÁI" (STATE) ---
if 'result_image_array' not in st.session_state:
    st.session_state.result_image_array = None
if 'input_image_pil' not in st.session_state:
    st.session_state.input_image_pil = None
# (CODE MỚI) Thêm state để lưu kết quả thô
if 'raw_results' not in st.session_state:
    st.session_state.raw_results = None
# ------------------------------------

st.title("Plant Disease Detector (MỘT APP) 🍃")
col1, col2 = st.columns(2)

with col1:
    st.header("Bước 1: Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

    predict_button = st.button("Bắt đầu Dự đoán (Predict)", type="primary")

    if predict_button:
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            st.session_state.input_image_pil = image_pil

            with st.spinner("'Bộ não' (Cục bộ) đang phân tích ảnh..."):
                try:
                    results = model.predict(source=image_pil, device='cpu', save=False)
                    # (CODE MỚI) Lưu kết quả thô vào state
                    st.session_state.raw_results = results 
                    
                    result_array = results[0].plot()
                    st.session_state.result_image_array = result_array
                except Exception as e:
                    st.error(f"Lỗi khi chạy 'Bộ não' Cục bộ: {e}")
                    st.session_state.result_image_array = None
                    st.session_state.raw_results = None # (CODE MỚI)
        else:
            st.warning("Vui lòng tải ảnh lên trước khi nhấn nút 'Bắt đầu Dự đoán'.")
            st.session_state.result_image_array = None
            st.session_state.raw_results = None # (CODE MỚI)

with col2:
    st.header("Bước 2: Kết quả")

    if st.session_state.result_image_array is not None:
        st.image(st.session_state.input_image_pil, caption="Ảnh bạn vừa tải lên.", use_container_width=True)
        st.divider() 
        st.image(st.session_state.result_image_array, caption="Ảnh kết quả từ 'Bộ não' (Cục bộ).", use_container_width=True)
        
        # === (TOÀN BỘ BLOK CODE MỚI BẮT ĐẦU TỪ ĐÂY) ===
        st.divider()
        st.subheader("🔍 Chi tiết phát hiện:")

        # 1. Lấy kết quả thô từ session state
        results = st.session_state.raw_results
        
        # 2. Lấy kết quả cho ảnh đầu tiên
        result = results[0]
        
        # 3. Lấy danh sách tên bệnh (class names) từ model
        class_names = model.names

        # 4. Lặp qua từng "box" (khung) phát hiện được
        if len(result.boxes) == 0:
            st.success("✅ Không phát hiện thấy bệnh.")
        else:
            for box in result.boxes:
                # Lấy tên bệnh từ ID
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                
                # Lấy thông số "Độ tin cậy" (Confidence)
                confidence = float(box.conf[0])
                
                # Lấy "Tọa độ" [x1, y1, x2, y2]
                coords = box.xyxy[0]
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                
                # Hiển thị tất cả thông tin
                st.markdown(f"**Tên bệnh:** `{class_name}`")
                st.markdown(f"**Độ tin cậy:** `{confidence:.2f}`") # Làm tròn 2 chữ số
                st.markdown(f"**Tọa độ [x1, y1, x2, y2]:** `[{x1}, {y1}, {x2}, {y2}]`")
                st.markdown("---") # Thêm một đường kẻ ngang
        # === (KẾT THÚC BLOK CODE MỚI) ===
        
    else:
        st.info("Kết quả dự đoán sẽ hiện ở đây sau khi bạn nhấn nút 'Bắt đầu Dự đoán'.")

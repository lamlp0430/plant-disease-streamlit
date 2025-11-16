import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# --- "BỘ NÃO" (GỘP) ---
@st.cache_resource # "Triệt để" cache "Bộ não"
def load_model():
    print("Loading YOLOv5 model (Cục bộ)...")
    model = YOLO("best.pt") # <--- "TRIỆT ĐỂ" TẢI "BỘ NÃO"
    print("✅ Model loaded successfully (Cục bộ)!")
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
                    result_array = results[0].plot()
                    st.session_state.result_image_array = result_array
                except Exception as e:
                    st.error(f"Lỗi khi chạy 'Bộ não' Cục bộ: {e}")
                    st.session_state.result_image_array = None
        else:
            st.warning("Vui lòng tải ảnh lên trước khi nhấn nút 'Bắt đầu Dự đoán'.")
            st.session_state.result_image_array = None

with col2:
    st.header("Bước 2: Kết quả")

    if st.session_state.result_image_array is not None:
        st.image(st.session_state.input_image_pil, caption="Ảnh bạn vừa tải lên.", use_container_width=True)
        st.divider() 
        st.image(st.session_state.result_image_array, caption="Ảnh kết quả từ 'Bộ não' (Cục bộ).", use_container_width=True)
    else:
        st.info("Kết quả dự đoán sẽ hiện ở đây sau khi bạn nhấn nút 'Bắt đầu Dự đoán'.")
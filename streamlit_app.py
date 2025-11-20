import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL (ONNX VERSION) ---
@st.cache_resource
def load_model():
    # Sử dụng file ONNX để tránh lỗi version
    model_path = "best.onnx" 
    
    if not os.path.exists(model_path):
        st.error("❌ Không tìm thấy file 'best.onnx'. Vui lòng kiểm tra lại!")
        return None
    
    # Load model với task='detect'
    try:
        model = YOLO(model_path, task="detect")
        print(f"✅ Đã load thành công model: {model_path}")
        return model
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None

model = load_model()

# --- CƠ SỞ DỮ LIỆU GIẢI PHÁP ---
solutions_db = {
    "Tomato___Leaf_Mold": {
        "mo_ta": "Bệnh mốc lá cà chua xuất hiện các đốm màu vàng nhạt ở mặt trên lá, mặt dưới có lớp nấm mốc.",
        "dieu_tri": "✔ Tỉa bớt lá già.\n✔ Sử dụng thuốc diệt nấm gốc Đồng (Copper).",
        "phong_ngua": "Tưới nước vào gốc, tránh làm ướt lá."
    },
    "Tomato___Bacterial_spot": {
        "mo_ta": "Đốm vi khuẩn gây ra các đốm nhỏ, sũng nước, sau chuyển sang màu nâu đen.",
        "dieu_tri": "✔ Loại bỏ cây bị bệnh.\n✔ Phun thuốc chứa Đồng (Copper).",
        "phong_ngua": "Sử dụng hạt giống sạch bệnh."
    },
    # ... (Bạn hãy bổ sung thêm các bệnh khác vào đây) ...
}

# --- SIDEBAR (THANH CÔNG CỤ BÊN TRÁI) ---
with st.sidebar:
    st.title("🌿 Plant Doctor AI")
    st.caption("Hệ thống chẩn đoán bệnh cây trồng (YOLOv10)")
    st.divider()

    # 1. Chế độ Camera
    st.subheader("📸 Cấu hình")
    use_camera = st.toggle("Sử dụng Camera trực tiếp", False)
    
    st.divider()

    # 2. Thanh trượt Ngưỡng tin cậy (ĐÃ KHÔI PHỤC)
    st.subheader("🎛️ Độ nhạy AI")
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (Confidence)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Kéo thấp để tìm tất cả các bệnh (dễ báo nhầm). Kéo cao để chính xác tuyệt đối."
    )
    
    # Hiển thị trạng thái
    if confidence_threshold < 0.3:
        st.warning("⚠️ Chế độ nhạy cao (Tầm soát)")
    elif confidence_threshold > 0.7:
        st.info("ℹ️ Chế độ khắt khe (Chính xác)")
    
    st.divider()

    # 3. Danh sách bệnh
    with st.expander("📝 Danh sách bệnh hỗ trợ"):
        if model and hasattr(model, 'names'):
            disease_list = list(model.names.values())
            disease_list.sort()
            for d in disease_list:
                st.markdown(f"- {d}")

# --- GIAO DIỆN CHÍNH ---
st.title("🍃 Plant Disease Detector")
st.markdown("**Hệ thống hỗ trợ nông nghiệp 4.0**")

col1, col2 = st.columns([1, 1.2], gap="large")

# --- CỘT 1: INPUT ---
with col1:
    st.header("1️⃣ Cung cấp hình ảnh")
    
    image_source = None
    if use_camera:
        camera_input = st.camera_input("Chụp ảnh lá cây", key="cam")
        image_source = camera_input
    else:
        upload_input = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"], key="upload")
        image_source = upload_input

    predict_btn = st.button("🔍 Phân tích ngay", type="primary", use_container_width=True)

# --- LOGIC DỰ ĐOÁN ---
if predict_btn and image_source:
    image_pil = Image.open(image_source)
    # Resize nhẹ để tăng tốc độ
    image_pil.thumbnail((1024, 1024)) 

    with st.spinner("AI đang phân tích..."):
        if model:
            # Chạy dự đoán với ngưỡng thấp nhất để lấy hết kết quả
            results = model.predict(image_pil, conf=0.05) 
            result = results[0]
            
            # LỌC KẾT QUẢ THEO THANH TRƯỢT
            detected_boxes = []
            for box in result.boxes:
                if float(box.conf[0]) >= confidence_threshold:
                    detected_boxes.append(box)
            
            # Vẽ lại ảnh
            res_plotted = result.plot(conf=confidence_threshold)
            
            # Lưu session
            st.session_state['result_img'] = res_plotted
            st.session_state['input_img'] = image_pil
            st.session_state['boxes'] = detected_boxes
            st.session_state['names'] = model.names

# --- CỘT 2: KẾT QUẢ ---
with col2:
    st.header("2️⃣ Kết quả chẩn đoán")

    if 'result_img' in st.session_state:
        boxes = st.session_state['boxes']
        names = st.session_state['names']

        if len(boxes) == 0:
            st.success("🎉 Cây có vẻ khỏe mạnh (hoặc chưa phát hiện bệnh ở ngưỡng này).")
            st.image(st.session_state['input_img'], use_container_width=True)
            st.balloons()
        else:
            st.error(f"⚠️ Phát hiện {len(boxes)} vị trí nhiễm bệnh!")

            # Tabs hiển thị
            tab_img, tab_detail, tab_solution = st.tabs(["🖼️ Trực quan", "📋 Chi tiết", "💊 Giải pháp"])

            with tab_img:
                st.image(st.session_state['result_img'], caption=f"Độ tin cậy > {confidence_threshold*100:.0f}%", use_container_width=True)

            with tab_detail:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.info(f"**{names[cls_id]}** - {conf*100:.1f}%")

            with tab_solution:
                unique_diseases = set([names[int(box.cls[0])] for box in boxes])
                for disease_name in unique_diseases:
                    st.markdown(f"### {disease_name}")
                    if disease_name in solutions_db:
                        sol = solutions_db[disease_name]
                        st.write(f"**Mô tả:** {sol['mo_ta']}")
                        st.write(f"**Điều trị:** {sol['dieu_tri']}")
                    else:
                        st.warning("Chưa có dữ liệu giải pháp chi tiết.")
                    st.divider()
    else:
        st.info("👈 Vui lòng tải ảnh để bắt đầu.")

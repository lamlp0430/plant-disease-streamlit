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
    # Sử dụng file ONNX để chạy ổn định trên mọi nền tảng
    model_path = "best.onnx" 
    
    if not os.path.exists(model_path):
        st.error("❌ Không tìm thấy file 'best.onnx'. Vui lòng upload file lên!")
        return None
    
    try:
        # Load model với task='detect'
        model = YOLO(model_path, task="detect")
        print(f"✅ Đã load thành công model: {model_path}")
        return model
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None

model = load_model()

# --- CƠ SỞ DỮ LIỆU GIẢI PHÁP (FULL 38 BỆNH) ---
solutions_db = {
    "Apple___Apple_scab": {
        "mo_ta": "Nấm gây đốm sậm trên lá và quả táo.",
        "dieu_tri": "Cắt bỏ lá bệnh, phun thuốc gốc đồng hoặc mancozeb.",
        "phong_ngua": "Tỉa tán thông thoáng, tránh để ẩm kéo dài."
    },
    "Apple___Black_rot": {
        "mo_ta": "Thối đen trên quả và vết loét trên cành.",
        "dieu_tri": "Loại bỏ quả/cành bệnh, phun thuốc trị nấm chlorothalonil.",
        "phong_ngua": "Vệ sinh vườn và cắt tỉa hàng năm."
    },
    "Apple___Cedar_apple_rust": {
        "mo_ta": "Đốm cam vàng trên lá do nấm từ cây tuyết tùng.",
        "dieu_tri": "Phun fungicide nhóm triazole.",
        "phong_ngua": "Tránh trồng gần cây tuyết tùng; cắt lá bệnh."
    },
    "Apple___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Duy trì chăm sóc và dinh dưỡng hợp lý."
    },
    "Blueberry___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Tưới tiêu hợp lý, đất chua phù hợp."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "mo_ta": "Nấm phấn trắng phủ trên lá non.",
        "dieu_tri": "Phun lưu huỳnh hoặc thuốc gốc strobilurin.",
        "phong_ngua": "Tạo thông thoáng, tránh tưới lên lá."
    },
    "Cherry_(including_sour)___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Chăm sóc và cắt tỉa hợp lý."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "mo_ta": "Đốm lá xám thuôn dài do nấm Cercospora.",
        "dieu_tri": "Phun fungicide nhóm QoI hoặc triazole.",
        "phong_ngua": "Luân canh và dùng giống kháng."
    },
    "Corn_(maize)___Common_rust_": {
        "mo_ta": "Rỉ sắt với các ổ bào tử màu nâu đỏ.",
        "dieu_tri": "Phun fungicide khi bệnh nặng.",
        "phong_ngua": "Chọn giống kháng và quản lý ẩm độ."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "mo_ta": "Đốm hình thoi lớn trên lá.",
        "dieu_tri": "Phun fungicide khi cần thiết.",
        "phong_ngua": "Luân canh và dùng giống kháng."
    },
    "Corn_(maize)___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Bón phân cân đối."
    },
    "Grape___Black_rot": {
        "mo_ta": "Thối quả đen và đốm lá nâu sẫm.",
        "dieu_tri": "Phun mancozeb hoặc myclobutanil.",
        "phong_ngua": "Tỉa lá, vệ sinh lá rụng."
    },
    "Grape___Esca_(Black_Measles)": {
        "mo_ta": "Lá cháy mép, sọc vàng nâu, quả héo.",
        "dieu_tri": "Cắt bỏ cành bệnh; không có thuốc đặc trị.",
        "phong_ngua": "Tránh tổn thương gỗ, quản lý nấm thân."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "mo_ta": "Đốm lá nâu đậm hình đa giác.",
        "dieu_tri": "Phun thuốc nấm gốc đồng.",
        "phong_ngua": "Tăng thông thoáng và vệ sinh vườn."
    },
    "Grape___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Chăm sóc dinh dưỡng và thoát nước tốt."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "mo_ta": "Vàng lá loang lổ, quả nhỏ méo, bệnh do vi khuẩn.",
        "dieu_tri": "Không chữa khỏi; loại bỏ cây bệnh.",
        "phong_ngua": "Kiểm soát rầy chổng cánh, dùng cây giống sạch bệnh."
    },
    "Peach___Bacterial_spot": {
        "mo_ta": "Đốm vi khuẩn trên lá và quả.",
        "dieu_tri": "Phun thuốc gốc đồng.",
        "phong_ngua": "Chọn giống kháng và tưới gốc."
    },
    "Peach___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Cắt tỉa và bón phân hợp lý."
    },
    "Pepper,_bell___Bacterial_spot": {
        "mo_ta": "Đốm nước, sậm màu trên lá và quả.",
        "dieu_tri": "Phun thuốc đồng hoặc kasugamycin.",
        "phong_ngua": "Tưới gốc, tránh ẩm lá."
    },
    "Pepper,_bell___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Chăm sóc và phòng sâu hại."
    },
    "Potato___Early_blight": {
        "mo_ta": "Đốm đồng tâm trên lá.",
        "dieu_tri": "Phun thuốc chứa chlorothalonil.",
        "phong_ngua": "Luân canh và bón phân cân bằng."
    },
    "Potato___Late_blight": {
        "mo_ta": "Đốm thối nâu lan nhanh, bệnh rất nguy hiểm.",
        "dieu_tri": "Phun fosetyl-Al hoặc metalaxyl.",
        "phong_ngua": "Thoát nước tốt, dùng giống kháng."
    },
    "Potato___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Vệ sinh luống và bón phân hữu cơ."
    },
    "Raspberry___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Tưới và cắt tỉa hợp lý."
    },
    "Soybean___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Luân canh và quản lý cỏ dại."
    },
    "Squash___Powdery_mildew": {
        "mo_ta": "Nấm phấn trắng phủ lá.",
        "dieu_tri": "Phun lưu huỳnh hoặc neem.",
        "phong_ngua": "Giảm ẩm, trồng thưa."
    },
    "Strawberry___Leaf_scorch": {
        "mo_ta": "Đốm đỏ nâu cháy lá.",
        "dieu_tri": "Phun thuốc gốc đồng.",
        "phong_ngua": "Tưới gốc và vệ sinh tàn dư."
    },
    "Strawberry___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Giữ luống khô thoáng."
    },
    "Tomato___Bacterial_spot": {
        "mo_ta": "Đốm nhỏ sậm trên lá và quả.",
        "dieu_tri": "Phun đồng hoặc streptomycin.",
        "phong_ngua": "Tưới gốc, chọn giống sạch bệnh."
    },
    "Tomato___Early_blight": {
        "mo_ta": "Đốm đồng tâm màu nâu.",
        "dieu_tri": "Phun chlorothalonil hoặc mancozeb.",
        "phong_ngua": "Luân canh và cắt bỏ lá bệnh."
    },
    "Tomato___Late_blight": {
        "mo_ta": "Thối nâu lan nhanh trên lá và quả.",
        "dieu_tri": "Phun metalaxyl hoặc cymoxanil.",
        "phong_ngua": "Giữ khô lá, dùng giống kháng."
    },
    "Tomato___Leaf_Mold": {
        "mo_ta": "Mốc vàng mặt trên và mốc xanh mặt dưới lá.",
        "dieu_tri": "Phun thuốc nhóm QoI hoặc đồng.",
        "phong_ngua": "Thông thoáng nhà màng."
    },
    "Tomato___Septoria_leaf_spot": {
        "mo_ta": "Đốm nhỏ xám viền nâu.",
        "dieu_tri": "Phun mancozeb hoặc copper.",
        "phong_ngua": "Vệ sinh lá bệnh và luân canh."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "mo_ta": "Nhện đỏ gây vàng lá, có tơ mịn.",
        "dieu_tri": "Phun dầu neem hoặc abamectin.",
        "phong_ngua": "Giữ ẩm, hạn chế khô nóng."
    },
    "Tomato___Target_Spot": {
        "mo_ta": "Đốm nâu có vòng tròn đồng tâm.",
        "dieu_tri": "Phun chlorothalonil.",
        "phong_ngua": "Thoáng khí và cắt lá bệnh."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "mo_ta": "Lá xoăn vàng do virus truyền bởi bọ phấn.",
        "dieu_tri": "Không có thuốc; nhổ bỏ cây bệnh.",
        "phong_ngua": "Kiểm soát bọ phấn, lưới chống côn trùng."
    },
    "Tomato___Tomato_mosaic_virus": {
        "mo_ta": "Lá biến dạng và loang vàng.",
        "dieu_tri": "Không trị được; loại bỏ cây bệnh.",
        "phong_ngua": "Khử trùng dụng cụ, giống sạch bệnh."
    },
    "Tomato___healthy": {
        "mo_ta": "Cây khỏe mạnh.",
        "dieu_tri": "Không cần.",
        "phong_ngua": "Chăm sóc tốt và tưới hợp lý."
    }
}

# --- SIDEBAR (THANH CÔNG CỤ BÊN TRÁI) ---
with st.sidebar:
    st.title("🌿 Plant Doctor ")
    st.caption("Hệ thống chẩn đoán bệnh cây trồng (YOLOv10)")
    st.divider()

    # 1. Chế độ Camera
    st.subheader("📸 Camera")
    use_camera = st.toggle("Sử dụng Camera trực tiếp", False)
    
    st.divider()

    # 2. Thanh trượt Ngưỡng tin cậy
    st.subheader("🎛️ Độ tin cậy (Confidence)")
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (Confidence)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Kéo thấp để tìm tất cả các bệnh. Kéo cao để chính xác tuyệt đối."
    )
    
    # Hiển thị trạng thái
    if confidence_threshold < 0.3:
        st.warning("⚠️ Chế độ nhạy cao (Tầm soát)")
    elif confidence_threshold > 0.7:
        st.info("ℹ️ Chế độ khắt khe (Chính xác)")
    
    st.divider()

    # 3. Danh sách bệnh
    with st.expander("📝 Danh sách 38 bệnh mô hình hiện tại có thể dự đoán"):
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
            # Chạy dự đoán với ngưỡng thấp (0.05) để lấy hết kết quả tiềm năng
            results = model.predict(image_pil, conf=0.05) 
            result = results[0]
            
            # LỌC KẾT QUẢ THEO THANH TRƯỢT CỦA NGƯỜI DÙNG
            detected_boxes = []
            for box in result.boxes:
                if float(box.conf[0]) >= confidence_threshold:
                    detected_boxes.append(box)
            
            # Vẽ lại ảnh với kết quả đã lọc
            res_plotted = result.plot(conf=confidence_threshold)
            
            # Lưu vào session state
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

            # Tabs hiển thị kết quả
            tab_img, tab_detail, tab_solution = st.tabs(["🖼️ Trực quan", "📋 Chi tiết", "💊 Giải pháp"])

            with tab_img:
                st.image(st.session_state['result_img'], caption=f"Độ tin cậy > {confidence_threshold*100:.0f}%", use_container_width=True)

            with tab_detail:
                # Hiển thị bảng chi tiết với tọa độ
                data_list = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = names[cls_id]
                    
                    # Lấy tọa độ
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    
                    data_list.append({
                        "Tên bệnh": name,
                        "Độ tin cậy": f"{conf*100:.1f}%",
                        "Tọa độ (Box)": f"[{x1}, {y1}, {x2}, {y2}]"
                    })
                st.dataframe(data_list, use_container_width=True)

            with tab_solution:
                # Lấy danh sách bệnh không trùng lặp
                unique_diseases = set([names[int(box.cls[0])] for box in boxes])
                
                for disease_name in unique_diseases:
                    st.markdown(f"### 🩺 {disease_name}")
                    
                    # Tra cứu trong Database giải pháp
                    if disease_name in solutions_db:
                        sol = solutions_db[disease_name]
                        st.info(f"**Mô tả:** {sol['mo_ta']}")
                        st.warning(f"**Điều trị:** {sol['dieu_tri']}")
                        st.success(f"**Phòng ngừa:** {sol['phong_ngua']}")
                    else:
                        st.warning(f"Chưa có dữ liệu chi tiết cho '{disease_name}'.")
                    st.divider()
    else:
        st.info("👈 Vui lòng tải ảnh hoặc dùng Camera để bắt đầu.")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configurar el ancho de la página a "wide"
#st.set_page_config(layout="wide")

st.header("Detector de polilla dorso de diamante")
st.write("Sube una imagen para detectar polillas.")

# Cargar modelo entrenado
@st.cache_resource
def load_yolo_model():
    # Asegúrate de que esta ruta sea correcta cuando ejecutes tu app
    return YOLO('./archivos_moth/best.pt')

model = load_yolo_model()

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen subida
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    #st.image(image, caption="Imagen Original", use_column_width=True, channels="BGR")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Imagen Original", use_container_width=True, channels="BGR")

    st.write("")
    #st.write("Detectando...")

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = model(pil_image, conf=0.65)

    image_with_detections = results[0].plot()

    num_moth = len(results[0].boxes)

    label_text = f"Polillas detectadas: {num_moth}"

    # Opcional: Añadir el texto manualmente si quieres un estilo específico
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)
    text_background_color = (255, 255, 255)

    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    org_x = 10
    org_y = 30

    cv2.rectangle(image_with_detections,
                  (org_x, org_y - text_height - 10),
                  (org_x + text_width + 10, org_y + baseline + 5),
                  text_background_color,
                  -1)
    cv2.putText(image_with_detections,
                label_text,
                (org_x, org_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA)

    #st.image(image_with_detections, caption=f"Imagen con Detecciones: {num_moth} polillas", use_column_width=True, channels="BGR")
    with col2:
        st.image(image_with_detections, caption=f"Con {num_moth} polillas detectadas", use_container_width=True, channels="BGR")

import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# --- Configuración de la página ---
st.set_page_config(page_title="SmartVision AI", page_icon="🤖", layout="centered")

# --- Encabezado con estilo web ---
st.markdown("""
<h1 style='text-align: center; color: #1E90FF;'>
🤖 SmartVision AI
</h1>
<p style='text-align: center; font-size: 18px; color: #555;'>
Carga o captura una imagen y deja que el modelo entrenado en <b>Teachable Machine</b> identifique su categoría.
</p>
<hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

# --- Información de entorno ---
st.caption(f"🧠 Versión de Python: {platform.python_version()}")

# --- Carga del modelo ---
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- Imagen de referencia ---
image = Image.open('OIG5.jpg')
st.image(image, caption="Modelo de referencia", use_container_width=False, width=350)

# --- Barra lateral ---
with st.sidebar:
    st.header("⚙️ Opciones y ayuda")
    st.markdown("""
    Esta app usa un modelo de <b>Teachable Machine</b> para reconocer imágenes.
    <br><br>
    📸 Usa la cámara para capturar una imagen y ver el resultado del modelo.
    """, unsafe_allow_html=True)

# --- Captura de imagen ---
img_file_buffer = st.camera_input("Toma una foto para analizar:")

# --- Procesamiento y predicción ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir imagen a numpy y normalizar
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    st.markdown("<h3 style='color:#1E90FF;'>📊 Resultados de la predicción:</h3>", unsafe_allow_html=True)

    # Mostrar resultados con estilo
    if prediction[0][0] > 0.5:
        st.success(f"⬅️ Izquierda, con probabilidad: **{prediction[0][0]:.2f}**")
    elif prediction[0][1] > 0.5:
        st.success(f"⬆️ Arriba, con probabilidad: **{prediction[0][1]:.2f}**")
    else:
        st.warning("⚠️ No se detectó una categoría con alta probabilidad. Intenta otra imagen.")

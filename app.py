import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("malaria_model.keras", compile=False)

st.title("Malaria Cell Detection")
st.write("Upload a blood cell image to detect malaria")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((64,64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    # Show prediction score
    st.write("Prediction Score:", round(float(prediction),4))

    # 🔁 Opposite logic (Fixed)
    if prediction > 0.5:
        st.success("✅ Uninfected Cell")
    else:
        st.error("🦠 Parasitized Cell Detected")
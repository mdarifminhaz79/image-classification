import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ------------------------------
# 1. Parameters / Paths
# ------------------------------
MODEL_PATH = "my_model.h5"       # path to your trained model (if saved)
DATASET_PATH = "FamousPeopleBD"  # folder containing subfolders of people

# ------------------------------
# 2. Load model
# ------------------------------
# If your model is already in memory, skip this line
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# 3. Load class names
# ------------------------------
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, d))])

# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.title("Famous People Recognition")
st.write("Upload an image, and the model will predict who is in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # --------------------------
    # 5. Preprocess image
    # --------------------------
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img = img.convert("RGB")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------
    # 6. Predict
    # --------------------------
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Person:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # --------------------------
    # 7. Optional: Show top 3 predictions
    # --------------------------
    top_indices = predictions[0].argsort()[-3:][::-1]
    st.subheader("Top 3 Predictions")
    for i in top_indices:
        st.write(f"{class_names[i]}: {predictions[0][i]:.2f}")

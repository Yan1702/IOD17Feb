import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


# Define class labels
class_names = [
    'bumper_dent',
    'bumper_scratch',
    'door_dent',
    'door_scratch',
    'glass_shatter',
    'head_lamp',
    'tail_lamp',
    'unknown'
]

# Load Image Modle
def load_image_model():
    model = load_model('CapstoneProject/best_model.keras')
    #model = load_model('CapstoneProject/best_model.keras', custom_objects={})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess and predict
def import_and_predict(image_data, model):
    # Match the input size my model expects
    size = (224, 224)
    # Resize and crop the image to fit the model's input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    # Convert to NumPy array and normalize to [0, 1]
    image = np.asarray(image).astype(np.float32) / 255.0
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    #img_reshape = image[np.newaxis, ...]
    img_reshape = np.expand_dims(image, axis=0)
    # Predict
    prediction = model.predict(img_reshape)
    return prediction

# Streamlit UI
def run():
    # Page title
    st.markdown("<h1 style='text-align: center; color:#007BFF;'>🚗 Car Damage Detection</h1>", unsafe_allow_html=True)

    # Description
    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; padding: 10px 0;'>
            <strong>🔍 Detect car damage across 8 classes:</strong><br>
            <span style='color: #444;'>Bumper Dent, Bumper Scratch, Door Dent, Door Scratch,<br>
            Glass Shatter, Head Lamp, Tail Lamp, Unknown</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Upload instruction
    st.markdown("<p style='text-align: center;'>📷 <strong>Upload a car image to analyze damage.</strong></p>", unsafe_allow_html=True)

    # Model loading
    model = load_image_model()

    # File uploader
    file = st.file_uploader("Upload an image file (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if file is None:
        st.info("⬆️ Please upload an image to begin.", icon="ℹ️")
    else:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        prediction = import_and_predict(image, model)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        # Prediction result
        st.markdown(f"""
        <div style='text-align: center; margin-top: 20px;'>
            <h3 style='color: #28a745;'>✅ Prediction: <span style='color: #000;'>{predicted_class}</span></h3>
            <p style='font-size: 16px;'>Confidence: <strong>{confidence:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # All class probabilities with progress bars
        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(class_names):
            prob = float(prediction[0][i])
            st.write(f"**{class_name}**: {prob:.2%}")
            st.progress(prob)


if __name__ == "__main__":
    run()
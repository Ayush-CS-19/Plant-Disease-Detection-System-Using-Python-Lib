import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS for Beautiful Styling
st.markdown("""
    <style>
    /* Global settings */
    body {
        background: linear-gradient(to bottom, #e3ffe7, #d9e7ff); /* Cool gradient */
        font-family: 'Trebuchet MS', sans-serif;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(45deg, #ff9a9e, #fad0c4); /* Gradient sidebar */
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2d3436;
        text-align: center;
        font-weight: bold;
    }
    .header {
        background: linear-gradient(135deg, #f3a683, #f7d794); /* Sunset gradient for header */
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 14px;
        padding: 10px;
        background-color: #ffffff;
        border-top: 2px solid #ccc;
        color: #555;
    }
    .stButton button {
        background-color: #6c5ce7; /* Deep purple button */
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton button:hover {
        background-color: #81ecec; /* Aqua hover */
        color: black;
        transform: scale(1.08);
    }
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3); /* Stylish shadow for images */
    }
    .stMarkdown p {
        font-size: 18px;
        line-height: 1.7;
        color: #4a4a4a;
        text-align: justify;
    }
    </style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to model's input size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("✨ Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.markdown("<div class='header'><h1>PLANT DISEASE RECOGNITION SYSTEM</h1></div>", unsafe_allow_html=True)
    image_path = "home_page.jpeg"
    try:
        st.image(image_path, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading the image: {e}")
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌱✨

    - **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    - **Analysis:** Identify plant diseases using cutting-edge AI models.
    - **Results:** Get accurate results and recommendations for healthy crops.

    ### Highlights:
    - **💡 Smart Detection:** AI-powered analysis for precision.
    - **🎨 Beautiful Interface:** A pleasant and intuitive design.
    - **🚀 Fast Performance:** Quick and reliable results.

    ### Steps to Get Started:
    1. Upload a plant image.
    2. Analyze and detect diseases.
    3. Take action for better harvests!
    """, unsafe_allow_html=True)

elif app_mode == "About":
    st.markdown("<div class='header'><h1>About the Project</h1></div>", unsafe_allow_html=True)
    st.markdown("""
    #### Dataset Overview:
    - Dataset contains 87,000+ RGB images of healthy and diseased plant leaves across 38 classes.
    - **Data Split:** 80% for training, 20% for validation.
    
    #### Key Features:
    - Supports multiple plant species and diseases.
    - Trained with state-of-the-art algorithms for accuracy.
    - Quick detection to support timely action.
    
    #### Content:
    - Training Images: 70,295
    - Validation Images: 17,572
    - Testing Images: 33
    
    🌟 **Aim:** Empower farmers with technology to save crops and improve yields!
    """, unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.markdown("<div class='header'><h1>Disease Recognition</h1></div>", unsafe_allow_html=True)
    test_image = st.file_uploader("📤 Upload a Plant Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("📸 Show Image"):
            st.image(test_image, use_container_width=True)
        
        if st.button("🔍 Analyze"):
            try:
                st.balloons()
                st.write("Analyzing the Image... Please wait.")
                result_index = model_prediction(test_image)
                
                # Disease Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy'
                ]
                
                st.success(f"🌟 **Disease Identified:** {class_name[result_index]}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("📁 Please upload an image to proceed.")

# Footer
st.markdown("<div class='footer'>Crafted with 💖 by Ayush</div>", unsafe_allow_html=True)
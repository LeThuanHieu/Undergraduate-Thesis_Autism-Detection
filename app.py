import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page config for a wider layout
st.set_page_config(
    page_title="Autism Detection Model Testing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling with background image
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-image: url('GUI/Gui_Build_V1_0704/Background_Image.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .main > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .uploadedFile {
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(240, 242, 246, 0.95);
        padding: 2rem 1rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #1b5e20;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white;
        border-radius: 5px;
    }
    
    /* Title and text styling */
    h1, h2, h3 {
        color: #1b5e20;
    }
    .stProgress > div > div {
        background-color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧩 Autism Detection Model Testing")
st.markdown("""
    This application allows you to test different autism detection models on images.
    You can either upload an image from your device or capture one using your camera.
""")

# Function to load and preprocess image
def preprocess_image(image, target_size=(192, 192)):
    # Handle different input types
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    # Reshape to match model's expected input shape (batch, height, width, channels)
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    
    return img_array

# Function to load model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar for model selection
st.sidebar.title("Model Selection")



# Get list of model directories
model_base_path = "Integrated Testing Model"
model_architectures = [d for d in os.listdir(model_base_path) 
                      if os.path.isdir(os.path.join(model_base_path, d))]

# Model architecture selection
selected_architecture = st.sidebar.selectbox(
    "Model Architecture",
    model_architectures
)

# Function to get model version from filename
def get_model_info(filename):
    parts = filename.replace('.h5', '').split('_')
    version = next((p for p in parts if p.startswith('v')), '')
    run_type = next((p for p in parts if p in ['Singular', 'CrossValidation']), '')
    return f"{run_type}-{version}" if run_type and version else filename

# Get available models for selected architecture
model_path = os.path.join(model_base_path, selected_architecture, "Autism Detection", "Model")
if os.path.exists(model_path):
    model_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
    # Create display names for the models
    model_options = {get_model_info(f): f for f in model_files}
    
    selected_version = st.sidebar.selectbox(
        "Model Version",
        list(model_options.keys())
    )
    
    selected_model = model_options[selected_version]
else:
    st.sidebar.error(f"No models found for {selected_architecture}")
    selected_model = None

# Main content area with tabs
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Camera Capture"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection", key="upload_detect"):
            # Load and run model
            model_full_path = os.path.join(model_path, selected_model)
            model = load_model(model_full_path)
            
            if model:
                # Preprocess and predict
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Display results
                st.subheader("Detection Results")
                probability = prediction[0][0]
                result = "Autism Detected" if probability > 0.5 else "No Autism Detected"
                confidence = probability if probability > 0.5 else 1 - probability
                
                st.markdown(f"**Result:** {result}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Display probability bar
                st.progress(float(confidence))

with tab2:
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        if st.button("Run Detection", key="camera_detect"):
            # Load and run model
            model_full_path = os.path.join(model_path, selected_model)
            model = load_model(model_full_path)
            
            if model:
                # Preprocess and predict
                image = Image.open(picture)
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Display results
                st.subheader("Detection Results")
                probability = prediction[0][0]
                result = "Autism Detected" if probability > 0.5 else "No Autism Detected"
                confidence = probability if probability > 0.5 else 1 - probability
                
                st.markdown(f"**Result:** {result}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Display probability bar
                st.progress(float(confidence))

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Developed for Autism Detection Research</p>
    </div>
""", unsafe_allow_html=True)
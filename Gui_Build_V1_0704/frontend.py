import streamlit as st
from PIL import Image
from backend import load_model, preprocess_image, get_available_models, get_model_versions, run_prediction
import os

# Set page config for a wider layout
st.set_page_config(
    page_title="Autism Detection Model Testing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Image display styling */
    .stImage {
        max-height: 400px;
        width: auto !important;
        margin: 0 auto;
        display: block;
    }
    .stImage > img {
        max-height: 400px;
        width: auto !important;
        object-fit: contain;
    }
    /* Main container styling */
    .stApp {
        background-image: url('Background_Image.png');
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
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.95), rgba(45, 43, 85, 0.95));
        padding: 2rem 1rem;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }
    [data-testid="stSidebar"] .stTitle {
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    /* Hamburger menu button */
    .sidebar-toggle {
        position: fixed;
        top: 0.5rem;
        left: 0.5rem;
        z-index: 99999;
        padding: 0.5rem;
        background: #1b5e20;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .sidebar-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(27, 94, 32, 0.5);
    }
    
    /* Title and text styling */
    h1, h2, h3 {
        color: #1b5e20;
    }
    .stProgress > div > div {
        background-color: #2e7d32;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(75, 0, 130, 0.9), rgba(30, 144, 255, 0.9));
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(27, 94, 32, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Add hamburger menu button
st.markdown("""
<div class='sidebar-toggle'>☰</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🧩 Autism Detection Model Testing")
st.markdown("""
    This application allows you to test different autism detection models on images.
    You can either upload an image from your device or capture one using your camera.
""")

# Sidebar for model selection
st.sidebar.title("Model Selection")

# Get list of model directories
model_base_path = "../Integrated Testing Model"
model_architectures = get_available_models(model_base_path)

# Model architecture selection
selected_architecture = st.sidebar.selectbox(
    "Model Architecture",
    model_architectures
)

# Get available models for selected architecture
model_path = os.path.join(model_base_path, selected_architecture, "Autism Detection", "Model")
model_options = get_model_versions(model_path)

if model_options:
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
        st.image(image, caption="Uploaded Image", use_column_width=False)
        
        if st.button("Run Detection", key="upload_detect"):
            # Load and run model
            model_full_path = os.path.join(model_path, selected_model)
            model = load_model(model_full_path)
            
            if model:
                # Preprocess and predict
                # Explicitly check if we're using ResNet50 and ensure proper preprocessing
                if 'ResNet50' in selected_architecture:
                    # For ResNet50, ensure we use the correct input size (224x224)
                    # First resize the image to 224x224 before preprocessing
                    resized_image = image.resize((224, 224))
                    processed_image = preprocess_image(resized_image, 'ResNet50')
                    print(f"Frontend: Processed ResNet50 image shape: {processed_image.shape if hasattr(processed_image, 'shape') else 'unknown'}")
                else:
                    processed_image = preprocess_image(image, selected_architecture)
                result, confidence = run_prediction(model, processed_image)
                
                # Display results
                st.subheader("Detection Results")
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
                # Explicitly check if we're using ResNet50 and ensure proper preprocessing
                if 'ResNet50' in selected_architecture:
                    # For ResNet50, ensure we use the correct input size (224x224)
                    processed_image = preprocess_image(image, 'ResNet50')
                else:
                    processed_image = preprocess_image(image, selected_architecture)
                result, confidence = run_prediction(model, processed_image)
                
                # Display results
                st.subheader("Detection Results")
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
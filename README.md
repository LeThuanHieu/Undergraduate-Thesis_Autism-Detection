# Autism Detection System 🧩

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive system for autism detection using deep learning models and a user-friendly GUI interface. This project implements multiple state-of-the-art neural network architectures to analyze facial images for autism detection.

## 🌟 Features

- Multiple deep learning model architectures:
  - CNN (Convolutional Neural Network)
  - ResNet50
  - EfficientNetB3
  - MobileNet
  - Vision Transformer (ViT)
- User-friendly Streamlit interface
- Real-time image processing and prediction
- Support for multiple model versions
- Comprehensive test dataset included

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone git@github.com:LeThuanHieu/Undergraduate-Thesis_Autism-Detection.git
cd Undergraduate-Thesis_Autism-Detection/GUI
```

2. Install dependencies:
```bash
pip install -r Gui_Build_V1_0704/requirements.txt
```

## 💻 Usage

1. Navigate to the GUI directory:
```bash
cd Gui_Build_V1_0704
```

2. Run the Streamlit application:
```bash
streamlit run frontend.py
```

3. Open your web browser and go to `http://localhost:8501`

4. Use the sidebar to:
   - Select a model architecture
   - Choose model version
   - Upload an image for analysis

## 🏗️ Project Structure

```
GUI/
├── Gui_Build_V1_0704/          # Main application code
│   ├── frontend.py              # Streamlit UI implementation
│   ├── backend.py               # Model loading and processing logic
│   └── requirements.txt         # Project dependencies
├── Integrated Testing Model/    # Different model architectures
│   ├── CNN/
│   ├── ResNet50/
│   ├── EfficientNetB3/
│   ├── MobileNet/
│   └── ViT/
└── Test Dataset/               # Sample images for testing
```

## 🧠 Model Architectures

### CNN
- Custom convolutional neural network
- Input size: 150x150 pixels
- Specialized for facial feature extraction

### ResNet50
- Deep residual network architecture
- Input size: 224x224 pixels
- Excellent at handling complex features

### EfficientNetB3
- Optimized convolutional neural network
- Input size: 300x300 pixels
- Balanced performance and efficiency

### MobileNet
- Lightweight architecture
- Input size: 224x224 pixels
- Optimized for mobile and embedded devices

### Vision Transformer (ViT)
- Transformer-based architecture
- Input size: 224x224 pixels
- State-of-the-art performance on vision tasks

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Le Thuan Hieu - [GitHub](https://github.com/LeThuanHieu)

One-man Undergraduate Thesis Researching & Exploring the Possibilities of Autism Detection in Children Utilizing Deep Learning Techniques, featuring a basic GUI for Demonstration.

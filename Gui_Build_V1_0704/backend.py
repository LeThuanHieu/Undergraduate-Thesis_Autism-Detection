import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Model-specific configurations
MODEL_CONFIGS = {
    'CNN': {
        'input_size': (150, 150),
        'preprocessing': lambda x: x / 255.0,
        'mean': None,
        'std': None
    },
    'ResNet50': {
        'input_size': (224, 224),
        'preprocessing': lambda x: tf.keras.applications.resnet50.preprocess_input(x),
        'mean': None,
        'std': None,
        'expected_shape': (1, 224, 224, 3)
    },
    'EfficientNetB3': {
        'input_size': (300, 300),
        'preprocessing': lambda x: tf.keras.applications.efficientnet.preprocess_input(x),
        'mean': None,
        'std': None
    },
    'MobileNet': {
        'input_size': (224, 224),
        'preprocessing': lambda x: tf.keras.applications.mobilenet.preprocess_input(x),
        'mean': None,
        'std': None
    },
    'ViT': {
        'input_size': (224, 224),
        'preprocessing': lambda x: x / 127.5 - 1,
        'mean': None,
        'std': None
    }
}

# Function to preprocess image
def preprocess_image(image, architecture='CNN'):
    # Get model configuration
    config = MODEL_CONFIGS.get(architecture, MODEL_CONFIGS['CNN'])
    target_size = config['input_size']
    
    # Special handling for ResNet50 to ensure shape (1, 224, 224, 3)
    is_resnet = architecture == 'ResNet50'
    
    # Always force target size for ResNet50 to be (224, 224) regardless of config
    if is_resnet:
        target_size = (224, 224)
        print(f"ResNet50 detected: Forcing target size to {target_size}")
    
    # Check if image is already a preprocessed numpy array with correct dimensions
    if isinstance(image, np.ndarray):
        # If it's already a 4D array with the correct input size, return it directly
        if len(image.shape) == 4 and image.shape[1:3] == target_size:
            # For ResNet50, ensure the shape is exactly (1, 224, 224, 3)
            if is_resnet and image.shape != (1, 224, 224, 3):
                # Reshape to ensure batch size is 1
                if image.shape[0] != 1:
                    image = image[0:1]
            return image
        # If it's a 4D array but with wrong dimensions, reshape it
        elif len(image.shape) == 4:
            # Extract the first image if it's a batch
            single_img = image[0] if image.shape[0] == 1 else image
            # Convert to PIL for resizing if needed
            try:
                if len(single_img.shape) == 3:  # It's a single image
                    img = Image.fromarray(np.uint8(single_img))
                    img = img.resize(target_size)
                    img_array = np.array(img).astype('float32')
                    img_array = config['preprocessing'](img_array)
                    return np.expand_dims(img_array, axis=0)
                else:
                    # Handle other shapes appropriately
                    raise ValueError(f"Unexpected array shape: {single_img.shape}")
            except Exception as e:
                # If conversion fails, try direct preprocessing
                processed = config['preprocessing'](image)
                # Ensure batch size is 1 for ResNet50
                if is_resnet and len(processed.shape) == 4 and processed.shape[0] != 1:
                    processed = processed[0:1]
                return processed
        # If it's a 3D array with the correct input size, just expand dimensions
        elif len(image.shape) == 3 and image.shape[0:2] == target_size:
            img_array = np.expand_dims(image, axis=0)
            # Apply preprocessing
            img_array = config['preprocessing'](img_array)
            return img_array
        # If it's a 3D array with wrong dimensions, resize it
        elif len(image.shape) == 3:
            try:
                img = Image.fromarray(np.uint8(image))
                img = img.resize(target_size)
                img_array = np.array(img).astype('float32')
                # Apply preprocessing first
                img_array = config['preprocessing'](img_array)
                # Then expand dimensions
                return np.expand_dims(img_array, axis=0)
            except Exception as e:
                # If conversion fails, try direct preprocessing
                img_array = config['preprocessing'](image)
                return np.expand_dims(img_array, axis=0)
    
    # Handle different input types
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        try:
            # Try to convert to uint8 first to ensure compatibility with PIL
            if hasattr(image, 'astype'):
                img = Image.fromarray(np.uint8(image))
            else:
                raise ValueError(f"Cannot convert image of type {type(image)} to PIL Image")
        except Exception as e:
            raise ValueError(f"Cannot convert image of type {type(image)} and shape {image.shape if hasattr(image, 'shape') else 'unknown'} to PIL Image: {str(e)}")
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array and ensure float32 type
    img_array = np.array(img).astype('float32')
    
    # Apply model-specific preprocessing
    img_array = config['preprocessing'](img_array)
    
    # Ensure the array has the correct shape (batch_size, height, width, channels)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
        
    # Final check for ResNet50
    if is_resnet:
        print(f"Warning: Checking ResNet50 image dimensions {img_array.shape}")
        if img_array.shape[1:3] != (224, 224):
            print(f"Forcing resize to (1, 224, 224, 3)")
            # Convert to tensor for resizing
            tensor_img = tf.convert_to_tensor(img_array)
            # Use TensorFlow's resize operation
            resized = tf.image.resize(tensor_img, (224, 224))
            # Apply ResNet50 preprocessing
            img_array = tf.keras.applications.resnet50.preprocess_input(resized)
    
    return img_array

# Function to load model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None, str(e)

# Function to get model version from filename
def get_model_info(filename):
    parts = filename.replace('.h5', '').split('_')
    version = next((p for p in parts if p.startswith('v')), '')
    run_type = next((p for p in parts if p in ['Singular', 'CrossValidation']), '')
    return f"{run_type}-{version}" if run_type and version else filename

# Function to get available models
def get_available_models(model_base_path):
    model_architectures = [d for d in os.listdir(model_base_path) 
                         if os.path.isdir(os.path.join(model_base_path, d))]
    return model_architectures

# Function to get model versions
def get_model_versions(model_path):
    if os.path.exists(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
        return {get_model_info(f): f for f in model_files}
    return {}

# Function to run prediction
def run_prediction(model, processed_image):
    # Get the model architecture from the model's name
    model_name = str(model.name).lower()
    architecture = 'CNN'  # default architecture
    
    # Check for specific architectures
    if 'resnet' in model_name:
        architecture = 'ResNet50'
    elif 'efficientnet' in model_name:
        architecture = 'EfficientNetB3'
    elif 'mobilenet' in model_name:
        architecture = 'MobileNet'
    elif 'vit' in model_name:
        architecture = 'ViT'
    
    # Get the expected input size for this architecture
    target_size = MODEL_CONFIGS[architecture]['input_size']
    
    # Special handling for ResNet50 to ensure shape (1, 224, 224, 3)
    is_resnet = architecture == 'ResNet50'
    expected_shape = (1, 224, 224, 3) if is_resnet else None
    
    # For ResNet50, always force the target size to be (224, 224)
    if is_resnet:
        target_size = (224, 224)
        
    # Debug information
    print(f"Model architecture detected: {architecture}, Target size: {target_size}")
    
    # For ResNet50, we need to ensure the image is properly resized to 224x224
    if is_resnet:
        # If it's already a numpy array, check its shape
        if isinstance(processed_image, np.ndarray):
            # If dimensions don't match 224x224, resize it
            if len(processed_image.shape) == 4 and processed_image.shape[1:3] != (224, 224):
                print(f"ResNet50: Resizing from {processed_image.shape} to (1, 224, 224, 3)")
                try:
                    # Try using TensorFlow's resize operation
                    resized = tf.image.resize(processed_image, (224, 224))
                    processed_image = tf.keras.applications.resnet50.preprocess_input(resized)
                except Exception as e:
                    print(f"TF resize failed: {str(e)}. Trying PIL resize.")
                    try:
                        # Try PIL resize as fallback
                        img = Image.fromarray(np.uint8(processed_image[0]))
                        img = img.resize((224, 224))
                        img_array = np.array(img).astype('float32')
                        processed_image = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))
                    except Exception as pil_error:
                        print(f"PIL resize failed: {str(pil_error)}")
                        raise ValueError(f"Unable to resize image for ResNet50")
            elif len(processed_image.shape) == 3 and processed_image.shape[0:2] != (224, 224):
                print(f"ResNet50: Resizing 3D array from {processed_image.shape} to (1, 224, 224, 3)")
                try:
                    # Try using PIL for resizing
                    img = Image.fromarray(np.uint8(processed_image))
                    img = img.resize((224, 224))
                    img_array = np.array(img).astype('float32')
                    processed_image = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))
                except Exception as e:
                    print(f"PIL resize failed: {str(e)}. Trying TF resize.")
                    try:
                        # Try TensorFlow resize as fallback
                        expanded = np.expand_dims(processed_image, axis=0)
                        resized = tf.image.resize(expanded, (224, 224))
                        processed_image = tf.keras.applications.resnet50.preprocess_input(resized)
                    except Exception as tf_error:
                        print(f"TF resize failed: {str(tf_error)}")
                        raise ValueError(f"Unable to resize image for ResNet50")
        else:
            # If not a numpy array, use preprocess_image with ResNet50 architecture
            processed_image = preprocess_image(processed_image, 'ResNet50')
    else:
        # For non-ResNet50 models, use the standard preprocessing pipeline
        if not isinstance(processed_image, np.ndarray):
            # If not a numpy array, preprocess it
            processed_image = preprocess_image(processed_image, architecture)
        elif len(processed_image.shape) == 3:
            # If it's a 3D array, check dimensions and resize if needed
            if processed_image.shape[0:2] != target_size:
                # Convert to PIL for resizing
                img = Image.fromarray(np.uint8(processed_image))
                img = img.resize(target_size)
                img_array = np.array(img).astype('float32')
                # Apply model-specific preprocessing
                img_array = MODEL_CONFIGS[architecture]['preprocessing'](img_array)
                processed_image = np.expand_dims(img_array, axis=0)
            else:
                # Just expand dimensions if size is correct
                processed_image = np.expand_dims(processed_image, axis=0)
                # Apply preprocessing if not already done
                processed_image = MODEL_CONFIGS[architecture]['preprocessing'](processed_image)
        elif len(processed_image.shape) == 4:
            # If it's a 4D array, check if dimensions match the expected input size
            if processed_image.shape[1:3] != target_size:
                # Reshape the image to match the expected input size
                try:
                    # Extract first image if it's a batch
                    img = processed_image[0]
                    # Convert to PIL for resizing to ensure consistent processing
                    img_pil = Image.fromarray(np.uint8(img))
                    img_pil = img_pil.resize(target_size)
                    img_array = np.array(img_pil).astype('float32')
                    # Apply model-specific preprocessing
                    img_processed = MODEL_CONFIGS[architecture]['preprocessing'](img_array)
                    processed_image = np.expand_dims(img_processed, axis=0)
                except Exception as e:
                    # If that fails, try to use the existing preprocessing function
                    print(f"Warning: Direct resize failed, attempting preprocessing: {str(e)}")
                    # Extract the image if it's a batch before preprocessing
                    single_img = processed_image[0] if processed_image.shape[0] == 1 else processed_image
                    processed_image = preprocess_image(single_img, architecture)
    
    # Ensure we have a 4D tensor with shape (batch_size, height, width, channels)
    if len(processed_image.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape {processed_image.shape}")
    
    # Final check for ResNet50 to ensure the shape is exactly (1, 224, 224, 3)
    if is_resnet and processed_image.shape != expected_shape:
        print(f"Final ResNet50 shape check: {processed_image.shape} vs expected {expected_shape}")
        # Last attempt to resize if needed
        try:
            resized = tf.image.resize(processed_image, (224, 224))
            processed_image = tf.keras.applications.resnet50.preprocess_input(resized)
        except Exception as e:
            print(f"Final resize attempt failed: {str(e)}")
            raise ValueError(f"Unable to properly format image for ResNet50 model")
    
    # Print debug information about the processed image shape
    print(f"Processed image shape: {processed_image.shape}, Expected input shape for {architecture}: {expected_shape if is_resnet else f'(batch, {target_size[0]}, {target_size[1]}, 3)'}")
    
    # Final check to ensure dimensions match
    if is_resnet and processed_image.shape != expected_shape:
        raise ValueError(f"ResNet50 image dimensions mismatch: got {processed_image.shape}, expected {expected_shape}")
    elif not is_resnet and processed_image.shape[1:3] != target_size:
        raise ValueError(f"Image dimensions mismatch: got {processed_image.shape[1:3]}, expected {target_size}")
    
    prediction = model.predict(processed_image)
    probability = prediction[0][0]
    result = "Autism Detected" if probability > 0.5 else "No Autism Detected"
    confidence = probability if probability > 0.5 else 1 - probability
    return result, confidence
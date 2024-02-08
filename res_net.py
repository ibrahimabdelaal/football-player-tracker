import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load the pre-trained ResNet model without the top (classification) layer
base_model = ResNet152(weights='imagenet', include_top=False)

# Add a global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# Add a dense layer for feature extraction
x = Dense(128, activation='relu')(x)  # Change the number of units (e.g., 1024) as desired

# Create a new model with the added layers
model = Model(inputs=base_model.input, outputs=x)

# Load the weights from the pre-trained model
model.load_weights('resnet152_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

# Load and preprocess the input image using OpenCV
image = cv2.imread('000001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image.astype('float32') / 255.0

# Expand dimensions to match the expected input shape of the model
image = tf.expand_dims(image, axis=0)

# Obtain the image features
features = model.predict(image)

# Print the shape of the features
print(features)


import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np

def extract_features(image, boxes, rescale=True):
    # Load the pre-trained ResNet model without the top (classification) layer
    base_model = ResNet152(weights='imagenet', include_top=False)
    
    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Create a new model with the added global average pooling layer
    model = Model(inputs=base_model.input, outputs=x)
    
    # Load the weights from the pre-trained model
    model.load_weights('resnet_weights.h5', by_name=True)
    
    # Resize the image if needed
    if rescale:
        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 2, height // 2))
        boxes = boxes // 2
    
    features_list = []
    
    for box in boxes:
        # Extract box coordinates
        x1, y1, x2, y2 = box
        
        # Crop the box from the image
        cropped_image = image[y1:y2, x1:x2]
        
        # Preprocess the cropped image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = cropped_image.astype('float32') / 255.0
        cropped_image = tf.expand_dims(cropped_image, axis=0)
        
        # Obtain the features
        features = model.predict(cropped_image)
        
        # Append the features to the list
        features_list.append(features)
    
    # Concatenate the features
    if len(features_list) > 0:
        features = np.concatenate(features_list, axis=0)
    else:
        features = np.empty((0, 2048))
    
    return features

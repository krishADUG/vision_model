import cv2
import numpy as np
import os
import tensorflow as tf

class ImageProcessor:
    def __init__(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image file not found")
        
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def preprocess_for_tf(self, target_size=(224, 224)) -> np.ndarray:
        """Resizes for model."""
        resized = cv2.resize(self.image_rgb, target_size)
        # Convert to float32 and add batch dimension (1, 224, 224, 3)
        input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)
        # Apply MobileNetV2 specific preprocessing (scales pixels to [-1, 1])
        return tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)
        
    def extract_edges(self)-> np.ndarray:
        """Canny Edge Detection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        return edges
    
    def display_edges(self):
        """Showing the edge detection."""
        edges = self.extract_edges()
        cv2.imshow("Original", self.image)
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        
    def extract_edges(self):
        """Grayscale conversion and Canny Edge Detection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
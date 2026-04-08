# main.py
from src.vision2d.processor import ImageProcessor
from src.vision3d.processor import PointCloudProcessor
from src.models.model import TFImageClassifier, PyTorchPointNet

def main():
    print("--- Starting Multi-Modal Vision Framework ---")
    
    # 1. Process 2D Data with OpenCV
    print("Processing 2D Image...")
    image_path = 'data/raw/sample_image.jpg'

    try:
        processor = ImageProcessor(image_path)
        print(f"Successfully loaded image from {image_path}")
    except FileNotFoundError as e:
        print(e)
        return
    tf_input = processor.preprocess_for_tf()
    classifier = TFImageClassifier()
    predictions = classifier.predict(tf_input)
    
    print("\n--- Top 3 Predictions ---")
    for pred in predictions:
        print(pred)
    
    # 2. Process 3D Data with Open3D
    print("Processing 3D Point Cloud...")
    # pcd_proc = PointCloudProcessor('data/raw/sample_cloud.pcd')
    # clean_cloud = pcd_proc.downsample_and_remove_noise()
    
    # 3. TensorFlow Inference
    print("Running TensorFlow 2D Model...")
    # tf_model = TFImageClassifier()
    
    # 4. PyTorch Inference
    print("Running PyTorch 3D Model...")
    # pt_model = PyTorchPointNet()

if __name__ == "__main__":
    main()
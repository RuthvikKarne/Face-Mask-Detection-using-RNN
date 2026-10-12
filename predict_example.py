import os
import sys
import pickle
import cv2
import numpy as np

# Ensure FaceMaskRNNModel is available in scope for unpickling
from face_mask_rnn_model import FaceMaskRNNModel

def load_face_mask_model(model_path="face_mask_rnn_model.pkl"):
    """
    Loads the trained Face Mask RNN model from the pickle file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pickle model file not found at: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded Face Mask RNN Model from '{model_path}' successfully!")
    return model

def predict_single_image(model, image_path):
    """
    Runs prediction on a single image file and displays results.
    """
    result = model.predict(image_path)
    label = result['label']
    confidence = result['confidence'] * 100
    prob_with_mask = result['probability_with_mask'] * 100
    
    badge = "[MASK DETECTED]" if label == "with_mask" else "[NO MASK DETECTED]"
    print(f"\nImage: {image_path}")
    print(f"Status:     {badge}")
    print(f"Class:      {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"P(Mask):    {prob_with_mask:.2f}%")
    return result

if __name__ == "__main__":
    model = load_face_mask_model("face_mask_rnn_model.pkl")
    
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        predict_single_image(model, target_path)
    else:
        # Test on sample images from dataset
        print("\n--- Running Demo Inference on Sample Images ---")
        sample_mask = os.path.join("data", "with_mask", os.listdir("data/with_mask")[0])
        sample_nomask = os.path.join("data", "without_mask", os.listdir("data/without_mask")[0])
        
        predict_single_image(model, sample_mask)
        predict_single_image(model, sample_nomask)

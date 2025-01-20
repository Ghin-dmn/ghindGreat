from ultralytics import YOLO
import torch
import os

def train_model():
    # Check and configure the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Load YOLO model
    model_path = r'yolov8n.pt'
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        return None

    try:
        with open(model_path, 'rb') as f:
            if f.read() == b'':
                print(f"Model file at {model_path} is empty.")
                return None
    except Exception as e:
        print(f"Failed to read the model file: {e}")
        return None

    model = YOLO(model_path)

    try:
        print("Starting training...")
        model.train(
            data=r'C:\Users\johna\PyCharmMiscProject\dataset\data.yaml',
            epochs=2,
            batch=4,
            imgsz=416,
            device=device,
            lr0=0.001,
            project=r'C:\Users\johna\PyCharmMiscProject\Dataset',
            name='veg_cut_training',
            verbose=True
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None

    return model

def test_model(model):
    # Test the trained model
    try:
        print("Starting evaluation on the test dataset...")
        results = model.val(
            data=r'C:\Users\johna\PyCharmMiscProject\Dataset\data.yaml',
            imgsz=416,
            split='test',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True,
            plots=True
        )
        print("Testing completed successfully!")
        print("Results:", results)
    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == "__main__":
    # Main block to train and test the model
    trained_model = train_model()

    if trained_model is not None:
        test_model(trained_model)

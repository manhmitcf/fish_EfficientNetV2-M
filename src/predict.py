import torch
from model import FishClassifier
from dataset import transform
from PIL import Image

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)
model.load_state_dict(torch.load("models/fish_classifier.pth"))
model.eval()

# Map lớp về giá trị score thực tế
labels_map_reverse = {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0,5: 7.0, 6: 8.0, 7: 9.0}

# Hàm dự đoán
def predict(image_path):
    """
    Hàm dự đoán điểm cảm quan từ ảnh cá
    Args:
        image_path (str): Đường dẫn đến ảnh
    Returns:
        float: Điểm cảm quan dự đoán
    """
    try:
        # Mở và xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            predicted_class = torch.argmax(output, dim=1).item()  # Lấy lớp có xác suất cao nhất
        
        # Debug thông tin xác suất
        print(f"Probabilities: {probabilities}")
        
        # Ánh xạ lớp về giá trị score thực tế
        score = labels_map_reverse.get(predicted_class, "Unknown")
        return score
    
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Test hàm dự đoán
img_path = "data/images/sample.jpg"  # Thay bằng đường dẫn ảnh thực tế
predicted_score = predict(img_path)
print(f"Điểm cảm quan dự đoán: {predicted_score}")

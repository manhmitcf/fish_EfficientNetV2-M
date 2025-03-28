import torch
from torch.utils.data import DataLoader
from dataset import FishDataset, transform
from model import FishClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)
try:
    model.load_state_dict(torch.load("models/fish_classifier.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file mô hình 'models/fish_classifier.pth'.")

# Load test dataset
CSV_PATH = "data/val.csv"
IMG_DIR = "data/images/"
try:
    dataset = FishDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Lỗi khi tải dataset: {e}")

# Đánh giá
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Outputs dạng xác suất

        # Lấy lớp dự đoán bằng cách chọn chỉ số có xác suất cao nhất
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Convert về numpy
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Tính accuracy và F1-score toàn bộ tập test
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")

# Báo cáo chi tiết từng lớp
try:
    class_names = ["2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]
    if len(set(all_labels)) > len(class_names):
        raise ValueError("Số lượng lớp thực tế lớn hơn số lớp được định nghĩa.")
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n")
    print(class_report)
except ValueError as e:
    print(f"Lỗi trong việc tạo báo cáo lớp: {e}")

# Tính và hiển thị Confusion Matrix
try:
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
except Exception as e:
    print(f"Lỗi khi tạo Confusion Matrix: {e}")

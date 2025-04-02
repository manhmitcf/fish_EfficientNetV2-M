import argparse
import torch
from torch.utils.data import DataLoader
from dataset import FishDatasetWithAugmentation
from model import FishClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import basic_transform

# 📌 Thêm argparse để nhận tham số từ command line
parser = argparse.ArgumentParser(description="Evaluate Fish Classifier")
parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn tới file mô hình")
parser.add_argument("--csv_path", type=str, required=True, help="Đường dẫn tới file CSV cho dataset")
args = parser.parse_args()

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)

try:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file mô hình '{args.model_path}'.")

# Load test dataset
CSV_PATH = args.csv_path  # Đọc CSV path từ dòng lệnh
IMG_DIR = "data/images/"
IMG_DIR_AUG = "data/train_augmented/"  # Đường dẫn tới thư mục ảnh đã tăng cường

try:
    dataset = FishDatasetWithAugmentation(CSV_PATH, IMG_DIR, IMG_DIR_AUG, transform=basic_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

except FileNotFoundError as e:
    raise FileNotFoundError(f"Lỗi khi tải dataset: {e}")

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
    plt.savefig("confusion_matrix.png")
except Exception as e:
    print(f"Lỗi khi tạo Confusion Matrix: {e}")

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import FishDataset, transform
from model import FishClassifier
import os

# 📌 Thêm argparse để nhận tham số từ terminal
parser = argparse.ArgumentParser(description="Train Fish Classifier")
parser.add_argument("--epochs", type=int, default=20, help="Số epoch để train (default: 20)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
args = parser.parse_args()

# Cấu hình từ argparse
TRAIN_CSV_PATH = "data/train.csv"
VAL_CSV_PATH = "data/val.csv"  # Thêm đường dẫn tới file validation CSV
IMG_DIR = "data/images/"
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_CLASSES = 8

# Load datasets
train_dataset = FishDataset(TRAIN_CSV_PATH, IMG_DIR, transform=transform)
val_dataset = FishDataset(VAL_CSV_PATH, IMG_DIR, transform=transform)

# Tính số lượng mẫu của từng lớp
class_counts = train_dataset.data['score'].value_counts()

# Tính trọng số cho từng lớp
class_weights = 1.0 / class_counts

# Tạo trọng số cho từng mẫu
sample_weights = train_dataset.data['score'].map(class_weights).to_numpy()

# Tạo sampler với trọng số
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Tạo DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier(num_classes=NUM_CLASSES).to(device)  # Đảm bảo model hỗ trợ classification

# Loss và Optimizer
criterion = nn.CrossEntropyLoss()  # Sử dụng SparseCrossEntropyLoss cho class index 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)  # Labels là class index (integer)
        
        optimizer.zero_grad()
        outputs = model(images)  # Output là logits với kích thước [batch_size, num_classes]
        loss = criterion(outputs, labels)  # Tính loss giữa logits và class index
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        
        # Tính số lượng đúng (accuracy)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_train_loss = train_running_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # Tắt tính gradient trong validation
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            # Tính số lượng đúng (accuracy)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = 100 * val_correct / val_total
    
    # In kết quả mỗi epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Lưu mô hình
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fish_classifier.pth")
print("Đã lưu mô hình!")
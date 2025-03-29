import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from dataset import FishDatasetWithAugmentation
from model import FishClassifier  # Đảm bảo model là FishClassifier sử dụng MobileNetV2
import os
from tqdm import tqdm

# 📌 Thêm argparse để nhận tham số từ terminal
parser = argparse.ArgumentParser(description="Train Fish Classifier")
parser.add_argument("--epochs", type=int, default=20, help="Số epoch để train (default: 20)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
args = parser.parse_args()

# Cấu hình từ argparse
TRAIN_CSV_PATH = "data/train.csv"
VAL_CSV_PATH = "data/val.csv"
IMG_DIR = "data/images/"
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_CLASSES = 8

# Transform cơ bản (áp dụng cho tất cả dữ liệu)
basic_transform = transforms.Compose([
    transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform dành riêng cho lớp thiểu số
minority_aug_transform = transforms.Compose([
    transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FishDatasetWithAugmentation(
    csv_file=TRAIN_CSV_PATH,
    img_dir=IMG_DIR,
    transform=basic_transform,
    aug_transform=minority_aug_transform,
    minority_classes=[0, 1, 2, 3, 5, 6, 7]  # Cần cập nhật lớp thiểu số theo bài toán
)

val_dataset = FishDatasetWithAugmentation(
    csv_file=VAL_CSV_PATH,
    img_dir=IMG_DIR,
    transform=basic_transform,
    aug_transform=None
)
class_counts = train_dataset.data['score'].value_counts()
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
sample_weights = train_dataset.data['score'].map(class_weights).to_numpy()
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,  # Đặt shuffle=False khi sử dụng sampler
    sampler=sampler
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier(num_classes=NUM_CLASSES)
model.to(device)

# Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
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
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = 100 * val_correct / val_total
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Lưu mô hình
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fish_classifier.pth")
print("Đã lưu mô hình!")

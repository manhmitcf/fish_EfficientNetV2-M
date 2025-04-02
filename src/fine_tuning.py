import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from dataset import FishDatasetWithAugmentation
from model import FishClassifier 
import os
from tqdm import tqdm
from dataset import basic_transform, aug_transform
# 📌 Thêm argparse để nhận tham số từ terminal
parser = argparse.ArgumentParser(description="Train Fish Classifier")
parser.add_argument("--epochs", type=int, default=20, help="Số epoch để train (default: 20)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--model", type=str, default="models/fish_classifier.pth", help="Tên file mô hình (default: fish_classifier.pth)")
args = parser.parse_args()

# Cấu hình từ argparse
TRAIN_CSV_PATH = "data/train_balanced.csv"
VAL_CSV_PATH = "data/val.csv"
IMG_DIR = "data/images/"
IMG_DIR_AUG = "data/train_augmented/"
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_CLASSES = 8


train_dataset = FishDatasetWithAugmentation(
    csv_file=TRAIN_CSV_PATH,
    img_dir=IMG_DIR,
    img_dir_aug=IMG_DIR_AUG,
    transform=None,
    aug_transform=aug_transform,  
)

val_dataset = FishDatasetWithAugmentation(
    csv_file=VAL_CSV_PATH,
    img_dir=IMG_DIR,
    img_dir_aug=IMG_DIR_AUG,
    transform=basic_transform,
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,   
)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier(num_classes=NUM_CLASSES).to(device)
checkpoint_path = args.model  # Đường dẫn đến file checkpoint
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found. Please check the path.")
checkpoint = torch.load(checkpoint_path, map_location=device)
# Load trọng số cũ vào mô hình
model.load_state_dict(checkpoint)
model.to(device)
# Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_train = []
loss_val = []
acc_train = []
acc_val = []
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
    
    loss_train.append(avg_train_loss)
    loss_val.append(avg_val_loss)
    acc_train.append(train_accuracy)
    acc_val.append(val_accuracy)
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Lưu mô hình
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fish_classifier_finetuned.pth")
print("Đã lưu mô hình!")



import matplotlib.pyplot as plt
import os

# Tạo thư mục "plots" nếu chưa có
os.makedirs("plots", exist_ok=True)

# Vẽ đồ thị loss và accuracy
plt.figure(figsize=(12, 6))

# Đồ thị Loss
plt.subplot(1, 2, 1)  # Chia figure thành 1 hàng, 2 cột, vẽ đồ thị đầu tiên ở vị trí 1
plt.plot(loss_train, label='Train Loss')
plt.plot(loss_val, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Đồ thị Accuracy
plt.subplot(1, 2, 2)  # Chia figure thành 1 hàng, 2 cột, vẽ đồ thị thứ hai ở vị trí 2
plt.plot(acc_train, label='Train Accuracy')
plt.plot(acc_val, label='Val Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Sắp xếp layout để đồ thị không bị chồng lên nhau
plt.tight_layout()

# Lưu đồ thị vào thư mục "plots"
plt.savefig("plots/loss_accuracy_plot_finetuned.png")  # Lưu đồ thị vào file PNG trong thư mục "plots"
print("Đã lưu đồ thị accuracy và loss vào thư mục 'plots'!")

# Hiển thị đồ thị trên màn hình
plt.show()


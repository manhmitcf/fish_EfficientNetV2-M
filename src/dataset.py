import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class FishDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Kiểm tra dữ liệu đầu vào
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Thư mục ảnh '{img_dir}' không tồn tại.")
        if self.data.empty:
            raise ValueError(f"File CSV '{csv_file}' không chứa dữ liệu.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Tên ảnh
        
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy ảnh '{img_name}'.")

        # Lấy nhãn và chuyển đổi
        label = self.data.iloc[idx, 1]  # Giả sử nhãn nằm ở cột thứ hai
        label = int(label)  # Chuyển nhãn thành số nguyên
        label = torch.tensor(label, dtype=torch.long)  # Định dạng cho CrossEntropyLoss
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Resize và thêm các kỹ thuật tăng cường dữ liệu
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize lớn hơn một chút
    transforms.RandomCrop((224, 224)),  # Cắt ngẫu nhiên về 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang ngẫu nhiên
    transforms.RandomRotation(degrees=15),  # Xoay ngẫu nhiên trong khoảng [-15, 15] độ
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Điều chỉnh độ sáng, màu sắc
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
])

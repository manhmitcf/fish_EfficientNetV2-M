import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
class FishDatasetWithAugmentation(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, aug_transform=None, minority_classes=[]):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.aug_transform = aug_transform
        self.minority_classes = minority_classes
        
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
        label = self.data.iloc[idx, 1]
        label = int(label) - 2  # Chuyển nhãn về chỉ số lớp (0-7)
        if label < 0 or label > 7:
            raise ValueError(f"Nhãn '{label}' không hợp lệ. Phải nằm trong khoảng [2, 9].")
        label = torch.tensor(label, dtype=torch.long)
        
        # Áp dụng augmentation nếu thuộc lớp thiểu số
        if label.item() in self.minority_classes and self.aug_transform:
            image = self.aug_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        return image, label

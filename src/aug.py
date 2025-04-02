import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from PIL import Image

# Load CSV chứa thông tin ảnh
csv_path = "data/train.csv"
df = pd.read_csv(csv_path)

# Đếm số lượng ảnh trong mỗi lớp
class_counts = df['score'].value_counts()
max_count = class_counts.max()  # Số lượng ảnh lớp nhiều nhất
print("Số lượng ảnh mỗi lớp:\n", class_counts)

# Định nghĩa Data Augmentation
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Lưu danh sách ảnh mới để cập nhật CSV
new_data = []

# Xử lý từng lớp bị thiếu
for class_label, count in class_counts.items():
    if count < max_count:
        num_augment = max_count - count  # Cần tạo thêm bao nhiêu ảnh
        class_images = df[df['score'] == class_label]['filename'].tolist()

        print(f"Tạo {num_augment} ảnh mới cho lớp {class_label}...")

        for i in range(num_augment):
            img_path = class_images[i % len(class_images)]  # Lặp lại nếu số ảnh ít hơn cần augment
            img_path = os.path.join("data/images/", img_path)
            image = read_image(img_path)  # Đọc ảnh thành tensor (uint8)
            
            # Chuyển tensor thành PIL Image
            image = transforms.ToPILImage()(image)
            
            # Áp dụng augmentation
            augmented_img = augment_transform(image)
            
            # Tạo tên ảnh mới
            new_img_name = f"{os.path.basename(img_path).split('.')[0]}_aug{i}.jpg"
            save_path = f"data/train_augmented/{new_img_name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Tạo thư mục nếu chưa có
            save_image(augmented_img, save_path)

            # Chỉ lưu tên ảnh + lớp vào CSV
            new_data.append([new_img_name, class_label])

# Tạo dataframe mới và lưu lại CSV
df_new = pd.DataFrame(new_data, columns=['image_name', 'score'])
df_final = pd.concat([df[['filename', 'score']], df_new.rename(columns={'image_name': 'filename'})], ignore_index=True)
df_final.to_csv("data/train_balanced.csv", index=False)

print("Tạo ảnh augmentation & cập nhật CSV thành công! 🎉")
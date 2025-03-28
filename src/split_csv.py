import os
import pandas as pd

def split_csv(input_csv, output_dir="data", 
              train_csv="train.csv", val_csv="val.csv", test_csv="test.csv",
              train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(input_csv)

    # Kiểm tra tổng tỷ lệ phải bằng 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Tổng train_ratio, val_ratio và test_ratio phải bằng 1.")

    # Shuffle dữ liệu (ngẫu nhiên hoàn toàn)
    df = df.sample(frac=1).reset_index(drop=True)

    # Tính toán kích thước từng tập dữ liệu
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    # Chia dữ liệu theo chỉ số
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Lưu vào thư mục `data/`
    train_df.to_csv(os.path.join(output_dir, train_csv), index=False)
    val_df.to_csv(os.path.join(output_dir, val_csv), index=False)
    test_df.to_csv(os.path.join(output_dir, test_csv), index=False)

    print("Dataset split completed with full shuffling:")
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

if __name__ == "__main__":
    split_csv("data/fish_scores.csv")

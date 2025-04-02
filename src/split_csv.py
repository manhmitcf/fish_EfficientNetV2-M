import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def split_csv(input_csv, output_dir="data", 
              train_csv="train.csv", val_csv="val.csv", test_csv="test.csv",
              train_ratio=0.8, test_ratio=0.2):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(input_csv)

    # Kiểm tra xem cột 'score' có tồn tại trong dữ liệu không
    if 'score' not in df.columns:
        raise ValueError("File CSV không chứa cột 'score'.")


    # Stratified Split sử dụng label
    sss = StratifiedShuffleSplit(n_splits=1, test_size = test_ratio, random_state=42)
    for train_val_index, test_index in sss.split(df, df['score']):
        train_val_df = df.iloc[train_val_index]
        test_df = df.iloc[test_index]
    
    # Chia train và validation từ tập train_val
    sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in sss_train_val.split(train_val_df, train_val_df['score']):
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]

    # Lưu vào thư mục `data/`
    train_df.to_csv(os.path.join(output_dir, train_csv), index=False)
    val_df.to_csv(os.path.join(output_dir, val_csv), index=False)
    test_df.to_csv(os.path.join(output_dir, test_csv), index=False)

    print("Dataset split completed with Stratified Shuffle Split:")
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

if __name__ == "__main__":
    split_csv("data/fish_scores.csv")

# Phân đoạn khối u não từ ảnh MRI (BraTS 2020 & BraTS-Africa)

Đề tài môn **Một số vấn đề về thị giác máy tính** · Nhóm 2 thành viên (A & B) · Định hướng nghiên cứu.

## Cấu trúc thư mục

```text
brats-tumor-seg/
├── configs/
│   ├── splits.json            # Danh sách chia tập train/val/test cố định (seed=42)
│   └── train_config.yaml      # Cấu hình huấn luyện và tiền xử lý khóa theo Quy ước v1.1
├── docs/
│   ├── QUY_UOC_CHUNG_GIUA_KY_v1.1.pdf
│   └── QUY_TRINH_PHOI_HOP_GIUA_KY.pdf
├── notebooks/                 # Notebook mỏng để chạy trên Google Colab Pro
├── results/                   # Bảng CSV kết quả, biểu đồ EDA, hình overlay
└── src/
    ├── preprocess/            # Kiểm tra NIfTI, tạo splits.json, tiền xử lý v1.1
    ├── datasets/              # PyTorch Dataset (lát 2D)
    ├── models/                # U-Net++ (A), SegFormer-B0 / TransUNet (B)
    ├── train/                 # Script huấn luyện
    ├── eval/                  # Đánh giá 3D: Dice, IoU, HD95 (mm) (B)
    └── viz/                   # Vẽ overlay ảnh gốc / GT / dự đoán (B)
```

## Lưu ý nguồn dữ liệu BraTS 2020
- **Bắt buộc dùng bộ 3D NIfTI gốc** (`awsaf49/brats20-dataset-training-validation` trên Kaggle), **không** dùng bộ `.h5` tách sẵn vì thiếu header NIfTI (`affine`, `spacing mm`) và đã bị z-score lên cả vùng nền đen.

## Hướng dẫn nhanh (Bước 1)

1. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Kiểm tra toàn vẹn 369 ca NIfTI (chạy trên Colab sau khi tải bộ NIfTI từ Kaggle):
   ```bash
   python -m src.preprocess.validate_nifti --data-dir /content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
   ```
3. Tiền xử lý toàn bộ 369 ca theo Quy ước v1.1:
   ```bash
   python -m src.preprocess.preprocess_nifti --data-dir /content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --out-dir /content/preprocessed/brats2020
   ```

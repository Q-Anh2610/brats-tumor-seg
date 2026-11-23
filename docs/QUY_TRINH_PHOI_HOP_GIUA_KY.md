# QUY TRÌNH PHỐI HỢP & MÔI TRƯỜNG THỰC HIỆN (GIỮA KỲ)

**Đề tài:** Phân đoạn khối u não từ ảnh MRI (BraTS 2020 & BraTS-Africa)  
**Phạm vi:** Từ khởi động đến hoàn thành báo cáo + slide giữa kỳ | **Nhóm 2 thành viên (A & B)**

---

## 1. Các bước thực hiện của riêng từng thành viên

### Thành viên A (BraTS 2020 & U-Net++ 2D)
1. **Kiểm tra dữ liệu & chia tập:** Quét 369 ca BraTS 2020 (đủ 4 chuỗi `t1, t1ce, t2, flair` + `seg`, kiểm tra mã nhãn `{0, 1, 2, 4}`, kiểm tra ca `355` hay sai tên file); xuất `configs/splits.json` (`seed=42`: 259 train / 55 val / 55 test).
2. **Thống kê & EDA BraTS 2020:** Xuất bảng thống kê thô (CSV/JSON: thể tích u, cường độ 4 chuỗi, số lát có u, tỷ lệ mất cân bằng lớp) + biểu đồ phân bố & ảnh mẫu 4 kênh.
3. **Tiền xử lý BraTS 2020:** Viết pipeline 7 bước theo Quy ước v1.1 (RAS+, crop bbox + lề 5, z-score vùng não, pad vuông, resize `224×224`, lưu `metadata/<case_id>.json`). Đẩy bản dữ liệu sạch lên Kaggle Private Dataset.
4. **Huấn luyện U-Net++ 2D:** Train với loss `0.5 * BCEWithLogits + 0.5 * SoftDiceLoss` + Augmentation trên tập train; chọn `best_model.pth` theo tập val.
5. **Dự đoán & khôi phục 3D tập test:** Dự đoán toàn bộ lát của 55 ca test → khôi phục về lưới gốc → lưu 55 file `<case_id>_pred_wt.nii.gz`.
6. **Chấm điểm & vẽ Overlay:** Dùng module `eval` và `viz` của B chạy trên 55 ca test để lấy bảng `Dice, IoU, HD95` (trung bình ± SD) và ảnh overlay (ca tốt / trung bình / xấu).

### Thành viên B (BraTS-Africa, Module dùng chung & So sánh phân bố)
1. **Kiểm tra BraTS-Africa (146 ca):** Xác minh đủ 4 chuỗi (`t1n, t1c, t2w, t2f`) + mask, xác nhận chính thức mã nhãn WT trên toàn bộ 146 ca.
2. **Viết module dùng chung (`eval` & `viz`) và giao sớm cho A:**
   - `src/eval/`: hàm khôi phục mask 3D (hoặc đọc `<case_id>_pred_wt.nii.gz`), tính **Dice, IoU, HD95 (mm)** trên lưới gốc, xử lý mask rỗng, xuất CSV từng ca + bảng tổng hợp.
   - `src/viz/`: script vẽ **overlay** (ảnh gốc + Ground Truth + Dự đoán) theo ca và theo lát.
3. **Tiền xử lý & thống kê BraTS-Africa:** Áp dụng đúng pipeline tiền xử lý v1.1 cho BraTS-Africa; xuất thống kê tương tự BraTS 2020.
4. **So sánh phân bố 2 bộ dữ liệu:** Ghép bảng thống kê BraTS 2020 (A gửi) với BraTS-Africa → vẽ biểu đồ so sánh phân bố + **1 biểu đồ PCA/t-SNE 2D** trên đặc trưng thống kê (đáp ứng yêu cầu *"trích xuất đặc trưng, giảm chiều"* ngay ở giữa kỳ).
5. **Thử nghiệm song song (không bắt buộc báo cáo giữa kỳ):** Cài thử SegFormer-B0 chạy vài epoch để kiểm tra pipeline và ước lượng VRAM/thời gian cho cuối kỳ.

---

## 2. Điểm kết hợp & Bàn giao giữa A và B (Chỉ có 4 điểm chạm)

| Mốc | A gửi B | B gửi A | Ghi chú chống kẹt tiến độ |
| :--- | :--- | :--- | :--- |
| **Chạm 1 – Khởi động** | `QUY_UOC_v1.1.pdf`, `configs/splits.json` | Xác nhận chốt v1.1 (size `224×224`, tên key `metadata.json`) | A không cần chờ B kiểm tra xong BraTS-Africa mới làm BraTS 2020. |
| **Chạm 2 – Xong EDA & Tiền xử lý** | File CSV/JSON thống kê BraTS 2020 + code tiền xử lý + link Kaggle Dataset sạch | Xác nhận mã nhãn BraTS-Africa | B dùng chung code tiền xử lý của A để xử lý BraTS-Africa cho đồng bộ. |
| **Chạm 3 – Đánh giá Test** | *(A đã lưu sẵn 55 file `*_pred_wt.nii.gz`)* | Module `src/eval/` và `src/viz/` hoàn chỉnh | A cứ train & xuất sẵn 55 file 3D; khi B giao `eval`/`viz` chỉ chạy 1 lệnh mất 1–2 phút. |
| **Chạm 4 – Báo cáo & Slide** | Phần BraTS 2020, Tiền xử lý, U-Net++ 2D, Kết quả Test, Overlay, Hạn chế | Phần BraTS-Africa, So sánh phân bố + PCA, Quy ước đánh giá 3D | Hai người ghép thành 1 báo cáo tiểu luận + 1 bộ slide giữa kỳ và review chéo. |

---

## 3. Tổ chức Repo & Cách chia file (Tránh xung đột code)

Hai thành viên dùng chung **1 repo GitHub**, mỗi người làm trên **branch riêng** (`feat/a-brats2020-unetpp` và `feat/b-africa-eval`) rồi merge vào `main` theo từng mốc. Mỗi người chỉ sửa thư mục/file thuộc quyền sở hữu của mình:

```text
brats-tumor-seg/
├── configs/
│   ├── splits.json            # [A tạo, chốt cứng – cả 2 chỉ đọc]
│   └── train_config.yaml      # [A + B chốt chung]
├── src/
│   ├── preprocess/            # [A viết chính; B dùng chung cho Africa]
│   ├── datasets/              # [A viết Dataset BraTS2020; B thêm file Africa nếu cần]
│   ├── models/
│   │   ├── unetpp.py          # [A sở hữu]
│   │   └── segformer.py       # [B sở hữu – chạy thử song song]
│   ├── train/
│   │   └── train_unetpp.py    # [A sở hữu]
│   ├── eval/                  # [B sở hữu – metrics 3D, HD95, xuất CSV]
│   └── viz/                   # [B sở hữu – script overlay]
├── notebooks/
│   ├── A_01_eda_brats2020.ipynb       # [File riêng của A]
│   ├── A_02_train_unetpp.ipynb        # [File riêng của A]
│   ├── B_01_eda_africa_compare.ipynb  # [File riêng của B]
│   └── B_02_eval_overlay.ipynb        # [File riêng của B]
├── results/                   # Lưu CSV kết quả, hình biểu đồ, ảnh overlay (không push file .nii/.pth nặng lên Git)
└── README.md                  # [A + B cập nhật chung hướng dẫn chạy lại]
```

> **Nguyên tắc chạy code:**
> - **Code lõi** đặt trong `src/` dưới dạng file `.py` (dễ tái sử dụng, không copy-paste lung tung giữa các notebook).
> - **Notebook (`.ipynb`)** tách riêng theo tiền tố `A_` và `B_`, chỉ dùng để gọi hàm trong `src/`, chạy trên Colab và hiển thị kết quả → **tuyệt đối không đụng hàng / conflict Git**.

---

## 4. Môi trường thực hiện & Chiến lược lưu trữ Dataset

### Môi trường chạy
- **Máy cá nhân (CPU):** Dùng để chạy script kiểm tra cấu trúc file, tạo `splits.json`, thống kê EDA và viết/commit code lên GitHub.
- **Google Colab Pro (A đã có sẵn):**
  - Tiền xử lý / debug: chọn runtime **CPU** hoặc **T4** cho tiết kiệm Compute Units.
  - Huấn luyện U-Net++ 2D: chọn **GPU L4 (24GB VRAM)** + bật Mixed Precision (`torch.cuda.amp`) — nhanh và tối ưu chi phí nhất. Chỉ bật **A100** khi cần train nặng ở cuối kỳ.
- **Ghi nhận môi trường:** Cả hai xuất `requirements.txt` (ghi rõ phiên bản PyTorch, NiBabel, SciPy, Albumentations, SMP) để đảm bảo tái lập theo yêu cầu thầy.

### Lưu trữ & chia sẻ Dataset (Nhanh, gọn, không tốn Google Drive)

| Loại dữ liệu | Nơi lưu trữ khuyến nghị | Cách dùng |
| :--- | :--- | :--- |
| **Dữ liệu gốc BraTS 2020 (~35–40GB)** | **Không upload lên Drive/Hugging Face.** Dùng trực tiếp nguồn có sẵn trên Kaggle. | Trên Colab tải thẳng vào ổ tạm `/content/` qua **Kaggle API** (mất ~2–3 phút, tốc độ mạng nội bộ Google–Kaggle rất nhanh). |
| **Dữ liệu đã tiền xử lý + `metadata/`** | **Kaggle Private Dataset** (chế độ riêng tư, share giữa A và B). | Chạy tiền xử lý **1 lần duy nhất** rồi đẩy bản sạch lên Kaggle Private. Mỗi lần mở Colab train chỉ kéo về `/content/` trong ~1 phút, đọc nhanh hơn Google Drive nhiều lần. |
| **Checkpoint (`.pth`) & Mask 3D dự đoán (`.nii.gz`)** | **Google Drive** (mount vào Colab) | Dung lượng nhỏ (~100–300MB), mount Drive để tự động lưu checkpoint mỗi khi val Dice tăng, tránh mất dữ liệu nếu Colab ngắt kết nối. |
| **Mã nguồn, `configs/`, CSV kết quả, biểu đồ** | **GitHub Repo chung** | Đồng bộ code và số liệu giữa A và B; nộp link repo kèm `README.md` cho thầy. |

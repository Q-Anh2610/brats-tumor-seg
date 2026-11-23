# BỐ CỤC BÁO CÁO GIỮA KỲ (TOÀN NHÓM – A & B)

**Đề tài:** Phân đoạn khối u não từ ảnh MRI (Semantic Segmentation 2D, ghép thành mask 3D)  
**Môn học:** Một số vấn đề về thị giác máy tính · **Nhóm 2 thành viên** · **Định hướng nghiên cứu**

> **Quy ước ký hiệu phân công trong tài liệu này:**
> - **`[A phụ trách]`**: Thành viên A viết chính (đã có gần như đủ số liệu & hình từ BraTS 2020 và U-Net++ 2D).
> - **`[B phụ trách]`**: Thành viên B viết chính (phần BraTS-Africa, so sánh phân bố 2 bộ dữ liệu + PCA, quy ước & module `eval`/`viz`).
> - **`[A + B viết chung]`**: Hai thành viên cùng rà soát và thống nhất mạch lập luận.

---

## TRANG BÌA & MỤC LỤC `[A + B viết chung]`
- Tên đề tài, tên môn học, giảng viên hướng dẫn, thông tin 2 thành viên (ghi rõ tên thật thay cho A và B).
- Danh mục bảng biểu, danh mục hình ảnh, danh mục từ viết tắt (`MRI`, `WT`, `RAS+`, `BCE`, `IoU`, `HD95`, `PCA`).

---

## CHƯƠNG 1. ĐẶT VẤN ĐỀ VÀ TỔNG QUAN DỰ ÁN `[A phụ trách chính, B rà soát]`

### 1.1. Đặt vấn đề & Bối cảnh bài toán `[A phụ trách]`
- Ý nghĩa lâm sàng của phân đoạn tự động vùng khối u não toàn phần (**Whole Tumor – WT**) trên ảnh MRI đa chuỗi (`T1`, `T1ce`, `T2`, `FLAIR`).
- Thách thức của dữ liệu MRI 3D: đa chuỗi, khác biệt cường độ giữa máy chụp/nguồn dữ liệu (domain shift), mất cân bằng lớp nghiêm trọng, và chi phí tính toán khi xử lý trực tiếp 3D.

### 1.2. Mục tiêu, câu hỏi nghiên cứu & Quy trình dự án AI `[A + B viết chung]`
- Trình bày sơ đồ **5 bước của quy trình dự án AI** theo đúng yêu cầu môn học:
  1. Xác định vấn đề → 2. Chọn & kiểm định dữ liệu → 3. Phân tích, tiền xử lý & trích xuất đặc trưng → 4. Mô hình hóa → 5. Khôi phục 3D, đánh giá & trực quan hóa.
- Tóm tắt **phạm vi giữa kỳ vs cuối kỳ** và 3 câu hỏi nghiên cứu (`RQ1`: 2D vs 2.5D; `RQ2`: So sánh U-Net++, TransUNet, SegFormer-B0; `RQ3`: Tổng quát hóa trên BraTS-Africa).

---

## CHƯƠNG 2. DỮ LIỆU, PHÂN TÍCH VÀ TIỀN XỬ LÝ

### 2.1. Bộ dữ liệu chính – BraTS 2020 `[A phụ trách]`
- Mô tả 369 ca (`293 HGG / 76 LGG`), 4 chuỗi (`t1, t1ce, t2, flair`) + `seg`, lưới `(240, 240, 155)`, spacing `1×1×1 mm`, hướng gốc `LPS`.
- Kết quả kiểm tra toàn vẹn `nifti_audit.csv`: **369/369 ca hợp lệ** (nêu lưu ý kỹ thuật ca `BraTS20_Training_355`).
- Quy ước gộp nhãn `{1, 2, 4}` → `WT = 1` (tỷ lệ từng thành phần trong WT: nhãn 1 `22,22%`, nhãn 2 `57,99%`, nhãn 4 `19,79%`).
- Số liệu EDA & 4 biểu đồ (`eda_01` → `eda_04`): `57.195` lát, `42,7%` lát có u, `6,87%` voxel WT trong não, thể tích WT `99,55 ± 59,43 cm³`.

### 2.2. Bộ dữ liệu bổ sung – BraTS-Africa 2023 `[B phụ trách]`
- Lý do bổ sung dữ liệu cùng loại nhưng khác phân phối nguồn chụp/bệnh nhân (phục vụ định hướng nghiên cứu RQ3).
- Mô tả tập BraTS-Africa (số ca hợp lệ trên tổng `146` ca, tên chuỗi `t1n, t1c, t2w, t2f`, kích thước/spacing/hệ tọa độ gốc, kết quả xác minh mã nhãn WT chính thức).
- Thống kê phân bố BraTS-Africa (thể tích u, tỷ lệ lát có u, phân bố cường độ 4 chuỗi) + ảnh mẫu minh họa.

### 2.3. So sánh phân bố 2 bộ dữ liệu & Giảm chiều dữ liệu (Feature Extraction + PCA) `[B phụ trách chính, dùng số liệu từ A & B]`
- Ghép bảng `eda_brats2020_cases.csv` (A gửi) với bảng thống kê BraTS-Africa (B xuất).
- Biểu đồ so sánh phân bố thể tích u, cường độ tín hiệu và tỷ lệ lát có u giữa BraTS 2020 và BraTS-Africa; rút ra nhận xét về khác biệt phân phối (domain shift).
- **Trích xuất đặc trưng thống kê & giảm chiều (PCA / t-SNE 2D):** Chiếu vector đặc trưng thống kê theo từng ca (mean, std, p05, p50, p95 của 4 chuỗi + thể tích não/u) xuống không gian 2D để trực quan hóa mức độ tách biệt/chồng lấp giữa BraTS 2020 và BraTS-Africa *(đáp ứng yêu cầu "trích xuất đặc trưng, giảm chiều dữ liệu" của thầy ngay ở giữa kỳ)*.

### 2.4. Chiến lược chia tập & Pipeline tiền xử lý dùng chung `[A viết chính mục 2.4, B bổ sung điểm riêng của Africa nếu có]`
- **Chia tập theo bệnh nhân (`configs/splits.json`):** Sắp xếp ID, `seed=42`, tỷ lệ `70/15/15` → **259 train / 55 val / 55 test** trên BraTS 2020; chứng minh 3 tập đồng đều phân bố thể tích u (`train: 99,48`, `val: 101,24`, `test: 98,17 cm³`). Khẳng định BraTS-Africa không trộn vào train/val.
- **Pipeline 7 bước theo Quy ước v1.1:** Kiểm tra lưới → đổi hướng về `RAS+` (lưu `ras_ornt`) → hợp vùng não khác `0` → crop bbox XY (+lề `5` voxel) → `z-score` từng kênh trên vùng não → pad đối xứng → resize `224×224` (giải thích lý do đổi từ `240×240` sang `224×224` để chia hết cho 32).
- **Lấy mẫu lát & Augmentation (chỉ trên tập Train):** Lọc lát không não, giữ 100% lát có u + 50% lát não không u mỗi epoch; bộ biến đổi hình học (`HorizontalFlip`, `VerticalFlip`, `Affine`) và cường độ trên vùng não (`Albumentations`).

---

## CHƯƠNG 3. PHƯƠNG PHÁP THỰC HIỆN

### 3.1. Tổng quan kiến trúc nghiên cứu & Khung đánh giá chung `[A + B viết chung]`
- Sơ đồ tổng thể pipeline: Đầu vào 4 chuỗi `[B, 4, 224, 224]` → Mô hình 2D (`logits`) → Khôi phục mask 3D về lưới gốc → Đánh giá `Dice, IoU, HD95`.
- Giới thiệu lộ trình 3 họ mô hình (`U-Net++`, `TransUNet`, `SegFormer-B0`), trong đó **giữa kỳ báo cáo chính thức U-Net++ 2D** (và nêu ngắn tình trạng cài đặt thử SegFormer-B0 ở phần tiến độ nếu có).

### 3.2. Mô hình báo cáo giữa kỳ: U-Net++ 2D `[A phụ trách]`
- Lý do chọn và nguyên lý U-Net++ (dense nested skip connections giúp thu hẹp semantic gap ở biên u).
- Cấu hình triển khai: backbone `resnet34` (pretrained ImageNet), `in_channels = 4`, `classes = 1`, đầu ra `logits`.
- **Hàm Loss kết hợp:** `0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss` (`smooth = 1e-5` batch-level).
- **Thiết lập huấn luyện:** `AdamW` (`lr = 3e-4`, `weight_decay = 1e-4`), `CosineAnnealingLR`, `batch_size = 32`, `AMP`, `epochs = 50`, Early Stopping `patience = 10` theo Val 3D Dice; chạy trên GPU NVIDIA L4 (Colab Pro).

### 3.3. Quy trình khôi phục mask 3D & Độ đo đánh giá `[B phụ trách chính, A đối chiếu phần khôi phục]`
- Quy trình khôi phục từ xác suất `224×224` → nội suy bilinear về kích thước pad → bỏ pad → đặt vào `crop_bbox` → ghép trục Z → đảo `ras_ornt` về `orig_shape` → áp ngưỡng `>= 0.5` → lưu `<case_id>_pred_wt.nii.gz`.
- Định nghĩa toán học và quy ước tính **Dice 3D**, **IoU 3D**, **HD95 (mm)** (lân cận 6 hướng, `distance_transform_edt` theo `spacing`, xử lý mask rỗng, không thay `∞` bằng `0`).

---

## CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ GIỮA KỲ

### 4.1. Quá trình huấn luyện U-Net++ 2D `[A phụ trách – ĐÃ CÓ ĐỦ SỐ LIỆU]`
- Chèn hình `training_curves.png` và bảng tóm tắt từ `training_history.csv`:
  - Tổng số epoch chạy: dừng do Early Stopping sau epoch `31` (không cải thiện sau 10 epoch kể từ mốc tốt nhất).
  - **Checkpoint tốt nhất tại Epoch 22:** `Val 3D Dice = 0,9124 ± 0,0412` (`train_loss = 0,03266`, `val_loss = 0,26111`).
  - Thời gian trung bình mỗi epoch trên GPU L4: ~`231` giây (~`3,85` phút/epoch).
  - **Giải thích kỹ thuật hiện tượng `val_loss ≈ 0,261` > `train_loss ≈ 0,033`:** Không phải overfitting mà do khác biệt phân bố lát giữa lúc train (lọc lát không não + chỉ giữ 50% lát âm, xáo trộn ngẫu nhiên nên batch nào cũng có u) và lúc val (giữ đủ 155 lát theo thứ tự tuần tự; các batch ở cực đầu/cuối hoàn toàn không có u khiến `SoftDiceLoss` batch-level tiến về `1.0`). Thực tế chỉ số `Val 3D Dice = 0,9124` cho thấy mô hình tổng quát hóa rất tốt.

### 4.2. Kết quả định lượng trên tập Test (55 ca BraTS 2020) `[A chạy qua module eval của B → A + B cùng viết nhận xét]`
- **Số liệu tham khảo sơ bộ (trên lưới 224×224 từ `pred_summary_test.csv`):** `55/55` ca có dự đoán (`0` ca rỗng), `quick_dice_224 = 0,8978 ± 0,0681` (trung vị `0,9199`).
- **Bảng kết quả CHÍNH THỨC (chờ chạy 55 file `*_pred_wt.nii.gz` qua `src/eval/` của B):**

| Mô hình | Dice 3D (Mean ± SD) ↑ | IoU 3D (Mean ± SD) ↑ | HD95 (mm, Mean ± SD) ↓ | Ca dự đoán rỗng |
| :--- | :---: | :---: | :---: | :---: |
| **U-Net++ 2D** | `[Chờ module eval của B]` | `[Chờ module eval của B]` | `[Chờ module eval của B]` | `0 / 55` |

### 4.3. Trực quan hóa Overlay và Phân tích lỗi ban đầu `[B cung cấp ảnh từ viz, A phân tích ca tốt/xấu]`
- Chèn bộ ảnh Overlay (Ảnh gốc + Ground Truth + Dự đoán) từ `src/viz/` của B cho các ca đại diện từ kết quả test:
  - **Nhóm ca tốt (Dice > 0,95):** ví dụ `BraTS20_Training_347` (`0,964`), `BraTS20_Training_112` (`0,963`), `BraTS20_Training_362` (`0,959`).
  - **Nhóm ca trung bình (~0,90):** ví dụ `BraTS20_Training_173` (`0,899`), `BraTS20_Training_126` (`0,900`).
  - **Nhóm ca lỗi / Dice thấp nhất:** `BraTS20_Training_048` (`0,634`), `BraTS20_Training_275` (`0,699`), `BraTS20_Training_058` (`0,748`).
- Phân tích nguyên nhân lỗi (khối u thể tích nhỏ, biên thâm nhiễm mờ, bỏ sót lát đầu/cuối khối u) → tạo cơ sở thực nghiệm cho hướng 2.5D ở cuối kỳ.

### 4.4. Tiến độ thử nghiệm mô hình song song (nếu có) `[B phụ trách – không bắt buộc]`
- Ghi nhận ngắn kết quả chạy thử `SegFormer-B0` vài epoch và ước lượng tài nguyên (VRAM, thời gian) chuẩn bị cho giai đoạn cuối kỳ.

---

## CHƯƠNG 5. HẠN CHẾ HIỆN TẠI VÀ KẾ HOẠCH CUỐI KỲ `[A + B viết chung]`
- **Những việc đã hoàn thành ở giữa kỳ:** Khóa quy ước v1.1, xử lý & EDA BraTS 2020 + BraTS-Africa, pipeline đầu-cuối U-Net++ 2D → 3D, module `eval`/`viz` dùng chung.
- **Hạn chế của mô hình hiện tại:** Mất ngữ cảnh liên lát theo trục Z (U-Net++ 2D), chưa đánh giá trên miền dữ liệu BraTS-Africa, chưa có hậu xử lý lọc nhiễu 3D.
- **Kế hoạch cuối kỳ:**
  - `A`: U-Net++ 2.5D (3 và 5 lát), hậu xử lý 3D, ablation, phân tích lỗi theo thể tích/vị trí lát.
  - `B`: Huấn luyện `TransUNet` & `SegFormer-B0`, kiểm thử ngoài nguồn BraTS-Africa, phân tích domain shift & kiểm định thống kê Wilcoxon.

---

## PHỤ LỤC & TÀI LIỆU THAM KHẢO `[A + B viết chung]`
- Link GitHub repo (`README.md` hướng dẫn cài đặt, tải dữ liệu, chạy huấn luyện và đánh giá lại).
- Bảng phân công chi tiết đóng góp của từng thành viên (ghi rõ tên thật).
- Tài liệu tham khảo (BraTS 2020, BraTS-Africa, U-Net++, TransUNet, SegFormer, NiBabel, SciPy).

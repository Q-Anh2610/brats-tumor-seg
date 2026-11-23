# BẢN NHÁP BÁO CÁO GIỮA KỲ – PHẦN THÀNH VIÊN A

**Đề tài:** Phân đoạn khối u não từ ảnh MRI đa chuỗi (Semantic Segmentation 2D, khôi phục mask 3D)  
**Định hướng:** Nghiên cứu  
**Phạm vi tài liệu này:** Các mục do **Thành viên A** phụ trách (Đặt vấn đề, Dữ liệu BraTS 2020 & Tiền xử lý, Phương pháp U-Net++ 2D, Kết quả huấn luyện và khung chờ bảng đánh giá chính thức từ module `eval`/`viz` của Thành viên B).

---

## CHƯƠNG 1. ĐẶT VẤN ĐỀ VÀ GIỚI THIỆU BÀI TOÁN

### 1.1. Bối cảnh và mục tiêu bài toán
U não (đặc biệt là u thần kinh đệm – glioma) có hình thái, kích thước và ranh giới rất đa dạng trên ảnh cộng hưởng từ (MRI). Việc khoanh vùng thủ công từng lát cắt 3D tốn nhiều thời gian của bác sĩ chẩn đoán hình ảnh và dễ phụ thuộc chủ quan. Do đó, bài toán **phân đoạn ngữ nghĩa (Semantic Segmentation)** nhằm tự động xác định **toàn bộ vùng khối u (Whole Tumor – WT)** đóng vai trò quan trọng trong hỗ trợ chẩn đoán, đo thể tích khối u và lập kế hoạch xạ trị/phẫu thuật.

Trong giai đoạn giữa kỳ, nhóm tập trung vào bài toán phân đoạn nhị phân vùng WT (`nền = 0`, `vùng u = 1`) từ **4 chuỗi MRI não đồng đăng ký** (`T1`, `T1ce`, `T2`, `FLAIR`) của cùng bệnh nhân, xử lý theo từng lát cắt 2D theo mặt phẳng axial rồi ghép và khôi phục về khối 3D trên lưới không gian gốc để đánh giá theo ca bệnh.

### 1.2. Quy trình thực hiện dự án AI và định hướng nghiên cứu
Nhóm triển khai bài toán theo đúng quy trình chuẩn của một dự án AI:
1. **Xác định vấn đề:** Phân đoạn WT trên ảnh MRI 3D theo hướng tiếp cận 2D → khôi phục 3D; đánh giá ở mức bệnh nhân bằng `Dice`, `IoU` và `HD95 (mm)`.
2. **Thu thập và kiểm định dữ liệu:** Sử dụng **BraTS 2020** (369 ca) làm dữ liệu huấn luyện/kiểm thử chính và bổ sung **BraTS-Africa** (146 ca) *(phần của Thành viên B)* để phân tích sự khác biệt phân phối (domain shift).
3. **Phân tích và tiền xử lý dữ liệu:** Kiểm tra tính toàn vẹn lưới không gian NIfTI, đưa về chuẩn hướng `RAS+`, tách vùng não và cắt khung (`crop`), chuẩn hóa `z-score` trên vùng não, tách lát 2D, đồng bộ kích thước `224×224` và tăng cường dữ liệu (augmentation) cho tập huấn luyện.
4. **Mô hình hóa:** Giai đoạn giữa kỳ xây dựng mô hình nền tảng **U-Net++ 2D** với hàm loss kết hợp `Dice + BCE` *(cuối kỳ mở rộng sang U-Net++ 2.5D, TransUNet và SegFormer-B0 để trả lời các câu hỏi nghiên cứu RQ1–RQ3)*.
5. **Đánh giá và trực quan hóa:** Khôi phục dự đoán 2D thành khối mask 3D về đúng hệ tọa độ gốc, tính `Dice`, `IoU`, `HD95` theo ca và trực quan hóa overlay với ground truth.

---

## CHƯƠNG 2. DỮ LIỆU BRATS 2020 VÀ TIỀN XỬ LÝ

### 2.1. Mô tả và kiểm định dữ liệu BraTS 2020
- **Nguồn dữ liệu:** Bộ dữ liệu NIfTI 3D chuẩn hóa của MICCAI BraTS 2020 gồm **369 ca bệnh nhân có nhãn** (293 ca bậc cao HGG và 76 ca bậc thấp LGG).
- **Thành phần mỗi ca:** 4 chuỗi MRI (`t1`, `t1ce`, `t2`, `flair`) và 1 mặt nạ phân đoạn (`seg`). Mỗi chuỗi cung cấp thông tin bổ trợ nhau: `T1` thể hiện giải phẫu mô não, `T1ce` làm nổi bật vùng lõi u ngấm thuốc tương phản, `T2` và `FLAIR` làm rõ vùng phù nề bao quanh khối u.
- **Kết quả kiểm tra toàn vẹn (`nifti_audit.csv`):**
  - Toàn bộ **369/369 ca đều hợp lệ** (0 ca lỗi), có cùng kích thước lưới `(240, 240, 155)`, độ phân giải không gian đẳng hướng `(1.0, 1.0, 1.0) mm`, lưới trực giao và hệ trục gốc `LPS`. *(Lưu ý kỹ thuật: ca `BraTS20_Training_355` có file nhãn đặt tên `W39_1998.09.19_Segm.nii` thay vì hậu tố `_seg.nii`, đã được bộ nạp file xử lý tự động theo tên thành phần.)*
- **Gộp nhãn Whole Tumor (WT):** Nhãn gốc của BraTS 2020 gồm nền (`0`), hoại tử/lõi không ngấm thuốc (`1` – chiếm `22.22%` voxel WT), vùng phù nề (`2` – chiếm `57.99%` voxel WT) và lõi ngấm thuốc (`4` – chiếm `19.79%` voxel WT). Nhóm gộp các nhãn `{1, 2, 4}` thành `WT = 1` và nền = `0`.

### 2.2. Thống kê và trực quan hóa dữ liệu (EDA)
Kết quả thống kê trên toàn bộ 369 ca (`eda_brats2020_summary.json` và `eda_brats2020_cases.csv`):

| Chỉ số thống kê | Giá trị trên BraTS 2020 (369 ca) |
| :--- | :--- |
| Tổng số lát axial | `57.195` lát (`155` lát/ca) |
| Số lát có khối u | `24.422` lát (**42,70%**) |
| Số lát có u trên mỗi ca | Trung bình `66,18 ± 15,05` (min `23`, max `102`) |
| Thể tích khối u WT | Trung bình `99,55 ± 59,43 cm³`; Trung vị `90,74 cm³` (`7,29` – `361,78 cm³`) |
| Tỷ lệ voxel WT trong vùng não | **6,87%** (93,13% là mô não lành) |
| Cường độ vùng não thô (Mean / P50 / P95) | `T1`: `675,1 / 696,1 / 910,6` · `T1ce`: `788,3 / 795,0 / 1064,1`<br>`T2`: `769,9 / 700,1 / 1371,1` · `FLAIR`: `460,4 / 462,7 / 696,8` |

**Nhận xét từ 4 biểu đồ EDA (`results/figures/`):**
1. **Phân bố thể tích khối u (`eda_01_tumor_volume_dist.png`):** Thể tích WT phân bố lệch phải khá mạnh, dao động từ các khối u rất nhỏ (~`7,3 cm³`) tới rất lớn (~`361,8 cm³`), đòi hỏi mô hình phải học tốt ở nhiều tỷ lệ kích thước khác nhau.
2. **Phân bố cường độ tín hiệu (`eda_02_intensity_distributions.png`):** Cường độ MRI thô của cả 4 chuỗi có thang đo khác nhau và đuôi dài tới `8.000–10.000` (do không có đơn vị chuẩn tuyệt đối như ảnh CT). Vì vậy bước chuẩn hóa `z-score` riêng từng kênh trên vùng não là bắt buộc.
3. **Mất cân bằng lớp (`eda_03_class_imbalance_slices.png`):** Có tới **57,3%** số lát axial không chứa khối u (đặc biệt ở hai vùng cực đầu/cuối `z < 20` và `z > 140` tỷ lệ có u gần như `0%`), và ở mức voxel trong vùng não chỉ có **6,87%** là khối u. Điều này giải thích lý do phải dùng hàm loss kết hợp `Dice + BCE` và chiến lược lấy mẫu lát có trọng số khi huấn luyện.
4. **Trực quan hóa ca mẫu (`eda_04_sample_cases.png`):** Minh họa 3 ca đại diện (u nhỏ `BraTS20_Training_314`, u trung bình `BraTS20_Training_230`, u lớn `BraTS20_Training_026`) cho thấy `T1ce` làm nổi rõ lõi u ngấm thuốc trong khi `FLAIR` và `T2` thể hiện rõ vùng phù nề bao phủ toàn bộ WT.

### 2.3. Chia tập theo bệnh nhân (`configs/splits.json`)
Để **chống rò rỉ dữ liệu (data leakage)** giữa các lát cắt liền kề của cùng một bệnh nhân, nhóm sắp xếp danh sách 369 `case_id` tăng dần rồi chia **trước khi tách lát** với `seed = 42` theo tỷ lệ `70% / 15% / 15%`:

| Tập dữ liệu | Số ca | Thể tích WT (`Mean ± SD`, cm³) | Trung vị WT (cm³) | Số lát có u/ca (`Mean ± SD`) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | `259` | `99,48 ± 57,48` | `87,61` | `66,56 ± 15,08` |
| **Validation** | `55` | `101,24 ± 67,31` | `89,52` | `65,04 ± 14,62` |
| **Test** | `55` | `98,17 ± 61,22` | `95,82` | `65,56 ± 15,52` |

*(Nhận xét: Giá trị trung bình, độ lệch chuẩn và trung vị thể tích u giữa 3 tập gần như tương đương nhau, xác nhận phép chia không gây lệch phân phối.)*

### 2.4. Pipeline tiền xử lý và tăng cường dữ liệu (Quy ước chung v1.1)
Dữ liệu BraTS 2020 đã được cắt sọ sẵn (skull-stripped), tuy nhiên vẫn còn phần nền đen rộng xung quanh đầu. Nhóm áp dụng pipeline 7 bước đồng bộ cho toàn bộ dữ liệu:
1. **Kiểm tra lưới không gian:** Xác nhận 4 chuỗi MRI và mask có cùng `shape` và `affine`.
2. **Chuẩn hóa hướng `RAS+`:** Dùng `nibabel.orientations` đưa đồng bộ 4 chuỗi và mask từ `LPS` về `RAS+`; lưu ma trận biến đổi `ras_ornt` vào `metadata/<case_id>.json` để khôi phục chính xác về lưới gốc khi đánh giá 3D.
3. **Tạo vùng não sơ bộ & Crop khung bao:** Hợp các voxel khác `0` của cả 4 chuỗi (`any(channel != 0)`) để xác định vùng não (tuyệt đối không dùng ground truth khi crop); lấy bounding box theo `X, Y` cộng thêm lề `5` voxel và cắt đồng bộ trên toàn bộ `Z = 155` lát.
4. **Chuẩn hóa Z-score:** Tính `mean` và `std` **chỉ trên các voxel khác 0** của từng chuỗi trên toàn ca rồi chuẩn hóa `(x - mean) / (std + 1e-8)`, giữ nguyên nền ngoài não bằng `0`.
5. **Padding đối xứng & Resize `224×224`:** Pad đối xứng bằng `0` thành hình vuông rồi nội suy về kích thước `224×224` bằng PyTorch (`bilinear` với `align_corners=False` cho MRI; `nearest-neighbor` cho mask). Lý do chọn `224×224` (thay cho `240×240`) là để chia hết cho `32`, tương thích trực tiếp với encoder 5 tầng của U-Net++, SegFormer-B0 và ViT backbone của TransUNet.
6. **Tách lát 2D & Lấy mẫu tập Train:** Mỗi lát chuẩn hóa thành tensor `float32` kích thước `[4, 224, 224]` theo thứ tự kênh `[T1, T1ce, T2, FLAIR]`. Khi train, loại bỏ các lát hoàn toàn không có mô não và giữ 100% lát có u kết hợp lấy mẫu ngẫu nhiên 50% số lát não không có u mỗi epoch để giảm mất cân bằng lớp mà vẫn hạn chế dự đoán dương tính giả. Tập `val` và `test` giữ đủ toàn bộ `155` lát/ca.
7. **Data Augmentation (chỉ áp dụng cho Train):** Dùng `albumentations` đồng bộ trên 4 kênh và mask gồm: lật ngang (`p=0.5`), lật dọc (`p=0.5`), biến đổi Affine nhẹ (`scale ∈ [0.9, 1.1]`, dịch chuyển `±5%`, xoay `±15°`, `p=0.5`), chỉnh hệ số tương phản/độ sáng (`[0.9, 1.1]`, `p=0.3`) và nhiễu Gaussian (`std ∈ [0.01, 0.05]`, `p=0.2`) chỉ tác động lên vùng não khác `0`.

---

## CHƯƠNG 3. PHƯƠNG PHÁP – MÔ HÌNH U-NET++ 2D

### 3.1. Kiến trúc U-Net++ 2D
Nhóm lựa chọn **U-Net++** (Nested U-Net) làm mô hình nền tảng đại diện cho họ CNN trong giai đoạn giữa kỳ:
- **Ý tưởng cốt lõi:** Thay thế kết nối nhảy trực tiếp (skip connection) đơn giản của U-Net truyền thống bằng hệ thống đường nối lồng nhau dày đặc (dense nested skip pathways), giúp thu hẹp khoảng cách ngữ nghĩa (semantic gap) giữa đặc trưng nông ở encoder và đặc trưng sâu ở decoder — đặc biệt hữu ích với khối u não có biên mờ và kích thước biến thiên lớn.
- **Cấu hình triển khai:** Sử dụng kiến trúc `UnetPlusPlus` với encoder `resnet34` (trọng số tiền huấn luyện ImageNet), sửa đổi lớp đầu vào nhận `4` kênh MRI (`[B, 4, 224, 224]`) và đầu ra `1` kênh logits (`[B, 1, 224, 224]`, không đặt Sigmoid ở lớp cuối để tính BCE ổn định số học).

### 3.2. Hàm mất mát (Loss) và thiết lập huấn luyện
- **Hàm Loss kết hợp:**
  $$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCEWithLogits}}(\hat{y}_{\text{logits}}, y) + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}(\sigma(\hat{y}_{\text{logits}}), y)$$
  Trong đó `BCEWithLogitsLoss` giúp ổn định gradient ở giai đoạn đầu và trên các lát không có khối u, còn `SoftDiceLoss` (tính theo batch với `smooth = 1e-5`) tối ưu trực tiếp độ chồng lấp vùng u trong điều kiện mất cân bằng lớp mạnh (`6,87%` voxel u).
- **Thiết lập tối ưu hóa:**
  - Optimizer: `AdamW` (`lr = 3e-4`, `weight_decay = 1e-4`), `batch_size = 32`.
  - Learning rate scheduler: `CosineAnnealingLR` (`eta_min = 1e-6`).
  - Mixed Precision (`torch.amp`) trên GPU NVIDIA L4 (24GB VRAM, Google Colab Pro).
  - Số epoch tối đa: `50`, kết hợp **Early Stopping (`patience = 10`)** theo chỉ số **Val 3D Dice** tính ở mức bệnh nhân trên 55 ca validation.

### 3.3. Quy trình khôi phục mask 3D sau suy luận
Để đánh giá công bằng trên không gian giải phẫu thực (đơn vị `mm`), mọi lát dự đoán của tập test được khôi phục theo đúng trình tự ngược của tiền xử lý:
1. Suy luận bản đồ xác suất `sigmoid(logits)` kích thước `224×224` cho đủ `Z` lát.
2. Nội suy bilinear (`align_corners=False`) về kích thước vuông trước resize (`padded_shape_xy`), bỏ padding (`pad_xy`).
3. Đặt lại vào đúng vùng `crop_bbox` trên lưới `RAS+` (ngoài vùng crop đặt xác suất bằng `0`).
4. Ghép theo trục `Z`, đảo phép biến đổi hướng `ras_ornt` về lưới gốc `orig_shape`.
5. **Áp ngưỡng nhị phân `>= 0.5` sau cùng** và xuất `<case_id>_pred_wt.nii.gz` (`uint8`) cùng `orig_affine` gốc.

---

## CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ *(ĐÃ ĐIỀN SỐ LIỆU HUẤN LUYỆN, CHỜ MODULE EVAL/VIZ CỦA B)*

### 4.1. Quá trình huấn luyện U-Net++ 2D
- Mô hình hội tụ rất nhanh và đạt **Val 3D Dice tốt nhất = `0,9124 ± 0,0412`** tại **Epoch 22** (`train_loss = 0,03266`, `val_loss = 0,26111`), sau đó kích hoạt Early Stopping (`patience = 10`) sau epoch `31`.
- Thời gian huấn luyện trung bình mỗi epoch: **~`231,2` giây (~`3,85` phút/epoch)** trên GPU NVIDIA L4.
- **Giải thích đường cong `training_curves.png`:** Giá trị `val_loss` ổn định quanh `0,261` trong khi `train_loss` giảm xuống `0,029` **không phải do overfitting** (vì `Val 3D Dice` vẫn giữ ở mức cao `~0,91`). Nguyên nhân là do tập train lọc lát không não và chỉ lấy 50% lát âm (batch luôn có u), còn tập val giữ đủ `155` lát theo thứ tự tuần tự nên các batch ở hai cực không có u khiến `SoftDiceLoss` batch-level trên các batch rỗng đó tiến về `1.0`.

### 4.2. Kết quả định lượng trên tập Test (55 ca BraTS 2020)
- **Kết quả kiểm tra sơ bộ sau khôi phục (`pred_summary_test.csv`):** Đã xuất đủ `55/55` file `<case_id>_pred_wt.nii.gz`, **không có ca nào dự đoán rỗng (`0/55`)**. Giá trị Dice tham khảo trên lưới `224×224` đạt **`0,8978 ± 0,0681`** (trung vị **`0,9199`**).
- **Bảng kết quả CHÍNH THỨC trên lưới gốc (chờ chạy qua module `src/eval/` của Thành viên B):**

| Mô hình | Dice (3D) ↑ | IoU (3D) ↑ | HD95 (mm) ↓ | Số ca dự đoán rỗng |
| :--- | :---: | :---: | :---: | :---: |
| **U-Net++ 2D** | `[Chờ module eval của B]` | `[Chờ module eval của B]` | `[Chờ module eval của B]` | `0 / 55` |

### 4.3. Trực quan hóa Overlay và phân tích lỗi ban đầu
*(Chạy 55 file `*_pred_wt.nii.gz` qua `src/viz/` của Thành viên B để xuất hình overlay cho các ca nổi bật dưới đây:)*
- **Nhóm ca dự đoán tốt nhất:** `BraTS20_Training_347` (quick Dice `0,964`), `BraTS20_Training_112` (`0,963`), `BraTS20_Training_362` (`0,959`).
- **Nhóm ca trung bình:** `BraTS20_Training_126` (`0,900`), `BraTS20_Training_173` (`0,899`).
- **Nhóm ca thấp nhất cần soi kỹ bằng overlay:** `BraTS20_Training_048` (`0,634`, thể tích dự đoán `14.016` voxel), `BraTS20_Training_275` (`0,699`, `21.644` voxel), `BraTS20_Training_058` (`0,748`, `76.324` voxel) → **tạo tiền đề cho giả thuyết dùng ngữ cảnh 2.5D (3 hoặc 5 lát lân cận) ở giai đoạn cuối kỳ (RQ1)**.

### 4.4. Hạn chế hiện tại và kế hoạch cuối kỳ (Phần A)
- Mô hình 2D xử lý độc lập từng lát cắt nên chưa khai thác được tính liên tục không gian dọc trục Z ở hai đầu khối u (lát đầu/cuối) và dễ phát sinh cụm dương tính giả nhỏ rời rạc.
- Hướng phát triển cuối kỳ: mở rộng sang **U-Net++ 2.5D** (3 và 5 lát lân cận), áp dụng **hậu xử lý 3D** (giữ thành phần liên thông lớn nhất), thử nghiệm ablation và kiểm thử khả năng tổng quát hóa trên bộ **BraTS-Africa**.

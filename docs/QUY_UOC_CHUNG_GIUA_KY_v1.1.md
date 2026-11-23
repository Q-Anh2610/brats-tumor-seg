# QUY ƯỚC CHUNG GIỮA KỲ (Phiên bản 1.1 – Đề xuất cập nhật từ A)

**Đề tài:** Phân đoạn khối u não từ ảnh MRI  
**Dành cho:** Thành viên A và B | **Cập nhật từ:** Phiên bản 1.0 (10/11/2026)

> **Tóm tắt các điểm cập nhật so với v1.0 (để A và B chốt nhanh trước khi khóa cấu hình):**
> 1. Đổi kích thước resize từ `240 × 240` → **`224 × 224`** (chia hết cho 32, tương thích trực tiếp với U-Net++, SegFormer-B0 và TransUNet).
> 2. Chốt cụ thể công thức và trọng số hàm loss **`0.5 * BCEWithLogits + 0.5 * SoftDiceLoss`**.
> 3. Chốt bộ phép **Augmentation** mặc định cho tập train.
> 4. Chốt schema JSON cụ thể cho **`splits.json`** và **`metadata/<case_id>.json`** để module tiền xử lý (A) khớp 100% với module khôi phục 3D & `eval`/`viz` (B).
> 5. Bổ sung phương án lưu bộ dữ liệu đã tiền xử lý trên **Kaggle Private Dataset** để cả hai kéo vào Colab nhanh.

> **Điểm cần xác nhận trước khi khóa cấu hình:** Mã nhãn BraTS-Africa phải được đối chiếu với dữ liệu và tài liệu bản phát hành trên toàn bộ 146 ca. Mọi giá trị ghi "dự kiến" chưa phải kết quả xác minh.

---

## 1. Bài toán và phạm vi giữa kỳ
- Phân đoạn nhị phân toàn bộ khối u (Whole Tumor – WT): nền = `0`, vùng u = `1`.
- Đầu vào gồm bốn chuỗi MRI của cùng bệnh nhân. Xử lý lát 2D axial, sau đó ghép và khôi phục mask 3D.
- **A** phụ trách dữ liệu BraTS 2020 và **U-Net++ 2D** làm mô hình báo cáo chính. **B** phụ trách dữ liệu BraTS-Africa, module đánh giá chung (`eval`) và trực quan hóa (`viz`).

---

## 2. Dữ liệu và vai trò sử dụng

| Bộ dữ liệu | Vai trò |
| :--- | :--- |
| **BraTS 2020** | Chia train/validation/test; huấn luyện, chọn checkpoint và đánh giá chính. |
| **BraTS-Africa** | Thống kê, so sánh phân bố và chuẩn bị kiểm thử ngoài nguồn. |

- Không trộn BraTS-Africa vào train hoặc validation. Không dùng kết quả trên Africa để chọn checkpoint, ngưỡng dự đoán hay tham số tiền xử lý.

---

## 3. Thứ tự kênh đầu vào

| Kênh | Chuỗi MRI | BraTS 2020 | BraTS-Africa |
| :---: | :--- | :--- | :--- |
| `0` | T1 | `t1` | `t1n` |
| `1` | T1ce | `t1ce` | `t1c` |
| `2` | T2 | `t2` | `t2w` |
| `3` | FLAIR | `flair` | `t2f` |

- Dataloader tìm file theo tên thành phần, không dựa vào thứ tự liệt kê file trong thư mục.

---

## 4. Gộp nhãn thành WT
- **BraTS 2020:** gộp các nhãn `1, 2, 4` thành `WT = 1`; nền = `0`.
- **BraTS-Africa:** dự kiến gộp các nhãn `1, 2, 3` thành `WT = 1` (chỉ áp dụng sau khi B xác nhận trên dữ liệu thực và tài liệu bản phát hành).
- Kiểm tra tập mã nhãn trên toàn bộ dữ liệu. Một ca riêng lẻ có thể không chứa đủ mọi loại nhãn u.
- Nếu xuất hiện mã ngoài danh sách đã xác nhận, báo lỗi để xem lại; **không** tự động gộp mọi giá trị khác `0`.

---

## 5. Pipeline tiền xử lý dùng chung

1. Kiểm tra bốn chuỗi MRI và mask cùng kích thước và cùng lưới không gian (shape, affine).
2. Đưa về hướng **RAS+** bằng hoán vị/lật trục đồng bộ cho ảnh và mask (dùng `nibabel.orientations`); lưu phép biến đổi (`ras_ornt`) để đảo lại. Không coi đổi hướng trục là đăng ký ảnh. Ca có lưới nghiêng hoặc khác lưới cần kiểm tra riêng trước khi dùng pipeline axial.
3. Tạo vùng não sơ bộ bằng hợp các voxel khác `0` của bốn chuỗi MRI (`brain_mask = any(channel != 0)`). Không dùng ground truth để chọn vùng crop.
4. Tính bounding box theo X và Y trên toàn ca, thêm lề `5` voxel và giới hạn trong ảnh. Dùng cùng một crop cho mọi lát, mọi kênh và mask. Giữ nguyên toàn bộ số lát Z.
5. Chuẩn hóa **z-score** riêng từng kênh theo toàn ca, dùng các voxel khác `0` của kênh đó **trước** crop/padding:
   ```text
   x_norm = (x - mean) / (std + 1e-8)
   ```
   Giữ các voxel nền bằng `0`.
6. Padding đối xứng thành hình vuông (giá trị pad = `0`). Nếu số pixel cần thêm là lẻ, phần dư đặt ở cuối trục (`x_after` / `y_after`).
7. **Resize về `224 × 224`** *(cập nhật từ `240 × 240` vì `224` chia hết cho `32`, tương thích trực tiếp với encoder 5 tầng của U-Net++, SegFormer-B0 và ViT backbone của TransUNet mà không cần pad nội bộ)*.
8. Nội suy: MRI dùng **bilinear**; mask dùng **nearest-neighbor**. Dùng chung PyTorch (`torch.nn.functional.interpolate`), với bilinear đặt `align_corners=False`.

Nếu vùng não rỗng, dữ liệu không hợp lệ hoặc ảnh không cùng lưới, ghi lỗi vào log để kiểm tra trước khi đưa vào huấn luyện.

### Cấu hình Augmentation (chỉ áp dụng cho tập Train)
- Dùng chung thư viện `albumentations`, biến đổi hình học đồng bộ giữa 4 kênh MRI và mask (Validation, Test và Africa **không** áp dụng augmentation):
  - `HorizontalFlip(p=0.5)`
  - `VerticalFlip(p=0.5)`
  - Xoay / tịnh tiến / co giãn nhẹ: `Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), cval=0, cval_mask=0, p=0.5)`
  - Biến đổi cường độ (chỉ áp dụng lên ảnh MRI trên vùng khác `0`, giữ nguyên nền `0` và không áp dụng lên mask):
    - Nhân hệ số tương phản/độ sáng nhẹ trong `[0.9, 1.1]` (`p=0.3`).
    - Nhiễu Gaussian nhẹ với `std ∈ [0.01, 0.05]` (`p=0.2`).
- Cấu hình cụ thể được lưu trong `configs/train_config.yaml` trước khi chạy.

---

## 6. Chia tập theo bệnh nhân
- **Seed chia tập:** `42`. **Tỷ lệ mục tiêu:** `70%` train / `15%` validation / `15%` test.
- Nếu đủ 369 ca hợp lệ: **259 train, 55 validation, 55 test**. Sắp xếp `case_id` tăng dần trước khi xáo trộn và lưu kết quả vào `configs/splits.json`.
- Chia tập trước khi tách lát. A và B dùng chung `splits.json`, không tự chia lại. Nếu loại ca lỗi, chốt lại danh sách trước huấn luyện.
- Validation dùng chọn checkpoint. Test chỉ dùng đánh giá sau khi đã chốt mô hình và thiết lập.

### Schema chuẩn của `configs/splits.json`
```json
{
  "dataset": "BraTS2020",
  "seed": 42,
  "ratios": {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
  },
  "counts": {
    "train": 259,
    "val": 55,
    "test": 55
  },
  "excluded_cases": [],
  "splits": {
    "train": ["BraTS20_Training_001"],
    "val": ["BraTS20_Training_002"],
    "test": ["BraTS20_Training_003"]
  }
}
```

---

## 7. Định dạng ảnh, mask, đầu ra mô hình và hàm Loss

| Thành phần | Quy ước |
| :--- | :--- |
| Một mẫu ảnh | `float32`; `[4, 224, 224]` |
| Một mask | `float32`; `[1, 224, 224]`; giá trị `0/1` |
| Batch ảnh | `[B, 4, 224, 224]` |
| Đầu ra mô hình | Logits `[B, 1, 224, 224]` (không đặt Sigmoid ở lớp cuối) |
| Xác suất | `probs = sigmoid(logits)` |
| Ngưỡng nhị phân | `probs >= 0.5` |

### Hàm Loss dùng chung
- Công thức:
  ```text
  Loss = 0.5 * BCEWithLogitsLoss(logits, target) + 0.5 * SoftDiceLoss(sigmoid(logits), target)
  ```
- **SoftDiceLoss:** tính trên toàn batch (batch-level) với `smooth = 1e-5` để ổn định gradient trên các lát không có khối u. Lưu ý: `smooth` chỉ dùng trong hàm loss khi huấn luyện, **không** dùng trong hàm đánh giá (Mục 10).
- Mỗi mẫu luôn kèm `case_id` và `slice_index` để ghép đúng bệnh nhân và đúng vị trí.

---

## 8. Đánh số lát và Schema `metadata/<case_id>.json` bắt buộc
- Sau khi đưa về RAS+, volume có thứ tự `[X, Y, Z]`. Lát axial `z` được lấy bằng `volume[:, :, z]`.
- Đánh số lát từ `0` đến `Z - 1`. Validation/test phải dự đoán **tất cả lát**, kể cả lát không có u.

### Schema chuẩn của `metadata/<case_id>.json`
```json
{
  "case_id": "BraTS20_Training_001",
  "dataset": "BraTS2020",
  "ref_image_path": "BraTS20_Training_001/BraTS20_Training_001_t1.nii",
  "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
  "orig_shape": [240, 240, 155],
  "orig_affine": [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 239.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "orig_zooms_mm": [1.0, 1.0, 1.0],
  "spatial_units": "mm",
  "orig_axcodes": ["L", "P", "S"],
  "ras_ornt": [[0, -1], [1, -1], [2, 1]],
  "ras_shape": [240, 240, 155],
  "crop_bbox": {
    "x_start": 40,
    "x_end": 200,
    "y_start": 30,
    "y_end": 210
  },
  "crop_shape_xy": [160, 180],
  "pad_xy": {
    "x_before": 10,
    "x_after": 10,
    "y_before": 0,
    "y_after": 0
  },
  "padded_shape_xy": [180, 180],
  "resized_shape_xy": [224, 224],
  "num_slices_z": 155,
  "interpolation": {
    "image": "bilinear",
    "mask": "nearest",
    "align_corners": false
  }
}
```
*(Ghi chú: `x_start`, `y_start` bao gồm; `x_end`, `y_end` không bao gồm. `ras_ornt` là ma trận biến đổi hướng từ ảnh gốc sang RAS+ xuất từ `nibabel.orientations.ornt_transform`, dùng để đảo ngược trực tiếp bằng `nibabel.orientations.inv_ornt_aff` / `apply_orientation`.)*

---

## 9. Khôi phục dự đoán thành mask 3D
- **Thứ tự:** xác suất `224 × 224` → resize về `padded_shape_xy` → bỏ padding theo `pad_xy` → đặt lại vào vùng `crop_bbox` của ảnh `ras_shape[:2]` → ghép lát theo Z → đảo phép đổi hướng (`ras_ornt`) về lưới gốc `orig_shape` → áp ngưỡng `>= 0.5` → lưu mask 3D.
- Nội suy bản đồ xác suất bằng bilinear (`align_corners=False`); ngoài vùng crop đặt xác suất bằng `0`. **Chỉ áp ngưỡng sau khi khôi phục.**
- Lưu dưới tên `<case_id>_pred_wt.nii.gz`, kiểu `uint8`, cùng `orig_shape` và `orig_affine` với ảnh tham chiếu gốc. Không ghi affine gốc lên mảng vẫn còn ở lưới crop/resize.

---

## 10. Đánh giá Dice, IoU và HD95
Đánh giá trên **mask 3D đã khôi phục về lưới gốc**, theo từng bệnh nhân. Ground truth WT cũng ở cùng lưới. Không lấy trung bình Dice từng lát để thay Dice theo ca.

| Độ đo | Định nghĩa |
| :--- | :--- |
| **Dice** | `2 × |P ∩ G| / (|P| + |G|)` |
| **IoU** | `|P ∩ G| / |P ∪ G|` |
| **HD95** | Khoảng cách biên đối xứng, đơn vị `mm` |

- `P` là mask dự đoán nhị phân, `G` là ground truth WT. Xử lý mẫu số bằng `0` theo bảng mask rỗng (Mục 11), không dùng epsilon để thay quy ước đánh giá.
- **Biên** = mask trừ erosion nhị phân, lân cận 6 hướng trong 3D; ngoài thể tích xem là nền.
- **HD95** = `max(percentile95(P đến G), percentile95(G đến P))`, dùng percentile nội suy tuyến tính ở cả hai chiều.
- Khoảng cách dùng `spacing` (`orig_zooms_mm`) theo đúng thứ tự trục và đơn vị `mm` (`scipy.ndimage.distance_transform_edt` với tham số `sampling`). Kiểm tra đơn vị header, đổi sang `mm` nếu cần; xác nhận lưới trực giao trước khi tính.

---

## 11. Mask rỗng và tổng hợp kết quả

| Ground truth | Dự đoán | Dice | IoU | HD95 |
| :--- | :--- | :---: | :---: | :---: |
| Rỗng | Rỗng | `1` | `1` | `0 mm` |
| Có u | Rỗng | `0` | `0` | `∞` |
| Rỗng | Có u | `0` | `0` | `∞` |
| Có u | Có u | Tính | Tính | Tính |

- **Dice và IoU:** trung bình ± độ lệch chuẩn theo ca; mỗi ca có trọng số bằng nhau. Độ lệch chuẩn mẫu dùng `ddof=1`; nếu chỉ có một ca thì SD ghi `N/A`.
- **HD95:** Nếu mọi HD95 hữu hạn, báo trung bình ± SD trên toàn bộ ca. Nếu có HD95 vô hạn, ghi rõ tổng hợp toàn bộ có giá trị vô hạn; báo bổ sung trung bình ± SD trên nhóm hữu hạn, kèm số ca hữu hạn / tổng số ca và số ca thất bại. Không thay vô hạn bằng `0` hoặc âm thầm bỏ ca.
- **Bảng CSV từng ca gồm các cột:** `case_id, dice, iou, hd95_mm, gt_empty, pred_empty`. Lưu vô hạn bằng `inf`; giữ số liệu thô để kiểm tra lại.

---

## 12. Khóa cấu hình, lưu trữ và bàn giao
- Dùng chung `configs/splits.json`, `configs/train_config.yaml`, cấu hình tiền xử lý và module `eval`/`viz`. Ghi phiên bản thư viện (`requirements.txt`) và seed thí nghiệm.
- **Chia sẻ dữ liệu đã tiền xử lý:** Sau khi chốt cấu hình, đẩy bộ dữ liệu đã tiền xử lý + `metadata/` lên **Kaggle Private Dataset** (share giữa A và B) để cả hai kéo trực tiếp vào Colab Pro nhanh và đồng bộ.
- B xác nhận mã nhãn BraTS-Africa sau kiểm tra 146 ca; A và B khóa cấu hình v1.1 trước khi huấn luyện. Thay đổi sau đó phải cập nhật chung và xác định kết quả nào cần chạy lại.

### Tài liệu tham khảo kỹ thuật
1. NiBabel – Coordinate systems and affines: https://nipy.org/nibabel/coordinate_systems.html
2. SciPy – `distance_transform_edt` (tham số `sampling`): https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

SO SÁNH BRATS 2020 VÀ BRATS-AFRICA, PCA GIỮA KỲ

1. CHẠY TRÊN COLAB

Tải brats_africa_compare.zip và giải nén trên máy, giữ cả ZIP.
Vào https://colab.research.google.com/ rồi File > Upload notebook.
Chọn notebooks/B_africa_compare.ipynb trong thư mục vừa giải nén.
Dùng runtime CPU và chạy lần lượt các ô 1 đến 10, không cần thêm code thủ công.

Ô 1, chọn brats_africa_compare.zip khi được yêu cầu bộ mã.
Ô 2, cài thư viện. Nếu Colab yêu cầu restart thì thực hiện rồi chạy lại từ ô 1.
Ô 3, kết nối Drive để lưu kết quả trong MyDrive/brats_africa_midterm/africa_compare_run01.
Ô 4, chọn kaggle.json nếu runtime chưa có xác thực.
Ô 5, lấy bảng EDA Africa từ Kaggle phiên bản 1, chỉ khoảng 80 KB, kiểm tra checksum các đầu vào.
Ô 6, kiểm tra ID và split, ghép 515 ca, xuất bảng so sánh ba nhóm.
Ô 7, vẽ ba hình so sánh phân bố.
Ô 8, chạy PCA chính và PCA bổ sung, hiển thị hai hình.
Ô 9, xem phương sai giải thích, hệ số đặc trưng và kiểm tra đúng 259 ca fit.
Ô 10, tải africa_compare_run01.zip về máy.

Chọn File > Save a copy in Drive để giữ notebook cùng output sau khi chạy.
Chạy lại cùng tên run01 sẽ ghi đè kết quả cùng tên, đổi run02 nếu làm thử nghiệm mới.
Không cần dữ liệu MRI gốc hoặc bộ NPZ để chạy notebook này.

2. DỮ LIỆU ĐẦU VÀO

results/eda_brats2020_cases.csv và configs/splits.json là bản chụp từ main của repo.
Commit tham chiếu e504d38d485b4008aa28fc9e22101687436f5d63.
configs/compare_inputs.json ghi SHA256, nguồn và phiên bản.
Checksum của CSV và JSON chuẩn hóa CRLF về LF để chạy nhất quán trên Windows và Colab.
Bảng Africa lấy từ trinhthuhuyenhuyen/brats-africa-preprocessed-224-v11/1.
Nếu B_africa_data đã tải dữ liệu trong cùng runtime thì dùng lại bảng có checksum khớp.
Dataset riêng tư, tài khoản trong kaggle.json cần có quyền xem.

Các thống kê MRI được tính trên voxel khác 0 của NIfTI đầu vào trước z-score chung.
Dữ liệu nguồn đã có các bước xử lý của challenge, không gọi các bảng này là cường độ ảnh chụp chưa xử lý.
Thể tích WT sử dụng NIfTI gốc và spacing, không tính mL từ mask đã resize.
Giữ riêng 95 glioma và 51 u khác khi so sánh với 369 ca BraTS 2020.

3. PCA

PCA chính, 21 đặc trưng gồm 5 thống kê mỗi MRI và thể tích não, không dùng biến từ WT.
PCA bổ sung, 22 đặc trưng, có thêm thể tích WT gốc nên có thông tin GT.
StandardScaler và PCA chỉ fit trên 259 ca train BraTS 2020.
Val, test và Africa chỉ được transform bằng các tham số đã fit trên train.
PC1/PC2 variance là phương sai giải thích trên train đã chuẩn hóa.
Hình phóng to chỉ đổi giới hạn hiển thị theo phân vị 1–99%, không xóa ca khỏi PCA hoặc CSV.
PCA chưa phải kết quả đánh giá khả năng tổng quát hóa của mô hình phân đoạn.

4. KẾT QUẢ

inputs/, bản sao ba tệp đầu vào và nguồn để tái lập.
combined_case_features.csv, 515 ca với dataset, cohort và split.
cohort_comparison.csv, thống kê WT và tỷ lệ lát, voxel của ba nhóm.
figures/, ba hình phân bố và hai hình PCA.
pca_mri_scores.csv, tọa độ 2D của PCA chính.
pca_mri_loadings.csv, hệ số của đặc trưng đã chuẩn hóa.
pca_mri_transform.json, scaler, PCA, variance và ID train đã dùng.
pca_with_wt_*, bộ kết quả PCA bổ sung có thể tích WT.
pca_summary.csv, bảng tổng hợp hai PCA.
comparison_manifest.json, hash đầu vào, hash script và cấu hình.
README_RESULTS.txt, ghi chú cách diễn giải kết quả.

5. ĐƯA VÀO REPO SAU KHI CHẠY THÀNH CÔNG

Thêm notebooks/B_africa_compare.ipynb, requirements-compare.txt và configs/compare_inputs.json.
Cập nhật src/preprocess/compare_africa.py bằng bản trong bộ mã mới.
Giữ nguyên audit_africa.py hiện có, bản đóng kèm chỉ phục vụ chạy bộ mã độc lập.
Giữ nguyên configs/splits.json và results/eda_brats2020_cases.csv hiện có trong repo.
Giải nén ZIP Colab vào results/africa/africa_compare_run01.
Không đưa kaggle.json vào Git, chưa cần push trước khi chạy Colab.

6. NGUỒN

https://github.com/Q-Anh2610/brats-tumor-seg
https://www.kaggle.com/datasets/trinhthuhuyenhuyen/brats-africa-preprocessed-224-v11
https://www.cancerimagingarchive.net/collection/brats-africa/
https://doi.org/10.7937/v8h6-8x67

# Kết quả kiểm tra pipeline trên server

Host: `ssh-x-lab-0`. Thư mục: `/home/thinhnd39/yolo_fewshot`.

## Dữ liệu

| Tập | Số ảnh |
|---|---:|
| cnmc_train | 3455 |
| cnmc_val | 3554 |
| cnmc_test | 3548 |
| aml_external_test | 7205 |
| bccd_train | 205 |
| bccd_val | 87 |
| bccd_test | 72 |

Đã kiểm tra lại ảnh, hash và nhãn; tách bệnh nhân C-NMC giữa train/val/test; không có pixel trùng giữa các tập giữ lại. Đủ số mẫu cho 20-shot. Ảnh nghi gần trùng chỉ được đánh dấu để review, không tự loại theo dHash.

## Chạy thử

- DINOv2-small: 32 ảnh train và 16 validation, vector CLS 384 chiều, L2-normalized, không có NaN/Inf.
- YOLO11n: 16 ảnh train, 8 validation, 1 epoch, input 320, batch 4; đã train, validate và lưu dự đoán trên 8 ảnh.
- Không dùng test hoặc AML để chọn mô hình trong smoke test.

| Nhánh | GPU | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---:|---:|
| dinov2_memory | NVIDIA H200 | 0.157 | 0.189 |
| yolo_memory | NVIDIA H200 | 0.315 | 0.340 |

Allocator PyTorch bị giới hạn 16 GiB, chừa khoảng cho CUDA dưới yêu cầu 20 GB; hai nhánh chạy tuần tự. Các số peak trên là bộ nhớ PyTorch, không phải tổng bộ nhớ của mọi người dùng trên GPU chung.

**Đây là kiểm tra khả năng chạy, không phải benchmark cuối.** Detection mới chỉ học 1 epoch nên dự đoán còn nhiều false positive; không dùng để đánh giá chất lượng cuối hoặc làm demo chẩn đoán.

Dữ liệu đủ cho classification và định vị WBC riêng biệt. Chưa có box + nhãn ung thư từng tế bào trên ảnh smear để đánh giá malignant detection toàn pipeline; AML khác cả nguồn và dòng bệnh.

Chi tiết model revision, checksum, metric validation và phiên bản phần mềm trong `output/reports/server_pipeline.json`. Chạy lại theo `docs/PIPELINE_RUN.md`.

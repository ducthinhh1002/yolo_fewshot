# Few-shot Blood Cell Analysis

Đã triển khai bước chuẩn bị dữ liệu và giao thức đánh giá cho proposal.

- [Giao thức, lựa chọn và giới hạn dữ liệu](docs/DATA_PROTOCOL.md)
- [Danh mục nguồn, phiên bản và license](docs/DATASET_CATALOG.json)
- [Cấu hình thí nghiệm](configs/data_protocol.json)

Chạy `scripts/download_data.py` rồi `scripts/prepare_data.py` để tái tạo dữ liệu và tập chia.
Xem `data/manifests/summary.json` cho số lượng thực tế sau audit.
Đã có bộ đọc dữ liệu PyTorch, trích đặc trưng DINOv2 và runner chạy thử YOLO trên tập nhỏ. Xem [hướng dẫn chạy trên GPU server](docs/PIPELINE_RUN.md).
Đã có LoRA ở block attention cuối, Prototypical Networks với projection học theo episode và chạy thử DE-ViT chính thức. Xem [so sánh classification](docs/ADAPTATION_RUN.md), [DE-ViT](docs/DEVIT_RUN.md) và [trạng thái mốc 14–20/11](docs/MILESTONE_14_20_STATUS.md).

Bước tiếp theo: [baseline YOLO đầy đủ và DINOv2 nearest-prototype](docs/BASELINE_RUN.md), chạy trên server và đánh giá bằng validation trước khi dùng test.

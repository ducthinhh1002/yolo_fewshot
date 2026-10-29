# Few-shot Blood Cell Analysis

Đã triển khai bước chuẩn bị dữ liệu và giao thức đánh giá cho proposal.

- [Giao thức, lựa chọn và giới hạn dữ liệu](docs/DATA_PROTOCOL.md)
- [Danh mục nguồn, phiên bản và license](docs/DATASET_CATALOG.json)
- [Cấu hình thí nghiệm](configs/data_protocol.json)

Chạy `scripts/download_data.py` rồi `scripts/prepare_data.py` để tái tạo dữ liệu và tập chia.
Xem `data/manifests/summary.json` cho số lượng thực tế sau audit.
Đây là phần dữ liệu/đánh giá; chưa bao gồm huấn luyện YOLO, DINOv2, LoRA hay DE-ViT.

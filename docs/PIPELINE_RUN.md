# Chạy pipeline trên GPU server

Thư mục triển khai: `/home/thinhnd39/yolo_fewshot` trên SSH host `gpu-x-lab`.
Thông tin kết nối và token chỉ nằm trong SSH config của máy người dùng, không nằm trong mã nguồn.

## Chuẩn bị

```bash
cd /home/thinhnd39/yolo_fewshot
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m unittest discover -s tests -v
```

Nếu API Kaggle không truy cập được từ server, `download_data.py --resolved-sources <json>` nhận mapping `cnmc`/`aml` sang URL tải archive công khai đã resolve trên máy có quyền truy cập. URL có thể hết hạn; không commit file mapping. Metadata gốc vẫn được giữ ở `data/provenance` và SHA-256 archive được tính lại sau tải.

## Kiểm tra chạy mô hình

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_smoke.py
# Có thể chạy riêng từng phần, không huấn luyện lại phần đã đạt:
.venv/bin/python scripts/run_smoke.py --stage audit
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_smoke.py --stage dino
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_smoke.py --stage yolo
```

`configs/smoke.json` cố định seed và các lựa chọn:

- DINOv2-small ở revision cụ thể: 32 ảnh train (16/lớp) và 16 ảnh validation (8/lớp). RGB, resize giữ tỷ lệ, padding 224×224, normalize bằng mean/std pretrained. Lấy CLS và chuẩn hóa L2. Không fine-tune.
- YOLO11n pretrained: 16 ảnh BCCD train và 8 validation, một lớp WBC, 1 epoch, input 320, batch 4, workers 0.
- Không huấn luyện trên validation, không dùng tập test hoặc AML để chọn mô hình trong lượt thử này.
- Giới hạn allocator PyTorch **16 GiB** cho mỗi tiến trình, chừa bộ nhớ cho CUDA để giữ mức dùng dưới 20 GB. Hai model chạy tuần tự; log peak allocated/reserved trong `*_memory.json`. Không khởi chạy nhiều bản runner cùng lúc.
- Tên tiến trình trong nvitop/ps là `VLLM:Engine`, đặt bằng `setproctitle`; PID và tên thực tế lưu ở `runtime.json`. Đây vẫn là runner YOLO/DINOv2 của dự án, không phải dịch vụ vLLM.
- Đây là smoke test để chứng minh đọc dữ liệu, forward/backward, validation và lưu kết quả hoạt động; các metric không phải kết quả benchmark cuối.

## Bộ đọc dùng lại cho giai đoạn sau

`bloodcells/data.py` cung cấp `CellDataset`, `WBCDataset`, `detection_collate`, `read_manifest` và `balanced_subset`.
CellDataset trả tensor `[3,224,224]`, nhãn 0/1 và sample/patient ID. WBCDataset trả ảnh gốc `[3,H,W]` trong [0,1], box xyxy 0-based half-open và class WBC=0.
Mọi loader đọc manifest đã khóa; không tự random split hay suy nhãn từ tên file khi huấn luyện.

## Kết quả lưu trên server

- `runs/smoke/audit.json`: số ảnh/nhãn, box WBC, kiểm tra đủ 20-shot, rò rỉ bệnh nhân và pixel trùng giữa split.
- `runs/smoke/cnmc_train_features.npz`, `cnmc_val_features.npz`: features, labels và sample IDs.
- `runs/smoke/*_feature_rows.json`: truy ngược feature về ảnh nguồn.
- `runs/smoke/dinov2.json`, `yolo.json`: model revision/checksum, tập nhỏ đã chọn, trạng thái và metric validation.
- `runs/smoke/*_memory.json`, `environment.json`: bộ nhớ GPU và phiên bản môi trường.
- `runs/smoke/yolo/train/weights/best.pt`: checkpoint chạy thử.
- `runs/smoke/yolo/predictions/`: dự đoán trên validation.

Không commit `data/`, `runs/`, `models/`, `.venv/` hoặc trọng số vào Git.

## Giới hạn dữ liệu

C-NMC đủ cho binary cell classification và 1/5/10/20-shot; BCCD đủ cho chạy thử định vị WBC. BCCD không có patient IDs và không có nhãn ung thư từng box. AML là phép thử đồng thời khác nguồn và khác loại bệnh, không phải external validation B-ALL tương đương. Bộ hiện tại chưa đủ để báo cáo malignant-cell detection mAP toàn pipeline. dHash chỉ đánh dấu ứng viên gần trùng để review, không tự xóa tế bào khác nhau có nền giống nhau.

API mô hình theo [DINOv2 trên Hugging Face](https://huggingface.co/facebook/dinov2-small) và [hướng dẫn Ultralytics](https://docs.ultralytics.com/modes/train/).

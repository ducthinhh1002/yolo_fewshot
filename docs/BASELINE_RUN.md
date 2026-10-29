# Baseline sau bước chuẩn bị pipeline

Phần tuần pipeline đã hoàn thành khi audit dữ liệu, các DataLoader, DINOv2 và YOLO smoke test chạy thành công. Bước tiếp theo là baseline trên tập train đầy đủ và validation, chưa sử dụng test.

## Chạy trên server

```bash
cd /home/thinhnd39/yolo_fewshot
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_baseline.py
```

Tên tiến trình: `VLLM:Engine`. Giới hạn allocator PyTorch 16 GiB. Không chạy đồng thời nhiều bản runner. Cấu hình cố định tại `configs/baseline.json`.

Các stage chạy tuần tự: `features`, `prototype`, `detector`, `demo`. Có thể dùng `--stage <tên>` để chạy riêng sau khi các đầu vào cần thiết đã được tạo. Không chạy trên máy local.

## Nội dung

1. Kiểm tra lock, ảnh tồn tại, nhãn, số mẫu, rò rỉ patient/pixel.
2. Trích DINOv2-small CLS features trên toàn bộ C-NMC train (3.455 ảnh) và validation (3.554 ảnh). RGB, pad giữ tỷ lệ về 224, normalize theo pretrained, vector 384 chiều chuẩn hóa L2. Không fine-tune backbone.
3. Baseline nearest-prototype cosine trên backbone đóng băng: dùng đúng support IDs của 300 trial × 1/5/10/20-shot trong giao thức. Query dùng toàn bộ validation; test và AML chưa được dùng. Mean và SD theo support trials, không gọi SD là CI bệnh nhân. Đây là baseline prototype chưa có episodic training của Prototypical Networks.
4. Huấn luyện YOLO11n pretrained với toàn bộ 205 ảnh BCCD train, 87 validation, tối đa 50 epoch, patience 12, input 640, batch 8, một lớp WBC. Chọn best checkpoint bằng validation, sau đó reload checkpoint best để tính metric.
5. Thử luồng ảnh → YOLO boxes → crop → DINOv2 → prototype trên 8 ảnh BCCD validation, chỉ nhằm kiểm tra tích hợp và định dạng đầu ra.

YOLO ở đây dùng đầy đủ annotation detection. Không coi nó có cùng ngân sách với detector K-shot. Chưa báo kết quả test và chưa so sánh LoRA/DE-ViT.

## Đầu ra

Trong `/home/thinhnd39/yolo_fewshot/runs/baseline_v1/`:

- `features.json`, `cnmc_train_features.npz`, `cnmc_val_features.npz`: đặc trưng toàn bộ train/val cùng sample IDs và nhãn.
- `prototype_validation_trials.csv`, `prototype.json`: kết quả validation ghép cặp theo số shot.
- `yolo/train/weights/best.pt`, `detector.json`: checkpoint tốt nhất và metric validation.
- `yolo/train/results.csv`, `yolo/validation/`: đường học và biểu đồ đánh giá.
- `demo_prototypes.npy`: prototype trial 0, 20-shot, từ support train.
- `pipeline_demo.json`: boxes, detector confidence và classification score chưa calibration trên ảnh demo.

Luồng demo không xác thực chẩn đoán: BCCD không có nhãn ung thư từng tế bào và khác miền dữ liệu C-NMC. Không dùng dự đoán classification trên BCCD làm bằng chứng sensitivity hoặc specificity. Để đánh giá toàn pipeline cần bộ smear có box và nhãn bệnh lý tương ứng.

Sau baseline: xem lỗi định vị/crop, kiểm tra hiệu chỉnh score trên validation; tiếp tục LoRA và Prototypical Networks, thử DE-ViT, rồi mới khóa cấu hình và chạy đánh giá test/cross-source theo kế hoạch.

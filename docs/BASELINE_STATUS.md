# Kết quả baseline trên server

Host `ssh-x-lab-0`, process `VLLM:Engine`; allocator giới hạn 16 GiB. Chỉ train/validation, chưa dùng test.

## Phân loại: frozen DINOv2 + nearest-prototype

Trích đặc trưng toàn bộ 3.455 ảnh train và 3.554 ảnh validation; 384 chiều, CLS L2-normalized. Prototype chỉ sử dụng K ảnh mỗi lớp trong support train; trích toàn bộ train không có nghĩa dùng toàn bộ nhãn cho mỗi trial. Cùng 300 trial ghép cặp theo giao thức.

| Shot/lớp | Macro-F1 mean ± SD | Sensitivity mean | Specificity mean | AUROC mean |
|---|---:|---:|---:|---:|
| 1 | 0.4723 ± 0.1398 | 0.4781 | 0.6259 | 0.6042 |
| 5 | 0.5603 ± 0.1147 | 0.5386 | 0.6674 | 0.6580 |
| 10 | 0.6056 ± 0.1034 | 0.5762 | 0.7106 | 0.7097 |
| 20 | 0.6474 ± 0.0797 | 0.6208 | 0.7421 | 0.7582 |

Đây là baseline không fine-tune, chưa phải Prototypical Networks có episodic training; SD thể hiện biến thiên support, không phải CI bệnh nhân.

## Định vị WBC: YOLO11n

Dùng đủ 205 ảnh train và 87 validation; đã chạy 23 epoch, tối đa 50, patience 12; input 640, batch 8. Đánh giá lại checkpoint best sau khi chọn bằng validation.

- `metrics/precision(B)`: 0.9445
- `metrics/recall(B)`: 0.9781
- `metrics/mAP50(B)`: 0.9867
- `metrics/mAP50-95(B)`: 0.7840
- `fitness`: 0.7840

Baseline này sử dụng annotation detection đầy đủ, không so như cùng ngân sách với detector K-shot.

## Kiểm tra tích hợp

Đã chạy ảnh BCCD validation → boxes WBC → crop → DINOv2 → prototype 20-shot trên 8 ảnh; đầu ra gồm tọa độ, detector confidence, label và classification score chưa calibration.

BCCD không có nhãn ung thư từng tế bào và khác miền dữ liệu classification. Đây chỉ là kiểm tra tích hợp; chưa đủ để kết luận chất lượng phát hiện ung thư toàn pipeline.

## Việc tiếp theo

- Phân tích lỗi detection/crop và độ nhạy với support; kiểm tra preprocessing trên validation.
- Triển khai LoRA và Prototypical Networks, kiểm tra DE-ViT.
- Thực hiện ablation màu/stain và khóa siêu tham số trước đánh giá test/cross-source.

Chi tiết: `output/reports/baseline_server.json`. Cách chạy: `docs/BASELINE_RUN.md`. Trọng số, feature và dữ liệu giữ trên server trong `runs/baseline_v1/`, không đưa vào Git.

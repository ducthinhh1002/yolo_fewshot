# Trạng thái mốc 14–20/11

**Đã hoàn thành mức triển khai/chạy thử theo mốc này.** Không đồng nghĩa hoàn thành thí nghiệm cuối của đề tài.

| Đầu việc | Bằng chứng |
|---|---|
| Prototypical Networks chạy được | Learned projection episodic trên DINOv2 đóng băng, 30 trial ghép cặp mỗi mức shot; xem ADAPTATION_STATUS.md |
| LoRA chạy thử | Q/V block cuối, rank 4, head-only control, cùng 30 trial/shot |
| DE-ViT chạy thử | Mã và checkpoint tác giả, đủ tham số, 1/5/10/20-shot × 8 validation images |
| Chốt support/query | Classification dùng cùng support/validation; detection có configs/detection_protocol.json, tách full supervision khỏi K-shot |
| Danh sách lỗi/rủi ro | docs/RISKS.md và các báo cáo lỗi classification/detection |

## DE-ViT smoke test

Actual `OpenSetDetectorWithExamples`, pretrained ViT-Small + RPN/head của tác giả; pinned commit `9cab390fbe78b3bc8d62c224411dce1c71056cf9`. Không thiếu tham số detector học được. Không thay bằng detector khác hoặc weights ngẫu nhiên.

Query 8 ảnh BCCD validation, 8 box WBC; một support trial. AP tính bằng pycocotools, không gọi là benchmark cuối.

| Shot | AP@50 | AP@50–95 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| 1 | 32.68% | 15.03% | 6 | 80 | 2 |
| 5 | 46.44% | 21.29% | 6 | 85 | 2 |
| 10 | 48.21% | 22.73% | 7 | 90 | 1 |
| 20 | 53.76% | 25.32% | 7 | 92 | 1 |

TP/FP/FN ở confidence 0.25, IoU 0.5. Nhiều false positive; việc load và inference thành công chỉ xác nhận khả thi kỹ thuật. Chưa đủ để chọn DE-ViT làm detector cuối.

Peak reserved PyTorch: 4.807 GiB; allocator cap 16 GiB, tiến trình `VLLM:Engine`, GPU H200.

## Còn lại ở các mốc sau

- Mở rộng query/trial cho detection và huấn luyện YOLO K-shot đúng cùng ngân sách trước khi so sánh chất lượng.
- Tuning trên validation, thử augmentation/stain, tiếp tục đánh giá 300 trial cho các mô hình classification.
- Khóa cấu hình rồi mới dùng test/cross-source; hiện chưa dùng test.
- Hoàn thiện explainability/demo/báo cáo. Dữ liệu hiện chưa có ground truth ung thư từng box để xác thực toàn pipeline.

Hướng dẫn: `docs/DEVIT_RUN.md`. Protocol: `configs/detection_protocol.json`. Rủi ro: `docs/RISKS.md`.

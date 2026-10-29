# Phân tích lỗi và so sánh LoRA / Prototypical Networks

Đánh giá phát triển trên validation C-NMC, 30 trial ghép cặp đầu tiên, cùng K-shot support và 3.554 query; không dùng test/AML. Các con số dưới đây không thay thế bảng baseline 300 trial trước đó.

## Macro-F1 trung bình ± SD

| Phương pháp | 1-shot | 5-shot | 10-shot | 20-shot |
|---|---:|---:|---:|---:|
| Baseline | 46.12% ± 18.08 | 54.31% ± 10.77 | 61.37% ± 8.96 | 63.34% ± 7.01 |
| Chỉ học head | 45.86% ± 17.87 | 56.07% ± 10.66 | 62.63% ± 7.25 | 64.23% ± 7.06 |
| ProtoNet projection | 45.73% ± 17.37 | 55.86% ± 10.99 | 62.81% ± 8.12 | 64.62% ± 7.14 |
| LoRA + head | 44.52% ± 17.59 | 55.73% ± 10.22 | 62.03% ± 6.59 | 64.51% ± 7.19 |

SD là biến thiên theo support; không phải confidence interval theo bệnh nhân. Không kết luận ưu thế thống kê chỉ từ bảng mean.

## Diễn giải

- Cải thiện ở 5/10/20-shot còn nhỏ; các cách thích nghi có thể kém baseline ở 1-shot. Không khẳng định LoRA hoặc ProtoNet tốt hơn ổn định.
- Head-only là đối chứng quan trọng: LoRA học cả adapter và classifier head; không quy toàn bộ thay đổi so với baseline cho adapter.
- LoRA ở block attention cuối, rank 4 trên Q/V, 6.144 adapter parameters + 768 head parameters. Mọi trọng số còn lại đóng băng; reset theo trial/K.
- ProtoNet ở đây học residual projection episodically trên DINOv2 đóng băng. 1-shot dùng hai view cùng ảnh, không có query cell độc lập để học nội bộ. Không phải thiết lập meta-learning trên lớp mới.
- Tính lại baseline bằng đúng phép reduction NumPy của vòng trước để tránh sai khác float ở ngưỡng 0.5; kiểm thử đối chiếu từng trial.

## Lỗi classification ở 20-shot

| Phương pháp | TN trung bình | FP trung bình | FN trung bình | TP trung bình | Tế bào sai ≥80% trial |
|---|---:|---:|---:|---:|---:|
| Baseline | 892.2 | 264.8 | 1006.2 | 1390.8 | 349 |
| Chỉ học head | 868.4 | 288.6 | 945.1 | 1451.9 | 207 |
| ProtoNet projection | 873.3 | 283.7 | 935.6 | 1461.4 | 240 |
| LoRA + head | 843.6 | 313.4 | 902.6 | 1494.4 | 174 |

Một số tế bào sai lặp lại qua nhiều support, cần review thay vì chỉ chọn một trial có điểm cao. Nhóm bệnh nhân baseline có tỷ lệ sai cao nhất:

- `cnmc:all:34`: 352 ảnh; tần suất sai trung bình 51.1%.
- `cnmc:all:33`: 181 ảnh; tần suất sai trung bình 51.0%.
- `cnmc:all:30`: 201 ảnh; tần suất sai trung bình 49.1%.

Các thống kê độ sáng/foreground theo từng nhãn được lưu trong JSON; chỉ là liên hệ mô tả, không chứng minh nguyên nhân sinh học hoặc lỗi nhãn. Những ca sai cần chuyên gia xác nhận.

## Lỗi định vị YOLO trên 87 validation

Confidence cố định 0.25, ghép một-một theo confidence; không phải operating point best-F1 hay mAP.

| IoU | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.5 | 87 | 18 | 0 | 0.829 | 1.000 |
| 0.75 | 84 | 21 | 3 | 0.800 | 0.966 |

Chênh lệch giữa IoU 0.5 và 0.75 giúp nhận ra box chưa khớp chặt dù đã tìm đúng vùng tế bào. FP/FN và ảnh minh họa ở `runs/adaptation_v1/`.

## Phạm vi và bước tiếp theo

Đây là một cấu hình cố định cho vòng validation 30 trial, chưa tuning search và chưa chạy 300 trial cho LoRA/ProtoNet. Không chọn mô hình cuối dựa trên cải thiện nhỏ này. Tiếp theo: review ca lỗi, thử learning rate/số bước và augmentation trên validation, kiểm tra độ ổn định theo support/bệnh nhân, rồi chốt cấu hình trước mở rộng thực nghiệm. Không dùng test để chỉnh tham số.

BCCD chỉ có nhãn WBC, không dùng để suy kết quả ung thư toàn pipeline. AML khác cả nguồn và dòng bệnh.

Peak reserved PyTorch của lượt adaptation: 3.207 GiB; allocator cap 16 GiB, tên tiến trình `VLLM:Engine`.

Chi tiết cách chạy và giới hạn: `docs/ADAPTATION_RUN.md`; số liệu máy đọc: `output/reports/adaptation_server.json`.

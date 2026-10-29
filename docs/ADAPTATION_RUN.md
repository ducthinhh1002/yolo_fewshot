# So sánh thích nghi few-shot và phân tích lỗi

Chạy trên server `/home/thinhnd39/yolo_fewshot`, GPU H200, tên tiến trình `VLLM:Engine`, allocator tối đa 16 GiB. Không chạy mô hình trên local.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_adaptation.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/analyze_errors.py
.venv/bin/python -m unittest discover -s tests -v
```

Có thể dùng `--stage cache` hoặc `--stage experiment`; stage experiment yêu cầu cache đã tạo. Cấu hình chốt trước chạy trong `configs/adaptation.json`. Cache token float32 đầu vào block cuối của DINOv2 được đối chiếu forward đầy đủ với sai lệch tối đa <1e-5; giữ nguyên prefix đóng băng khi thích nghi. Không phải fine-tune mọi block.

## Thiết lập so sánh

- Dùng 30 trial đầu của cùng file episode, mỗi trial có 1/5/10/20 ảnh mỗi lớp; cùng support IDs, cùng 3.554 query validation và threshold 0.5 cho mọi phương pháp. Đây là vòng phát triển validation, chưa phải thực nghiệm cuối 300 trial/method. Baseline 300 trial cũ giữ nguyên; bảng so sánh mới chỉ lấy đúng 30 trial ghép cặp.
- Không dùng test hoặc AML để huấn luyện/chọn cấu hình. Không dùng toàn bộ nhãn train để học metric rồi gọi là K-shot.
- **Baseline:** DINOv2 đóng băng + cosine nearest-prototype như vòng trước.
- **Head-only:** đối chứng chỉ học cosine classifier head trên đúng support K-shot, 15 bước, LR 0.005, logit scale 10.
- **LoRA:** pretrained backbone đóng băng; rank 4, alpha 8 vào Q/V của block attention cuối, 6.144 adapter parameters, cộng 768 tham số cosine classifier head. Reset adapter và head cho mỗi trial/K; 15 bước Adam, LR adapter 0.001, head 0.005, logit scale 10. Cache prefix không thay đổi gradient của block được thích nghi. Không fine-tune layernorm hoặc các block còn lại.
- **Prototypical Networks — biến thể learned projection:** DINOv2 đóng băng, residual MLP 384→64→384, embedding L2-normalized, prototype là trung bình support embedding, logits bằng âm khoảng cách Euclidean bình phương ×10. Học projection bằng episodic cross-entropy trên support, 40 bước Adam LR 0.001. Khởi tạo residual ở identity; reset từng trial/K.
- K≥5: tách các ảnh support có nhãn thành phần support/query nội bộ episode. K=1: support/query là hai view của cùng ảnh gốc (ảnh gốc và mirror + brightness 1.05), **không phải hai tế bào độc lập**. Không tăng số unique shots và không gọi đây là novel-class meta-learning. Hai lớp đã biết ở cả train/val.
- Đây là các biến thể cụ thể, không đại diện cho mọi cách LoRA hoặc episodic fine-tuning backbone. Không chọn cấu hình bằng kết quả test.

## Phân tích lỗi

- Classification: tại 20-shot, thống kê FP/FN trung bình qua 30 support trial, tần suất sai của từng tế bào, biến động score, nhóm bệnh nhân và phân tầng độ sáng/tỷ lệ foreground trong cùng nhãn. Các mối liên hệ này không chứng minh nguyên nhân sinh học.
- YOLO: dùng checkpoint best đã có, trên 87 validation, confidence cố định 0.25; ghép prediction/ground truth một-một theo confidence tại IoU 0.5 và 0.75. Các số này khác operating point best-F1 của Ultralytics và không phải mAP.
- Ảnh minh họa dùng nhãn gốc, không tự gán kết luận y khoa từ hình thái.

## Đầu ra

Trong `runs/adaptation_v1/`: `comparison_trials.csv`, `comparison.json`, `validation_scores.npz`, `classification_errors.csv`, `classification_error_analysis.json`, `detection_error_analysis.json`, hai hình lỗi và checkpoint trial 0 cho mỗi K. Các file token cache khoảng 2.8 GB chỉ nằm trên server, không commit.

Mean và SD theo support trial; không diễn giải SD hoặc win fraction như CI độc lập theo bệnh nhân. Đối chứng head-only cần được xem cùng LoRA; không quy mọi cải thiện của LoRA+head cho riêng LoRA.

Tham khảo phương pháp gốc: [LoRA](https://arxiv.org/abs/2106.09685), [Prototypical Networks](https://arxiv.org/abs/1703.05175).

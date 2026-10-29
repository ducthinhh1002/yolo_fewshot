# Danh sách lỗi và rủi ro hiện tại

| Mức | Vấn đề / bằng chứng | Hệ quả | Hướng xử lý |
|---|---|---|---|
| Cao | DE-ViT chuyển từ checkpoint ảnh tự nhiên sang BCCD, nhiều false positive ở smoke test | Chạy được không đồng nghĩa đủ chất lượng | Review RPN proposal recall, score và prototype; tuning chỉ trên validation, rồi mở rộng số query/trial |
| Cao | BCCD không có nhãn ung thư từng box | Không có ground truth để đánh giá malignant-cell detection toàn pipeline | Bổ sung smear có box và nhãn bệnh lý được xác nhận; tạm báo detection WBC và cell classification riêng |
| Cao | BCCD không có patient IDs | Không chứng minh split detection độc lập bệnh nhân | Ghi rõ image-level split; cần thêm nguồn có mapping bệnh nhân |
| Cao | AML khác nguồn, dòng bệnh và kiểu tế bào âm tính so với C-NMC | Không diễn giải external stress test là B-ALL source-only validation | Tách mục đánh giá, khai báo endpoint và shift; tìm nguồn B-ALL phù hợp khi có điều kiện |
| Vừa | LoRA / ProtoNet projection mới so sánh 30 trial trên validation, mức tăng nhỏ | Chưa đủ kết luận ưu thế ổn định | Review lỗi, chọn cấu hình, chạy 300 trial ghép cặp và báo biến thiên; không dùng test để tuning |
| Vừa | 1-shot không có hai cell khác nhau trong support để tạo episode query | Query huấn luyện ProtoNet là hai view cùng ảnh | Ghi rõ biến thể, không gọi novel-class meta-learning hoặc tăng số unique shots |
| Vừa | LoRA học thêm classifier head | Không quy toàn bộ gain cho adapter | Giữ head-only control trong bảng kết quả |
| Vừa | Confidence classification chưa được calibration; baseline có score gần 0.5 | Quyết định nhạy với support và float reduction | Khóa phép tính, kiểm thử tái lập; calibration/temperature chỉ chọn trên val |
| Vừa | DE-ViT dùng 80 auxiliary reference classes và pretrained RPN/head | Ngoài K support còn có external pretraining supervision | Khai báo nguồn/checksum; không dùng RPN đã học đủ BCCD trong comparison K-shot |
| Vừa | YOLO full supervision dùng 205 ảnh, DE-ViT support K ảnh | Không phải comparison cân bằng annotation budget | Để riêng; khi so K-shot phải dùng cùng K target images/boxes và cùng query |
| Vừa | DE-ViT upstream cũ, server Python/torch mới, thiếu xFormers | Có rủi ro sai khác số học / hiệu năng và build lại | Môi trường riêng, pinned commit, compat diff, strict parameter-load checks; không nhận là tái lập metric gốc |
| Vừa | Resize/proposal cap của DE-ViT giảm để giữ VRAM | Có thể giảm recall/AP | Công bố config; tuning val dưới cap 16 GiB; không đổi âm thầm khi so sánh |
| Vừa | dHash không phân biệt tốt tế bào khác nhau cùng nền đen | Tự xóa theo dHash sẽ làm mất nhiều dữ liệu | Chỉ đánh dấu candidate để review; pixel duplicate và patient overlap kiểm tra riêng |
| Thấp | Bộ raw/prototype/checkpoint có license và attribution riêng | Mã nguồn mở không đồng nghĩa có quyền phát hành lại mọi ảnh/trọng số | Giữ nguồn, citation và checksum; không đưa raw/models/runs vào Git |

Trạng thái này là danh sách kỹ thuật/nghiên cứu, không phải xác nhận sử dụng lâm sàng. Các ca classification sai không được tự coi là nhãn gốc sai nếu chưa có chuyên gia review.

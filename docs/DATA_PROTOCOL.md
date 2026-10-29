# Dữ liệu và giao thức đánh giá — phiên bản 1.0.0

Ngày chốt: 25/10/2026. Cấu hình thực thi: `configs/data_protocol.json`.
Danh mục nguồn và điều kiện sử dụng: `docs/DATASET_CATALOG.json`.
Số lượng thực tế sau audit: `data/DATA_SUMMARY.md` và `data/manifests/summary.json`.

## 1. Phạm vi và nhãn

Phân loại chính: tế bào tiền thân bình thường/hematogone (`normal=0`) so với B-ALL lymphoblast (`abnormal=1`) trên C-NMC. Không diễn giải là nhận biết mọi ung thư máu.

External stress test: typical lymphocyte (`LYT=0`) so với myeloblast (`MYO=1`) trên AML-Cytomorphology_LMU. Đây là kiểm tra **khác nguồn + khác dòng bệnh + thay đổi loại tế bào âm tính**. Không gọi đó là external validation B-ALL tương đương, cũng không quy toàn bộ chênh lệch cho microscope/stain. `MYO` là myeloblast; `MYB` là myelocyte và bị loại. Mapping đã đối chiếu với [bảng dữ liệu của phòng thí nghiệm tác giả](https://github.com/marrlab/DinoBloom).

Normal là nhãn hình thái ở mức tế bào, không phải kết luận bệnh nhân khỏe mạnh; abnormal là mục tiêu sàng lọc theo lớp dữ liệu, không phải chẩn đoán.

Detection tách riêng: YOLO/DE-ViT tìm **WBC** trên BCCD, class 0. Không gán nhãn ung thư cho WBC. Các bộ đã chọn chưa có đầy đủ smear + box + nhãn bệnh lý từng tế bào để đo malignant-cell detection mAP toàn pipeline. Demo ghép detector/classifier được, nhưng để đánh giá end-to-end cần thêm annotation chuyên gia. Quét toàn lam là phần mở rộng.

## 2. Nguồn và lựa chọn

| Nguồn | Vai trò | Chọn trước audit |
|---|---|---|
| C-NMC 2019 | Classification chính, có mã bệnh nhân | 10.661 ảnh training release, ba fold chính thức |
| AML-Cytomorphology_LMU | External single-cell stress test | 7.205 ảnh LYT/MYO từ mirror JPEG v1 |
| BCCD | WBC detection | 364 ảnh và annotation VOC từ commit cố định |

Nguồn chính thức: [C-NMC/TCIA](https://doi.org/10.7937/tcia.2019.dc64i46r), [AML/TCIA](https://doi.org/10.7937/tcia.2019.36f5o9ld), [BCCD](https://github.com/Shenggan/BCCD_Dataset).
Mirror, phiên bản, citation và license ghi trong catalog. Mã tải lưu metadata, URL và SHA-256 archive trong `data/provenance`.
TCIA công bố C-NMC và AML theo CC BY 3.0; các mirror có thông tin license khác/thiếu, nên giữ nguyên ghi chú khác biệt và attribution nguồn chính. Mirror AML là JPEG chuyển đổi: không tuyên bố pixel giống TIFF gốc hay đây là benchmark gốc của tác giả.
BCCD ghi MIT trong repository, giữ LICENSE trong raw. Không đưa ảnh vào mã nguồn Git; người tái lập tải từ nguồn đã ghi.

Đã tải và kiểm tra Taleqani ALL: ảnh Original và Segmented chứa nhiều tế bào. Loại khỏi đánh giá single-cell vì không được lấy nhãn cả ảnh gán cho từng tế bào. Raw/provenance giữ lại để ghi nhận việc kiểm tra, nhưng không xuất hiện trong manifest/episode huấn luyện hay test chính.
ALL-IDB cần đơn ký và email xin truy cập nên không là phụ thuộc triển khai; không lấy bản sao không xác minh quyền.

## 3. Chia tập cố định

- C-NMC `fold_0` → train/support pool.
- C-NMC `fold_1` → validation: chọn hyperparameter, temperature, số bước, checkpoint và preprocessing.
- C-NMC `fold_2` → internal held-out test/query.
- AML LYT/MYO → external_test, chỉ inference; không lấy target support hay fine-tune.
- BCCD → train/val/test upstream, sau đó quarantine ảnh trùng theo quy tắc dưới.

Đây là benchmark nội bộ dùng training release của C-NMC, không phải điểm challenge leaderboard. Không dùng preliminary/final challenge test, kể cả làm unlabeled SSL.
Patient ID C-NMC đọc từ `UID_<subject>_<image>_<cell>_<class>.bmp`, namespace theo class để tránh nhầm healthy/ALL. Script dừng nếu một patient xuất hiện ở nhiều split; không random split theo cell.
AML filenames là mã lớp và số thứ tự cell, không có patient mapping đáng tin cậy: để trống `patient_id`, giữ toàn bộ ở external_test, không tự dựng mã bệnh nhân.
BCCD không có patient IDs: kết quả detection chỉ là image-level split, không được gọi là patient-independent.

## 4. Audit và chuẩn hóa annotation

1. Xác nhận đủ bản phát hành đã chọn; mở/decode mọi ảnh; kiểm tra folder label khớp filename.
2. Lưu kích thước, SHA-256 file, hash pixel RGB, dHash 64 bit; raw không bị sửa.
3. Ảnh giống pixel: giữ một bản. Nếu nhãn xung đột, loại toàn bộ nhóm.
4. Ảnh có dHash Hamming ≤ 2 giữa split/nguồn: ghi candidate để review, **không tự động loại**. Nền đen của C-NMC làm nhiều tế bào khác nhau có dHash gần giống nhau.
5. Ưu tiên giữ external_test > test > val > train. Không chuyển bệnh nhân qua split.
6. Lưu trạng thái từng ảnh và mọi cặp bị loại vào `duplicate_audit.csv`.

dHash chỉ là chỉ dấu để review, không chứng minh ảnh trùng; không tự loại ảnh chỉ vì dHash. Nó cũng không phát hiện mọi crop/rotation trùng. Không tuyên bố đã loại mọi near-duplicate. Quy tắc này khóa trước huấn luyện, không dùng kết quả mô hình để chọn ảnh loại.

Box BCCD VOC được hiểu là 1-based inclusive, chuyển sang 0-based half-open rồi YOLO normalized xywh; kiểm tra trong biên ảnh. Giữ WBC thành class 0; RBC/Platelets là background của nhiệm vụ một lớp, không phải normal/abnormal.

## 5. Few-shot classification

2-way, K-shot với K ∈ {1,5,10,20}: K **ảnh mỗi lớp**, tổng 2K ảnh support có nhãn.
300 trial với seed `20261025 + trial`, trial từ 0 đến 299.
Lấy mẫu không hoàn lại: xáo trộn bệnh nhân từng lớp, lấy một ảnh mỗi người theo vòng rồi mới lấy ảnh thứ hai của cùng người. K là số ảnh, không phải số bệnh nhân; ghi thêm số support patient nếu phân tích ngân sách bệnh nhân.

Mỗi trial lấy tối đa 20 ảnh mỗi lớp, các K nhỏ dùng tiền tố để so sánh ghép cặp. Mọi phương pháp dùng cùng support IDs trong `data/episodes/classification.jsonl`.
Query là **toàn bộ ảnh giữ lại** trong manifest test tương ứng, cố định qua trial/K/phương pháp; không cân bằng lại test. Bệnh nhân support/query tách nhau trong C-NMC.

Các lớp đã xuất hiện ở train/val/test. Đây là few-label adaptation với lớp đã biết, không tuyên bố novel-class meta-learning.
Proto dùng đặc trưng pretrained và prototype của đúng 2K ảnh support. LoRA reset cùng pretrained backbone cho từng trial rồi thích nghi bằng đúng 2K ảnh đó.
Không dùng toàn bộ nhãn train để pretrain classifier rồi gọi là cùng ngân sách K-shot; full-supervision baseline phải ghi riêng.
Validation có nhãn đầy đủ là ngân sách phát triển chung: không tuyên bố toàn dự án chỉ có 2K nhãn. Khóa hyperparameter trên val, không chọn riêng theo kết quả test/external hay trial test.
Pretrained weights phải ghi nguồn và dữ liệu đã biết; không dùng weights đã fine-tune trên tập benchmark. Nếu không xác minh được pretraining overlap, ghi đó là giới hạn.

## 6. Few-shot detection

`data/episodes/detection.jsonl` có 300 × 4 cấu hình ghép cặp.
Chỉ lấy ảnh BCCD train giữ lại có đúng một box WBC: K ảnh = K annotated WBC instances. DE-ViT lấy crop tham chiếu từ các box đó; YOLO few-shot chỉ dùng cùng K ảnh/box.
Query là toàn bộ BCCD test giữ lại; validation chung là BCCD val. K nhỏ là tiền tố K lớn.
YOLO dùng toàn bộ BCCD train là baseline **full detector supervision**, báo riêng với ngân sách annotation đầy đủ. Không so nó với DE-ViT K-shot như hai mô hình có cùng ngân sách.

## 7. Preprocessing, SSL và stress test

Classification: RGB, resize giữ tỷ lệ và pad về 224×224 bằng bicubic; normalize theo backbone pretrained. Mọi nhánh dùng cùng preprocessing. Kích thước detector khóa chung khi triển khai model.
Augmentation chỉ trên support/train, không làm tăng số unique shots; không augmentation test mặc định.
Stain reference phải fit trên C-NMC train; hyperparameter, temperature/calibration khóa trên validation. Không fit trên test/external, kể cả cách fit không nhãn.
Nếu tiếp tục self-supervised adaptation DINOv2: chỉ dùng C-NMC train không nhãn, khai báo số ảnh unlabeled/chi phí, áp dụng nhất quán cho các nhánh so sánh. Không SSL trên test hay AML trong giao thức inductive này.

External: C-NMC support → AML query, không target labels/support, target adaptation hay target calibration. Báo đây là source-and-disease shift, không source-only shift.
Bốn ablation: preprocessing cơ sở; +color augmentation; +stain normalization; +cả hai. Chọn kỹ thuật/siêu tham số trên val trước test. C-NMC đã được xử lý stain ở upstream nên cần nêu khi diễn giải.

## 8. Chỉ số và thống kê

Classification positive class=1; ngưỡng chính cố định 0.5. Primary macro-F1; báo sensitivity, specificity, positive F1, balanced accuracy, AUROC, AUPRC, confusion matrix và prevalence.
Proto phải chuyển score thành xác suất hai lớp bằng softmax với temperature khóa trên val; không nạp raw cosine vào cột `prob_abnormal`. Confidence không mặc nhiên đã được calibration.
Mean và sample SD qua 300 support trial. Query dùng chung nên không lấy SD/√300 làm confidence interval khái quát hóa.
So sánh phương pháp bằng chênh lệch trên cùng trial/K/query, không coi các cell là bệnh nhân độc lập.
Evaluator xuất thêm CI percentile 95% macro-F1 bằng 2.000 bootstrap bệnh nhân phân tầng class trên C-NMC **trial 0 đã chọn trước**. Đây là CI điều kiện theo support trial 0, không bao gồm mọi bất định thích nghi.
AML không có patient mapping nên không xuất patient-level CI; SD qua trial chỉ thể hiện độ nhạy với support.

Detection: mAP@0.5, mAP@[0.5:0.95], precision/recall tại operating point khóa trên val, số ảnh và số box. Dùng cùng convention/evaluator cho YOLO và DE-ViT. Không trộn mAP và classification F1 thành một điểm, không báo malignant detection mAP từ nhãn WBC.

## 9. Chạy và tái lập

```powershell
python -m pip install -r requirements-data.txt
python scripts/download_data.py
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/evaluate_predictions.py results/method_predictions.csv --output results/method_evaluation
```

Python trên máy này: `C:\Users\PC\AppData\Local\Programs\Python\Python311\python.exe`.
CSV dự đoán: `dataset,trial,shots_per_class,sample_id,prob_abnormal`; dataset là `cnmc` hoặc `aml`. Một file cho một phương pháp/cấu hình, đủ hai nguồn × 300 trial × 4 K. Thiếu/lặp query, ID sai, NaN hoặc xác suất ngoài [0,1] bị từ chối.
Không có kết quả mô hình thật được tạo ở bước chuẩn bị dữ liệu này.

Đầu ra:

- `data/manifests/classification.csv`: nhãn, patient/group, split, hash, trạng thái.
- `data/manifests/bccd.csv`, `bccd_boxes.csv`, `duplicate_audit.csv`, `summary.json`.
- `data/splits/*.csv`: chỉ các ảnh giữ lại, tập chia cố định.
- `data/episodes/*.jsonl`: support classification/detection.
- `data/processed/bccd/dataset.yaml`, `images/`, `labels/`: YOLO dataset một lớp.
- `data/provenance/manifests.lock.json`: SHA-256 của manifest/split/episode/config/catalog.
- `data/provenance/preparation_environment.json`: Python/package versions thực tế.

Không commit `data/raw`, `data/downloads`, `data/processed`. Script chạy lại kiểm tra lock; đầu ra thay đổi thì báo lỗi và giữ lock cũ. Muốn thay dataset, nhãn, seed, split, audit hay query: tăng protocol version, xem xét thay đổi rồi mới dùng `--update-lock` và chạy lại các thí nghiệm liên quan.

`--reuse-verified-index` chỉ dành cho xử lý lại metadata khi raw đã được decode/hash kiểm tra và không thay đổi; lần tái lập độc lập phải chạy lệnh mặc định để kiểm tra lại ảnh.

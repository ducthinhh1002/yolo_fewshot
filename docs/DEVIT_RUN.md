# DE-ViT chạy thử trên BCCD

## Mã nguồn và trọng số

- [Mã chính thức](https://github.com/mlzxy/devit), commit `9cab390fbe78b3bc8d62c224411dce1c71056cf9`, MIT; [bài báo / trang dự án](https://mlzxy.github.io/devit/).
- Dùng **OpenSetDetectorWithExamples thực của tác giả**, backbone ViT-Small và COCO pretrained RPN/head. Không thay detector bằng YOLO/Faster R-CNN khác và không chạy mô hình khởi tạo ngẫu nhiên.
- Checkpoint `vits_0089999.pth` cùng background và COCO reference prototypes lấy từ [Box tác giả](https://rutgers.box.com/s/2lco6ab66pn3ufq6rh4gmyfzg9vfkm23). Tải từng file khoảng 190 MB tổng, không tải bundle 34 GB. SHA-256 và Box file ID ở `models/devit/provenance.json` trên server.
- Khi load, chỉ thay hai buffer `train_class_weight`/`test_class_weight` để thêm WBC. Nếu thiếu hoặc dư bất kỳ tham số detector học được nào thì runner dừng, không chấp nhận mô hình load một phần.

## Môi trường riêng trên server

Thư mục `/home/thinhnd39/yolo_fewshot`. GPU H200; `VLLM:Engine`; allocator 16 GiB. Không chạy trên local.

```bash
bash scripts/setup_devit.sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv-devit/bin/python scripts/run_devit_smoke.py
.venv-devit/bin/python scripts/evaluate_devit_smoke.py
.venv/bin/python -m unittest discover -s tests -v
```

Upstream ban đầu yêu cầu Python 3.9 / torch 1.13.1 / torchvision 0.14.1 / NumPy 1.22 / xFormers cũ. Trên server dùng Python 3.13, torch 2.14 CUDA 13 và torchvision 0.29 có sẵn; tạo `.venv-devit` riêng, không hạ phiên bản môi trường đang dùng cho YOLO/LoRA.

Đã biên dịch native extension Detectron2 dạng CPU cho các hàm phụ trợ; **mạng, ROIAlign và NMS thường vẫn chạy CUDA qua torch/torchvision**. Không kiểm thử kernel CUDA rotated/deformable, không dùng các tác vụ đó trong lần chạy này. Không cài xFormers; attention dùng fallback PyTorch thực của upstream, không phải output giả.

`patch_devit_compat.py` sửa API NumPy/Pillow đã bị bỏ (`np.int`, `np.float`, `numpy.lib.pad`, `Image.LINEAR`), giữ nguyên kiến trúc và trọng số. Bản diff lưu trong `runs/devit_smoke/compat.patch`. Checkpoint pickle chính thức được load với `weights_only=False`; chỉ áp dụng cho các file đã xác định nguồn, không dùng tùy chọn đó để mở checkpoint lạ.

## Support/query và giới hạn

`configs/detection_protocol.json` là giao thức detection chung; `configs/devit_smoke.json` là cấu hình chạy thử:

- Dùng đúng episode detection trial 0 đã tạo, K ∈ {1,5,10,20}. K ảnh train tương ứng đúng K box WBC; không hoàn lại, support K nhỏ nằm trong K lớn.
- Prototype WBC là trung bình foreground patch features trong box support, sau đó normalize. Dùng DINOv2 ViT-Small đã có trong checkpoint; preprocessing prototype theo ImageNet mean/std của tool tác giả, bbox soft mask và lưới patch 14.
- Giữ 80 COCO reference class prototypes làm lớp tham chiếu phụ trợ cho nhánh one-vs-rest; chỉ lấy output WBC. Không tính các lớp này như ảnh WBC support; nguồn pretraining và reference supervision phải được khai báo khi so sánh.
- Query thử là 8 hàng đầu tiên trong manifest BCCD validation, giống nhau ở mọi K. Không dùng test. Query test trong canonical episode được giữ cho vòng đánh giá sau khi khóa cấu hình, không bị sửa.
- Không fine-tune DE-ViT trên BCCD. RPN của tác giả không lấy từ YOLO đã học toàn bộ BCCD, tránh làm lộ annotation target ngoài K.
- Để phù hợp tài nguyên, query short side 480 / max 640, batch 1, tối đa 100 RPN proposals, top 50 detections, score floor 0.01. Khác cấu hình benchmark tác giả, nên không so metric như tái lập benchmark chính thức.
- Nhánh YOLO K-shot tương lai phải học đúng cùng K ảnh/box và đánh giá cùng query; YOLO đã học đủ 205 ảnh là baseline full supervision riêng. Lần này chưa chạy comparison YOLO K-shot nhiều trial.

## Đánh giá và bằng chứng

`runs/devit_smoke/result.json` lưu support/query IDs, prototypes, boxes/scores, checksum checkpoint, commit và bộ nhớ GPU. `evaluation.json` dùng pycocotools COCO bbox AP trên cùng 8 ảnh, kèm TP/FP/FN tại confidence 0.25, IoU 0.5. Đây là smoke test một trial, không phải kết quả cuối 300 trial hay kiểm chứng phát hiện ung thư.

DE-ViT có thể forward thành công nhưng domain shift từ ảnh tự nhiên sang ảnh tế bào vẫn lớn. False positive nhiều; không chọn nó thay YOLO chỉ từ việc chạy thành công. Xem `docs/MILESTONE_14_20_STATUS.md` và `docs/RISKS.md`.

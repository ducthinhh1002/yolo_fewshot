import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'runs/devit_smoke'
    result=json.loads((out/'result.json').read_text())
    evaluation=json.loads((out/'evaluation.json').read_text())
    if result['status']!='passed' or result['missing_learned_parameters']:raise ValueError('DE-ViT incomplete')
    data={'host':socket.gethostname(),'devit':result,'evaluation':evaluation,'detection_protocol':json.loads((ROOT/'configs/detection_protocol.json').read_text()),'weight_provenance':json.loads((ROOT/'models/devit/provenance.json').read_text())}
    destination=ROOT/'output/reports';destination.mkdir(parents=True,exist_ok=True)
    (destination/'milestone_14_20.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Trạng thái mốc 14–20/11','','**Đã hoàn thành mức triển khai/chạy thử theo mốc này.** Không đồng nghĩa hoàn thành thí nghiệm cuối của đề tài.','','| Đầu việc | Bằng chứng |','|---|---|','| Prototypical Networks chạy được | Learned projection episodic trên DINOv2 đóng băng, 30 trial ghép cặp mỗi mức shot; xem ADAPTATION_STATUS.md |','| LoRA chạy thử | Q/V block cuối, rank 4, head-only control, cùng 30 trial/shot |','| DE-ViT chạy thử | Mã và checkpoint tác giả, đủ tham số, 1/5/10/20-shot × 8 validation images |','| Chốt support/query | Classification dùng cùng support/validation; detection có configs/detection_protocol.json, tách full supervision khỏi K-shot |','| Danh sách lỗi/rủi ro | docs/RISKS.md và các báo cáo lỗi classification/detection |','','## DE-ViT smoke test','','Actual `OpenSetDetectorWithExamples`, pretrained ViT-Small + RPN/head của tác giả; pinned commit `'+result['upstream_commit']+'`. Không thiếu tham số detector học được. Không thay bằng detector khác hoặc weights ngẫu nhiên.','','Query 8 ảnh BCCD validation, 8 box WBC; một support trial. AP tính bằng pycocotools, không gọi là benchmark cuối.','','| Shot | AP@50 | AP@50–95 | TP | FP | FN |','|---|---:|---:|---:|---:|---:|']
    for r in evaluation['results']:
        lines.append(f"| {r['shots_per_class']} | {100*r['ap50']:.2f}% | {100*r['ap_50_95']:.2f}% | {r['tp_conf025_iou05']} | {r['fp_conf025_iou05']} | {r['fn_conf025_iou05']} |")
    lines+=['','TP/FP/FN ở confidence 0.25, IoU 0.5. Nhiều false positive; việc load và inference thành công chỉ xác nhận khả thi kỹ thuật. Chưa đủ để chọn DE-ViT làm detector cuối.', '',f"Peak reserved PyTorch: {result['memory']['peak_reserved_gib']:.3f} GiB; allocator cap 16 GiB, tiến trình `VLLM:Engine`, GPU H200.",'','## Còn lại ở các mốc sau','','- Mở rộng query/trial cho detection và huấn luyện YOLO K-shot đúng cùng ngân sách trước khi so sánh chất lượng.','- Tuning trên validation, thử augmentation/stain, tiếp tục đánh giá 300 trial cho các mô hình classification.','- Khóa cấu hình rồi mới dùng test/cross-source; hiện chưa dùng test.','- Hoàn thiện explainability/demo/báo cáo. Dữ liệu hiện chưa có ground truth ung thư từng box để xác thực toàn pipeline.', '', 'Hướng dẫn: `docs/DEVIT_RUN.md`. Protocol: `configs/detection_protocol.json`. Rủi ro: `docs/RISKS.md`.','']
    (ROOT/'docs/MILESTONE_14_20_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Milestone report complete')


if __name__=='__main__':main()

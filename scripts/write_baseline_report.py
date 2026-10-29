"""Publish validation-only baseline evidence after all stages complete."""
import csv
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'configs/baseline.json').read_text())
    out = ROOT/config['output']
    result = {'host':socket.gethostname(), 'config':config}
    for name in ['audit','features','prototype','detector','pipeline_demo']:
        result[name] = json.loads((out/(name+'.json')).read_text())
    with (out/'yolo/train/results.csv').open() as stream:
        epochs = list(csv.DictReader(stream))
    result['completed_epochs'] = len(epochs)
    results_dir = ROOT/'output/reports'
    results_dir.mkdir(parents=True,exist_ok=True)
    (results_dir/'baseline_server.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines = ['# Kết quả baseline trên server', '', f"Host `{result['host']}`, process `VLLM:Engine`; allocator giới hạn 16 GiB. Chỉ train/validation, chưa dùng test.", '', '## Phân loại: frozen DINOv2 + nearest-prototype', '', 'Trích đặc trưng toàn bộ 3.455 ảnh train và 3.554 ảnh validation; 384 chiều, CLS L2-normalized. Prototype chỉ sử dụng K ảnh mỗi lớp trong support train; trích toàn bộ train không có nghĩa dùng toàn bộ nhãn cho mỗi trial. Cùng 300 trial ghép cặp theo giao thức.', '', '| Shot/lớp | Macro-F1 mean ± SD | Sensitivity mean | Specificity mean | AUROC mean |', '|---|---:|---:|---:|---:|']
    for k,value in result['prototype']['summary'].items():
        lines.append(f"| {k} | {value['macro_f1']['mean']:.4f} ± {value['macro_f1']['sd']:.4f} | {value['sensitivity']['mean']:.4f} | {value['specificity']['mean']:.4f} | {value['auroc']['mean']:.4f} |")
    lines += ['', 'Đây là baseline không fine-tune, chưa phải Prototypical Networks có episodic training; SD thể hiện biến thiên support, không phải CI bệnh nhân.', '', '## Định vị WBC: YOLO11n', '', f"Dùng đủ 205 ảnh train và 87 validation; đã chạy {len(epochs)} epoch, tối đa 50, patience 12; input 640, batch 8. Đánh giá lại checkpoint best sau khi chọn bằng validation.", '']
    for key,value in result['detector']['validation'].items():
        lines.append(f'- `{key}`: {value:.4f}')
    lines += ['', 'Baseline này sử dụng annotation detection đầy đủ, không so như cùng ngân sách với detector K-shot.', '', '## Kiểm tra tích hợp', '', 'Đã chạy ảnh BCCD validation → boxes WBC → crop → DINOv2 → prototype 20-shot trên 8 ảnh; đầu ra gồm tọa độ, detector confidence, label và classification score chưa calibration.', '', 'BCCD không có nhãn ung thư từng tế bào và khác miền dữ liệu classification. Đây chỉ là kiểm tra tích hợp; chưa đủ để kết luận chất lượng phát hiện ung thư toàn pipeline.', '', '## Việc tiếp theo', '', '- Phân tích lỗi detection/crop và độ nhạy với support; kiểm tra preprocessing trên validation.', '- Triển khai LoRA và Prototypical Networks, kiểm tra DE-ViT.', '- Thực hiện ablation màu/stain và khóa siêu tham số trước đánh giá test/cross-source.', '', 'Chi tiết: `output/reports/baseline_server.json`. Cách chạy: `docs/BASELINE_RUN.md`. Trọng số, feature và dữ liệu giữ trên server trong `runs/baseline_v1/`, không đưa vào Git.', '']
    (ROOT/'docs/BASELINE_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Baseline report complete')


if __name__ == '__main__': main()

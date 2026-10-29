"""Summarize completed smoke runs without copying datasets or model weights."""
import json
from pathlib import Path
import platform
import socket

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'configs/smoke.json').read_text())
    out = ROOT/config['output']
    report = {'host': socket.gethostname(), 'python': platform.python_version(), 'root': str(ROOT), 'config': config}
    for name in ['audit', 'dinov2', 'yolo', 'dinov2_memory', 'yolo_memory', 'environment', 'runtime']:
        report[name] = json.loads((out/(name+'.json')).read_text())
    for name in ['audit', 'dinov2', 'yolo']:
        if report[name]['status'] != 'passed':
            raise ValueError(f'{name} failed')
    for name in ['dinov2_memory', 'yolo_memory']:
        if report[name].get('peak_reserved_gib', 0)*1024**3 >= 20_000_000_000:
            raise ValueError('Memory limit exceeded')
    destination = ROOT/'output/reports'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'server_pipeline.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Kết quả kiểm tra pipeline trên server', '', f"Host: `{report['host']}`. Thư mục: `{ROOT}`.", '', '## Dữ liệu', '', '| Tập | Số ảnh |', '|---|---:|']
    for name, value in report['audit']['datasets'].items():
        lines.append(f"| {name} | {value['images']} |")
    lines += ['', 'Đã kiểm tra lại ảnh, hash và nhãn; tách bệnh nhân C-NMC giữa train/val/test; không có pixel trùng giữa các tập giữ lại. Đủ số mẫu cho 20-shot. Ảnh nghi gần trùng chỉ được đánh dấu để review, không tự loại theo dHash.', '', '## Chạy thử', '', '- DINOv2-small: 32 ảnh train và 16 validation, vector CLS 384 chiều, L2-normalized, không có NaN/Inf.', '- YOLO11n: 16 ảnh train, 8 validation, 1 epoch, input 320, batch 4; đã train, validate và lưu dự đoán trên 8 ảnh.', '- Không dùng test hoặc AML để chọn mô hình trong smoke test.', '', '| Nhánh | GPU | Peak allocated (GiB) | Peak reserved (GiB) |', '|---|---|---:|---:|']
    for name in ['dinov2_memory', 'yolo_memory']:
        value = report[name]
        lines.append(f"| {name} | {value.get('device')} | {value.get('peak_allocated_gib',0):.3f} | {value.get('peak_reserved_gib',0):.3f} |")
    lines += ['', 'Allocator PyTorch bị giới hạn 16 GiB, chừa khoảng cho CUDA dưới yêu cầu 20 GB; hai nhánh chạy tuần tự. Các số peak trên là bộ nhớ PyTorch, không phải tổng bộ nhớ của mọi người dùng trên GPU chung.', '', '**Đây là kiểm tra khả năng chạy, không phải benchmark cuối.** Detection mới chỉ học 1 epoch nên dự đoán còn nhiều false positive; không dùng để đánh giá chất lượng cuối hoặc làm demo chẩn đoán.', '', 'Dữ liệu đủ cho classification và định vị WBC riêng biệt. Chưa có box + nhãn ung thư từng tế bào trên ảnh smear để đánh giá malignant detection toàn pipeline; AML khác cả nguồn và dòng bệnh.', '', 'Chi tiết model revision, checksum, metric validation và phiên bản phần mềm trong `output/reports/server_pipeline.json`. Chạy lại theo `docs/PIPELINE_RUN.md`.', '']
    (ROOT/'docs/PIPELINE_STATUS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Server report written')


if __name__ == '__main__':
    main()

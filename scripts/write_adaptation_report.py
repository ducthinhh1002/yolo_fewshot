import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]


def main():
    c=json.loads((ROOT/'configs/adaptation.json').read_text())
    out=ROOT/c['output']
    comparison=json.loads((out/'comparison.json').read_text())
    classification=json.loads((out/'classification_error_analysis.json').read_text())
    detection=json.loads((out/'detection_error_analysis.json').read_text())
    labels={'baseline':'Baseline','head_only':'Chỉ học head','protonet':'ProtoNet projection','lora':'LoRA + head'}
    report={'host':socket.gethostname(),'comparison':comparison,'classification_errors':classification,'detection_errors':detection}
    destination=ROOT/'output/reports'; destination.mkdir(parents=True,exist_ok=True)
    (destination/'adaptation_server.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Phân tích lỗi và so sánh LoRA / Prototypical Networks','','Đánh giá phát triển trên validation C-NMC, 30 trial ghép cặp đầu tiên, cùng K-shot support và 3.554 query; không dùng test/AML. Các con số dưới đây không thay thế bảng baseline 300 trial trước đó.','','## Macro-F1 trung bình ± SD','','| Phương pháp | 1-shot | 5-shot | 10-shot | 20-shot |','|---|---:|---:|---:|---:|']
    for method,label in labels.items():
        cells=[]
        for k in ['1','5','10','20']:
            v=comparison['summary'][method][k]['macro_f1']; cells.append(f"{v['mean']*100:.2f}% ± {v['sd']*100:.2f}")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','SD là biến thiên theo support; không phải confidence interval theo bệnh nhân. Không kết luận ưu thế thống kê chỉ từ bảng mean.','','## Diễn giải','','- Cải thiện ở 5/10/20-shot còn nhỏ; các cách thích nghi có thể kém baseline ở 1-shot. Không khẳng định LoRA hoặc ProtoNet tốt hơn ổn định.','- Head-only là đối chứng quan trọng: LoRA học cả adapter và classifier head; không quy toàn bộ thay đổi so với baseline cho adapter.','- LoRA ở block attention cuối, rank 4 trên Q/V, 6.144 adapter parameters + 768 head parameters. Mọi trọng số còn lại đóng băng; reset theo trial/K.','- ProtoNet ở đây học residual projection episodically trên DINOv2 đóng băng. 1-shot dùng hai view cùng ảnh, không có query cell độc lập để học nội bộ. Không phải thiết lập meta-learning trên lớp mới.','- Tính lại baseline bằng đúng phép reduction NumPy của vòng trước để tránh sai khác float ở ngưỡng 0.5; kiểm thử đối chiếu từng trial.','','## Lỗi classification ở 20-shot','','| Phương pháp | TN trung bình | FP trung bình | FN trung bình | TP trung bình | Tế bào sai ≥80% trial |','|---|---:|---:|---:|---:|---:|']
    for method,label in labels.items():
        v=classification[method]; cm=v['average_confusion_tn_fp_fn_tp']
        lines.append('| '+label+' | '+' | '.join(f'{n:.1f}' for n in cm)+f" | {v['cells_error_frequency_at_least_80pct']} |")
    lines+=['','Một số tế bào sai lặp lại qua nhiều support, cần review thay vì chỉ chọn một trial có điểm cao. Nhóm bệnh nhân baseline có tỷ lệ sai cao nhất:','']
    for p in classification['baseline']['patient_error_rates'][:3]:
        lines.append(f"- `{p['patient_id']}`: {p['images']} ảnh; tần suất sai trung bình {p['mean_cell_error_frequency']*100:.1f}%.")
    lines+=['','Các thống kê độ sáng/foreground theo từng nhãn được lưu trong JSON; chỉ là liên hệ mô tả, không chứng minh nguyên nhân sinh học hoặc lỗi nhãn. Những ca sai cần chuyên gia xác nhận.','','## Lỗi định vị YOLO trên 87 validation','','Confidence cố định 0.25, ghép một-một theo confidence; không phải operating point best-F1 hay mAP.','','| IoU | TP | FP | FN | Precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for threshold,v in detection['totals'].items():
        lines.append(f"| {threshold} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | {v['recall']:.3f} |")
    lines+=['','Chênh lệch giữa IoU 0.5 và 0.75 giúp nhận ra box chưa khớp chặt dù đã tìm đúng vùng tế bào. FP/FN và ảnh minh họa ở `runs/adaptation_v1/`.','','## Phạm vi và bước tiếp theo','','Đây là một cấu hình cố định cho vòng validation 30 trial, chưa tuning search và chưa chạy 300 trial cho LoRA/ProtoNet. Không chọn mô hình cuối dựa trên cải thiện nhỏ này. Tiếp theo: review ca lỗi, thử learning rate/số bước và augmentation trên validation, kiểm tra độ ổn định theo support/bệnh nhân, rồi chốt cấu hình trước mở rộng thực nghiệm. Không dùng test để chỉnh tham số.','','BCCD chỉ có nhãn WBC, không dùng để suy kết quả ung thư toàn pipeline. AML khác cả nguồn và dòng bệnh.','',f"Peak reserved PyTorch của lượt adaptation: {comparison['memory']['peak_reserved_gib']:.3f} GiB; allocator cap 16 GiB, tên tiến trình `VLLM:Engine`.",'','Chi tiết cách chạy và giới hạn: `docs/ADAPTATION_RUN.md`; số liệu máy đọc: `output/reports/adaptation_server.json`.','']
    (ROOT/'docs/ADAPTATION_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    x=[1,5,10,20]
    for method,label in labels.items():
        v=[comparison['summary'][method][str(k)]['macro_f1'] for k in x]
        ax.errorbar(x,[r['mean']*100 for r in v],yerr=[r['sd']*100 for r in v],marker='o',capsize=3,label=label)
    ax.set_xlabel('Số ảnh có nhãn mỗi lớp (shot)'); ax.set_ylabel('Macro-F1 validation (%)')
    ax.set_title('So sánh 30 trial ghép cặp — mean ± SD')
    ax.set_xticks(x); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out/'comparison_validation.png',dpi=160); plt.close(fig)
    print('Adaptation report complete')


if __name__=='__main__': main()

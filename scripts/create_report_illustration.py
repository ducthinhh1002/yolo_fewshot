"""Export an annotated data figure; all annotations are dataset ground truth."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    cnmc = read('data/splits/cnmc_train.csv')
    aml = read('data/splits/aml_external_test.csv')
    bccd = read('data/splits/bccd_train.csv')[0]
    boxes = [b for b in read('data/manifests/bccd_boxes.csv') if b['sample_id'] == bccd['sample_id']]
    choices = [next(r for r in cnmc if r['label'] == '0'), next(r for r in cnmc if r['label'] == '1'), next(r for r in aml if r['label'] == '0'), next(r for r in aml if r['label'] == '1'), bccd, bccd]
    titles = ['C-NMC | Tế bào tiền thân lành tính (HEM)', 'C-NMC | Tế bào blast B-ALL (ALL)', 'AML | Tế bào lympho điển hình (LYT)', 'AML | Nguyên tủy bào (MYO)', 'BCCD | Ảnh phết máu gốc', 'BCCD | Khung định vị bạch cầu theo nhãn gốc']
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), facecolor='white')
    for i, (ax, row, title) in enumerate(zip(axes.flat, choices, titles)):
        with Image.open(ROOT / row['path']) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(title, fontsize=12, loc='left', pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 5:
            for b in boxes:
                x, y, x2, y2 = [int(b[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor='#00ff96', linewidth=2.5))
                ax.text(x+3, max(15, y-6), 'Bạch cầu', fontsize=11, weight='bold', color='black', bbox={'facecolor': '#00ff96', 'edgecolor': 'none', 'pad': 3})
    fig.suptitle('Ví dụ minh họa các bộ dữ liệu đã chuẩn bị', x=.06, ha='left', fontsize=19, weight='bold', y=.985)
    fig.subplots_adjust(left=.06, right=.96, top=.94, bottom=.025, hspace=.24, wspace=.12)
    out = ROOT / 'output/figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'prepared_data_examples.png', dpi=180, facecolor='white')
    plt.close(fig)
    print(out / 'prepared_data_examples.png')


if __name__ == '__main__':
    main()

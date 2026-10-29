"""Create a small contact sheet for visual checks; does not change source data."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = sorted((ROOT / 'data/raw/taleqani/Original').iterdir())
    canvas = Image.new('RGB', (800, 220*len(folders)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, folder in enumerate(folders):
        for j, path in enumerate(sorted(folder.glob('*.jpg'))[:4]):
            with Image.open(path) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (190, 185))
            canvas.paste(thumb, (j*200, i*220+25))
            draw.text((j*200+5, i*220+5), folder.name, fill='black')
    output = ROOT / 'tmp/data_qa/taleqani_preview.jpg'
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == '__main__':
    main()

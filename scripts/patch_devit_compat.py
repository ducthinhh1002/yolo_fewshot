"""Minimal API compatibility patches for the pinned official DE-ViT checkout."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
repo=ROOT/'vendor/devit'
for path in [*repo.glob('detectron2/**/*.py'),*repo.glob('lib/**/*.py')]:
    text=path.read_text()
    updated=text.replace('from numpy.lib import pad','from numpy import pad')
    updated=updated.replace('Image.LINEAR','Image.BILINEAR').replace('Image.ANTIALIAS','Image.Resampling.LANCZOS')
    updated=re.sub(r'\bnp\.int\b','int',updated)
    updated=re.sub(r'\bnp\.float\b','float',updated)
    if updated!=text:
        path.write_text(updated)
        print(path.relative_to(repo))
out=ROOT/'runs/devit_smoke'; out.mkdir(parents=True,exist_ok=True)
(out/'compat.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=repo))

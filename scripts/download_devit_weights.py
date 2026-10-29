"""Download only required official author-hosted DE-ViT files, not the 34 GB bundle."""
from pathlib import Path
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parents[1]
FILES={'vits_0089999.pth':1665925730315,'background_prototypes.vits14.pth':1665927155639,'fs_coco14_base_train.vits14.pkl':1665918834162,'fs_coco_trainval_novel_10shot.vits14.pkl':1665919465346}


def main():
    destination=ROOT/'models/devit';destination.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for name,identifier in FILES.items():
        path=destination/name
        if not path.exists():
            url=f'https://rutgers.box.com/index.php?rm=box_download_shared_file&shared_name=2lco6ab66pn3ufq6rh4gmyfzg9vfkm23&file_id=f_{identifier}'
            with requests.get(url,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type',''): raise ValueError('Expected binary checkpoint')
                temporary=path.with_suffix(path.suffix+'.part')
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024): stream.write(chunk)
            temporary.replace(path)
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        receipts.append({'name':name,'box_file_id':identifier,'bytes':path.stat().st_size,'sha256':digest})
        print(name,path.stat().st_size,flush=True)
    (destination/'provenance.json').write_text(json.dumps(receipts,indent=2))


if __name__=='__main__':main()

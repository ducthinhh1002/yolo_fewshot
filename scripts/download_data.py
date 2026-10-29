"""Download pinned public datasets; retain provenance, checksums and original files."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
RESOLVED_SOURCES = {}
SOURCES = {
    'cnmc': ('https://www.kaggle.com/api/v1/datasets/download/andrewmvd/leukemia-classification?datasetVersionNumber=2', 'https://www.kaggle.com/api/v1/datasets/view/andrewmvd/leukemia-classification'),
    'taleqani': ('https://www.kaggle.com/api/v1/datasets/download/mehradaria/leukemia?datasetVersionNumber=1', 'https://www.kaggle.com/api/v1/datasets/view/mehradaria/leukemia'),
    'bccd': ('https://codeload.github.com/Shenggan/BCCD_Dataset/zip/d272fb14cdff6e473fafeeeba32aba5f560e9e43', None),
    'aml': ('https://www.kaggle.com/api/v1/datasets/download/umarsani1605/aml-cytomorphology-2?datasetVersionNumber=1', 'https://www.kaggle.com/api/v1/datasets/view/umarsani1605/aml-cytomorphology-2'),
}


def download(name):
    url, metadata_url = SOURCES[name]
    canonical_url = url
    url = RESOLVED_SOURCES.get(name, url)
    provenance = ROOT / 'data/provenance'
    provenance.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data/downloads' / (name + '.zip')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if metadata_url and name not in RESOLVED_SOURCES:
        r = requests.get(metadata_url, timeout=60)
        r.raise_for_status()
        (provenance / (name + '_metadata.json')).write_text(json.dumps(r.json(), ensure_ascii=False, indent=2), encoding='utf-8')
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            resolved = response.url
            total = int(response.headers.get('Content-Length', 0))
            ranged = response.headers.get('Accept-Ranges') == 'bytes' and total > 200_000_000
            if ranged:
                response.close()
                chunks_dir = archive.parent / (name + '_chunks')
                chunks_dir.mkdir(exist_ok=True)
                prefix = partial.stat().st_size if partial.exists() else 0
                chunk_size = 8 * 1024 * 1024

                def fetch_chunk(bounds):
                    start, end = bounds
                    target = chunks_dir / f'{start}-{end}.bin'
                    if target.exists() and target.stat().st_size == end-start+1:
                        return target
                    for attempt in range(4):
                        try:
                            r = requests.get(resolved, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 120))
                            r.raise_for_status()
                            if r.status_code != 206 or len(r.content) != end-start+1:
                                raise ValueError('Invalid HTTP range response')
                            target.write_bytes(r.content)
                            return target
                        except (requests.RequestException, ValueError):
                            if attempt == 3:
                                raise
                            time.sleep(2)

                bounds = [(start, min(start+chunk_size-1, total-1)) for start in range(prefix, total, chunk_size)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                    futures = [pool.submit(fetch_chunk, b) for b in bounds]
                    for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        future.result()
                        if done % 8 == 0:
                            print(f'{name}: {done}/{len(bounds)} parallel chunks complete', flush=True)
                with partial.open('ab') as out:
                    for start, end in bounds:
                        out.write((chunks_dir / f'{start}-{end}.bin').read_bytes())
                if partial.stat().st_size != total or not zipfile.is_zipfile(partial):
                    raise ValueError('Invalid assembled archive')
                partial.replace(archive)
            else:
                size = 0
                tick = time.monotonic()
                with partial.open('wb') as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
                        size += len(chunk)
                        if time.monotonic() - tick > 20:
                            print(f'{name}: {size / 1024**2:.1f} MiB', flush=True)
                            tick = time.monotonic()
        if not archive.exists():
            if not zipfile.is_zipfile(partial):
                raise ValueError(f'{name}: server did not return a ZIP archive')
            partial.replace(archive)
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    dest = ROOT / 'data/raw' / name
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive member')
            if not target.exists() or target.stat().st_size != member.file_size:
                z.extract(member, dest)
    receipt = {'dataset': name, 'requested_url': canonical_url, 'sha256': digest, 'archive_bytes': archive.stat().st_size, 'archive_members': len(z.infolist())}
    (provenance / (name + '_download.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=SOURCES, default=['cnmc', 'aml', 'bccd'])
    parser.add_argument('--resolved-sources', type=Path, help='Temporary resolved public archive URLs when the server cannot access the Kaggle API')
    args = parser.parse_args()
    if args.resolved_sources:
        RESOLVED_SOURCES.update(json.loads(args.resolved_sources.read_text()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(download, args.datasets))

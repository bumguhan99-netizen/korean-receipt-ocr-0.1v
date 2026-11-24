"""Download official KORIE archives without changing existing project datasets."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

import gdown
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'detection_val': '15wXqZUzWaYEJu-rWZwCPuMvHFMZgWQOD',
    'detection_train': '1M3C_xG8Vg47DIbPP2fYpXs6mv5t6YOVq',
    'detection_test': '1UJZIcTX38FnMa8PZHYj--5OJ8-deSMRI',
    'ocr_val': '1v_0iGpBjB5WdWOeKI4C903eeqkRBjTsM',
    'ocr_train': '1I4BzOqKgF7zbNPlNeood4f7g8pi2xh26',
    'ocr_test': '1GtEzSUA2wTNfOujO67-JEZ_PLpJOdBhg',
    'ie_val': '1ff3z44tfhkeba-CvKv7fQp7EliFQ8EGu',
    'ie_train': '1W6XYnRtsQ2E8UZlo-MGwxQ0SkByTsdQV',
    'ie_test': '107ckPQ59gda7172Ls_iiEg5JomWhLtVA',
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def snapshot():
    folder = ROOT / 'sources'
    folder.mkdir(exist_ok=True)
    target = folder / 'korie_repository.json'
    if target.exists():
        return
    response = requests.get('https://api.github.com/repos/MahmoudSalah/KORIE/commits/main', timeout=60)
    response.raise_for_status()
    commit = response.json()['sha']
    response = requests.get(f'https://raw.githubusercontent.com/MahmoudSalah/KORIE/{commit}/README.md', timeout=60)
    response.raise_for_status()
    readme = folder / f'KORIE_README_{commit}.md'
    readme.write_bytes(response.content)
    save_new(target, dict(commit=commit, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          readme=readme.name, sha256=sha256(readme),
                          license='No LICENSE identified in README; usage conditions require confirmation'))


def acquire(name):
    archives = ROOT / 'archives'
    archives.mkdir(exist_ok=True)
    records = ROOT / 'records'
    records.mkdir(exist_ok=True)
    path = archives / f"{name}.{'csv' if name.startswith('ie_') else 'zip'}"
    record = records / f'{name}_download.json'
    url = f'https://drive.google.com/file/d/{SOURCES[name]}/view'
    if record.exists():
        previous = json.loads(record.read_text(encoding='utf-8'))
        if not path.exists() or sha256(path) != previous['sha256']:
            raise ValueError(f'Previously recorded archive changed: {name}')
        print(f'Already verified: {name}', flush=True)
        return
    if not path.exists():
        temporary = archives / f'{name}.partial'
        if not temporary.exists() or not name.startswith('ie_'):
            result = gdown.download(url, str(temporary), fuzzy=True, resume=True, quiet=True)
            if not result:
                raise RuntimeError(f'Download failed: {name}')
        if name.startswith('ie_'):
            with temporary.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or 'image_name' not in reader.fieldnames:
                    raise ValueError(f'Unexpected IE CSV header: {reader.fieldnames}')
                list(reader)
        elif not zipfile.is_zipfile(temporary):
            raise ValueError(f'Download is not ZIP: {temporary}')
        temporary.rename(path)
    if name.startswith('ie_'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        data = dict(name=name, source_url=url, local_file=path.name, size=path.stat().st_size,
                    sha256=sha256(path), downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    format='CSV', fields=fields, rows=len(rows),
                    receipt_ids=len({row['image_name'] for row in rows}),
                    has_bounding_boxes=False)
        save_new(record, data)
        print(json.dumps(data, ensure_ascii=False), flush=True)
        return
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
        entries = archive.infolist()
        for entry in entries:
            normalized = entry.filename.replace('\\', '/')
            if normalized.startswith('/') or '..' in Path(normalized).parts or ':' in normalized:
                raise ValueError(f'Unsafe ZIP member: {entry.filename}')
        data = dict(name=name, source_url=url, local_file=path.name, size=path.stat().st_size,
                    sha256=sha256(path), downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    entries=len(entries), uncompressed_bytes=sum(e.file_size for e in entries),
                    extensions=dict(collections.Counter(Path(e.filename).suffix.lower() for e in entries if not e.is_dir())),
                    members=[dict(name=e.filename, size=e.file_size, crc=e.CRC) for e in entries])
    save_new(record, data)
    print(json.dumps({k: v for k, v in data.items() if k != 'members'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+', choices=list(SOURCES))
    args = parser.parse_args()
    snapshot()
    for name in args.names:
        acquire(name)

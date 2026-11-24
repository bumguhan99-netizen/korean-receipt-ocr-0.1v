"""Extract official archives into new roots, checking traversal and CRC."""
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parent

def main():
    records = []
    for kind in ('ocr', 'detection'):
        for split in ('train', 'val', 'test'):
            archive_path = ROOT / 'archives' / f'{kind}_{split}.zip'
            destination = ROOT / 'data' / kind
            marker = destination / f'{split}_extraction.json'
            if marker.exists():
                records.append(json.loads(marker.read_text()))
                continue
            with zipfile.ZipFile(archive_path) as archive:
                entries = archive.infolist()
                for entry in entries:
                    name = entry.filename.replace('\\', '/')
                    target = (destination / name).resolve()
                    if not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                        raise ValueError(f'Unsafe member: {name}')
                    if name.split('/')[0] != split:
                        raise ValueError(f'Unexpected split layout: {name}')
                    if target.exists():
                        raise FileExistsError(target)
                bad = archive.testzip()
                if bad:
                    raise ValueError(f'CRC failure: {bad}')
                destination.mkdir(parents=True, exist_ok=True)
                archive.extractall(destination)
            record = dict(archive=archive_path.name, split=split, members=len(entries), crc='passed')
            with marker.open('x') as stream:
                json.dump(record, stream, indent=2)
            records.append(record)
            print(record, flush=True)
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()

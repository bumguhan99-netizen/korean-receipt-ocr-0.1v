"""Rename generated images/labels to matching stable numeric filenames."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'dataset'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    records = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    index = json.loads((ROOT/'visual_review_source_index.json').read_text(encoding='utf-8'))
    index_path = ROOT/'name_index.json'
    names = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {r['id']:f'{n:03d}' for n,r in enumerate(index,1)}
    moves = []
    for record in records:
        name = names[record['id']]
        for field, folder, extension in [('image','images','.jpg'), ('label','labels','.json')]:
            source = (ROOT/record[field]).resolve()
            target = (ROOT/folder/(name+extension)).resolve()
            assert source.parent == target.parent == (ROOT/folder).resolve(), 'Unexpected target directory'
            assert source.is_file(), source
            if source != target:
                assert not target.exists(), f'Refusing to replace existing file: {target}'
                moves.append((source,target,hashlib.sha256(source.read_bytes()).hexdigest()))
            record[field] = target.relative_to(ROOT).as_posix()
        record['display_name'] = name
    # Preflight all destinations before changing any file.
    for source,target,digest in moves:
        source.rename(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    write_json(index_path,names)
    write_json(ROOT/'manifest.json',records)
    by_id = {r['id']:r for r in records}
    for split in ['train','validation','test']:
        path=ROOT/f'{split}.jsonl'
        entries=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for entry in entries:
            entry.update(by_id[entry['id']])
        path.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in entries),encoding='utf-8')
    with (ROOT/'filename_mapping.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','source','image','label','split','group_id','source_license'])
        writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in records)
    print(f'Renamed {len(moves)} files; {len(records)} image-label pairs verified.')


if __name__ == '__main__': main()

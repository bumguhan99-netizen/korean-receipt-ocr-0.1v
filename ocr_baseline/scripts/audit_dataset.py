"""Read-only audit; preserve GT whitespace and official files."""
import argparse
from collections import defaultdict
import itertools
import random
import re
from pathlib import Path
from PIL import Image
from common import DEFAULT_DATA, MANIFEST_FIELDS, sha256, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--make-clean-split', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows, errors, counts = [], [], {}
    for split in ('train', 'val', 'test'):
        folder = root / split
        if not folder.is_dir():
            errors.append(f'Missing split: {folder}')
            continue
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))
        labels = {p for p in folder.rglob('*') if p.suffix.lower() == '.txt'}
        paired = set()
        counts[split] = dict(images=len(images), labels=len(labels))
        if not images:
            errors.append(f'Empty split: {split}')
        for image in images:
            label = image.with_suffix('.txt')
            try:
                if label not in labels:
                    raise ValueError('Missing matching TXT')
                paired.add(label)
                # newline='' preserves internal CR/LF; strip file-terminal CR/LF only.
                with label.open(encoding='utf-8', newline='') as stream:
                    gt = stream.read().rstrip('\r\n')
                if not gt or not gt.split() or '\n' in gt or '\r' in gt:
                    raise ValueError('Empty/whitespace-only or multiline GT')
                match = re.fullmatch(r'(IMG\d+)_(.+)', image.stem)
                if not match:
                    raise ValueError('Unexpected receipt/field filename')
                with Image.open(image) as im:
                    im.verify()
                rows.append(dict(eval_split=split, source_split=split, receipt_id=match[1], field=match[2],
                                 image_path=image.relative_to(root).as_posix(), gt_text=gt, image_sha256=sha256(image)))
            except Exception as error:
                errors.append(f'{image.relative_to(root)}: {type(error).__name__}: {error}')
        errors.extend(f'Orphan TXT: {p.relative_to(root)}' for p in sorted(labels - paired))
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['image_sha256']].append(row)
    conflicts = [dict(image_sha256=h, records=group) for h, group in by_hash.items()
                 if len({r['gt_text'] for r in group}) > 1]
    if conflicts:
        errors.append(f'Conflicting GT for {len(conflicts)} image hashes')
    overlaps = {}
    for left, right in itertools.combinations(('train', 'val', 'test'), 2):
        subsets = [[r for r in rows if r['source_split'] == s] for s in (left, right)]
        overlaps[f'{left}_{right}'] = {key:len({r[key] for r in subsets[0]} & {r[key] for r in subsets[1]})
                                             for key in ('receipt_id', 'image_sha256')}
    report = dict(data_root=str(root), counts=counts, total_images=len(rows), unique_image_hashes=len(by_hash),
                  unique_receipt_ids=len({r['receipt_id'] for r in rows}), overlaps=overlaps,
                  duplicate_hash_groups=sum(len(g)>1 for g in by_hash.values()), conflicts=conflicts,
                  errors=errors, status='failed' if errors else 'passed',
                  gt_handling='UTF-8; only terminal CR/LF removed; spaces/punctuation preserved')
    write_json(args.output_root / 'data_audit.json', report)
    if errors:
        raise RuntimeError(f'Audit failed: {errors[:10]}')
    write_csv(args.output_root / 'official_manifest.csv', rows, MANIFEST_FIELDS)
    write_csv(args.output_root / 'label_samples.csv',
              [r for s in ('train', 'val', 'test') for r in [x for x in rows if x['source_split']==s][:20]], MANIFEST_FIELDS)
    if args.make_clean_split:
        # Union receipt IDs sharing identical image bytes, even if IDs differ.
        parent = {r['receipt_id']:r['receipt_id'] for r in rows}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for group in by_hash.values():
            for row in group[1:]:
                parent[find(row['receipt_id'])] = find(group[0]['receipt_id'])
        units = sorted({find(x) for x in parent})
        random.Random(args.seed).shuffle(units)
        n = len(units)
        assignments = {unit:('train' if i<int(n*.8) else 'val' if i<int(n*.9) else 'test') for i,unit in enumerate(units)}
        clean = [dict(group[0], eval_split=assignments[find(group[0]['receipt_id'])]) for _,group in sorted(by_hash.items())]
        for key in ('receipt_id', 'image_sha256'):
            seen = {}
            for row in clean:
                assert seen.setdefault(row[key], row['eval_split']) == row['eval_split'], 'Clean split leakage'
        write_csv(args.output_root / 'clean_manifest.csv', clean, MANIFEST_FIELDS)
        write_json(args.output_root / 'clean_split_info.json', dict(seed=args.seed, grouping='receipt_id union shared hashes',
            unique_images=len(clean), groups=n, counts={s:sum(r['eval_split']==s for r in clean) for s in ('train','val','test')},
            official_comparable=False, official_manifest_sha256=sha256(args.output_root/'official_manifest.csv')))
    print(report)


if __name__ == '__main__':
    main()

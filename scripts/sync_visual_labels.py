"""Sync core transcriptions without changing original automatic OCR or splits."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'dataset'


def main():
    records = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    corrections = {r['id']: r for r in json.loads((ROOT / 'visual_corrections.json').read_text(encoding='utf-8'))}
    pending = []
    for record in records:
        path = ROOT / record['label']
        label = json.loads(path.read_text(encoding='utf-8-sig'))
        correction = corrections[record['id']]
        assert correction['source_sha256'] == record['source_sha256']
        assert label['annotation_status'] != 'reviewed', 'Do not overwrite reviewed labels'
        label['fields'].update({key: correction[key] for key in (
            'merchant_name', 'transaction_date', 'total_amount', 'paid_amount', 'currency')})
        if 'items' in correction:
            label['fields']['items'] = correction['items']
        label['fields_status'] = correction['status']
        label['field_status'] = {key.removesuffix('_status'): value for key, value in correction.items()
                                 if key.endswith('_status')}
        label['items_status'] = 'user_corrected' if 'items' in correction else 'not_structured'
        label['transcription'] = {key: correction[key] for key in
            ('text', 'scope', 'method', 'status', 'notes', 'user_confirmation') if key in correction}
        if 'documents' in correction:
            label['documents'] = correction['documents']
            label['items_status'] = 'structured_per_document'
        pending.append((path, label))
    backup = ROOT / 'backups' / 'before_core_label_sync'
    backup.mkdir(parents=True, exist_ok=True)
    for path, label in pending:
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split in ('train', 'validation', 'test'):
        path = ROOT / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            row['annotation'] = json.loads((ROOT / row['label']).read_text(encoding='utf-8'))
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(f'Synced core labels and JSONL: {len(pending)}; original OCR preserved')


if __name__ == '__main__':
    main()

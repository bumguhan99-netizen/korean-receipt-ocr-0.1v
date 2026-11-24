"""Export Excel-readable Korean CSVs from the existing receipt labels."""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
EXPORTS = ROOT / 'reports' / 'receipts'


def excel_text(value):
    # OCR strings must remain text, including lines starting with =, +, - or @.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def save(name, columns, rows):
    EXPORTS.mkdir(parents=True, exist_ok=True)
    path = EXPORTS / name
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[excel_text(v) for v in row] for row in rows])
    with path.open(encoding='utf-8-sig', newline='') as handle:
        loaded = list(csv.reader(handle))
    assert len(loaded) == len(rows) + 1
    assert all(len(row) == len(columns) for row in loaded)
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    print(f'{name}: {len(rows)} rows')


def main():
    records = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    correction_path = DATA / 'visual_corrections.json'
    corrections = {r['id']: r for r in json.loads(correction_path.read_text(encoding='utf-8'))} if correction_path.exists() else {}
    if corrections:
        backup = DATA / 'ocr_original_exports'
        backup.mkdir(exist_ok=True)
        for name in ['영수증_목록.csv', '영수증_줄별내용.csv']:
            if (EXPORTS/name).exists() and not (backup/name).exists():
                shutil.copy2(EXPORTS/name, backup/name)
    summaries, details = [], []
    status_names = {'auto_labeled_needs_review': '자동 인식·미검수', 'pending': '인식 대기', 'reviewed': '검수 완료', 'rejected': '제외'}
    for number, record in enumerate(sorted(records, key=lambda r: r.get('display_name', r['source'])), 1):
        number = int(record['display_name']) if record.get('display_name', '').isdigit() else number
        label = json.loads((DATA / record['label']).read_text(encoding='utf-8-sig'))
        lines = (label.get('ocr') or {}).get('lines', [])
        fields = label.get('fields') or {}
        status = status_names.get(label['annotation_status'], label['annotation_status'])
        source_name = Path(record['source']).name
        correction = corrections.get(record['id'])
        if correction:
            assert correction['source_sha256'] == record['source_sha256'], 'Stale correction'
            fields = correction
            lines = [{'text': text} for text in correction['text'].splitlines()]
            status = 'AI 이미지 대조·핵심항목'
        notes = correction['notes'] if correction else '자동 OCR 미검수'
        if correction and fields.get('merchant_name') is None and not notes:
            notes = '상호명 확인 불가'
        summaries.append([
            number, Path(record['image']).name, Path(record['label']).name,
            fields.get('merchant_name') or {'not_present':'미기재','cropped_out':'잘림'}.get(fields.get('merchant_name_status'),'확인 불가'), fields.get('transaction_date') or '',
            fields.get('total_amount') if fields.get('total_amount') is not None else '',
            fields.get('paid_amount') if fields.get('paid_amount') is not None else '',
            '\n'.join(line['text'] for line in lines), notes,
        ])
        for line_number, line in enumerate(lines, 1):
            details.append([number, Path(record['image']).name, line_number, line['text']])
    save('영수증_목록.csv', ['번호', '이미지', '라벨', '상호', '날짜', '합계', '결제액', '내용', '메모'], summaries)
    save('영수증_줄별내용.csv', ['번호', '이미지', '줄', '내용'], details)


if __name__ == '__main__':
    main()

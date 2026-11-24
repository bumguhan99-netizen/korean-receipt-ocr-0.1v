"""KORIE preparation/configuration and paired recognition evaluation.

Original data and baseline code are read-only. Run --help for commands.
"""
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import sys
import traceback
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ocr_baseline/scripts'))
from metrics import measure, aggregate


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return path


def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    return dict(n=len(values), **{f'p{p}': values[math.ceil(p / 100 * len(values))-1]
                                  for p in (50, 95, 99, 100)})


def dictionary(path):
    chars = Path(path).read_text(encoding='utf-8').splitlines()
    if not chars or any(len(c) != 1 for c in chars) or len(chars) != len(set(chars)):
        raise ValueError('Expected unique single-character official dictionary lines')
    return chars


def prepare(args):
    from PIL import Image
    import yaml
    if args.output.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError('Output must be outside original dataset')
    if args.max_length < 1:
        raise ValueError('max-length must be positive')
    chars = dictionary(args.dictionary)
    native = yaml.safe_load(args.reference.read_text(encoding='utf-8'))['PostProcess']['character_dict']
    if native not in (chars, chars + [' ']):
        raise ValueError('Official training dictionary/order differs from baseline inference dictionary')
    allowed = set(chars) | {' '}
    args.output.mkdir(parents=True, exist_ok=False)
    rows, issues, summary = [], [], {}
    for split in ('train', 'val', 'test'):
        folder = args.data_root / split
        images = sorted(folder.rglob('*.jpg'))
        labels = set(folder.rglob('*.txt'))
        paired = set()
        lengths, widths, reasons, oov = [], [], collections.Counter(), collections.Counter()
        split_rows = []
        if not images:
            raise ValueError(f'Missing/empty split: {folder}')
        for image in images:
            rel = image.relative_to(args.data_root).as_posix()
            label = image.with_suffix('.txt')
            errors = []
            row = dict(image_path=rel, label_path=label.relative_to(args.data_root).as_posix(),
                       source_split=split, eval_split=split)
            try:
                with label.open(encoding='utf-8', newline='') as f:
                    raw = f.read().rstrip('\r\n')
                paired.add(label)
                match = re.fullmatch(r'(IMG\d+)_(.+)', image.stem)
                if not match or not raw or not raw.split() or any(c in raw + rel for c in '\r\n\t'):
                    raise ValueError('Invalid filename/empty/multiline/tab label')
                with Image.open(image) as im:
                    im.load()
                    width, height = im.size
                text = unicodedata.normalize('NFC', raw)
                missing = sorted(set(text) - allowed)
                if missing:
                    errors.append('oov')
                    oov.update(c for c in text if c not in allowed)
                if len(text) > args.max_length:
                    errors.append('length')
                elif len(text) >= args.max_length - 1:
                    errors.append('nrtr_length')
                lengths.append(len(text))
                widths.append(round(48 * width / height, 2))
                row.update(receipt_id=match[1], field=match[2], gt_raw=raw, gt_train=text,
                           image_sha256=digest(image), label_sha256=digest(label),
                           width=width, height=height, length=len(text),
                           oov=[f'U+{ord(c):04X} {c}' for c in missing],
                           nfc_changed=raw != text)
            except Exception as exc:
                errors.append('invalid_input')
                row['error'] = str(exc)
            row.update(issues=errors, trainable=not errors)
            reasons.update(errors)
            for reason in errors:
                issues.append(dict(image_path=rel, split=split, reason=reason,
                                   detail=row.get('error', json.dumps(row.get('oov', []), ensure_ascii=False))))
            rows.append(row)
            split_rows.append(row)
        for label in sorted(labels - paired):
            issues.append(dict(image_path=label.relative_to(args.data_root).as_posix(), split=split,
                               reason='orphan_or_unreadable_label', detail='Not successfully paired'))
        good = [r for r in split_rows if r['trainable']]
        with (args.output / f'{split}_list.txt').open('w', encoding='utf-8', newline='\n') as f:
            for r in good:
                f.write(r['image_path'] + '\t' + r['gt_train'] + '\n')
        summary[split] = dict(images=len(images), labels=len(labels), usable=len(good),
                              held=len(images)-len(good), reasons=dict(reasons),
                              nfc_changed=sum(r.get('nfc_changed', False) for r in split_rows),
                              oov={f'U+{ord(c):04X} {c}': n for c, n in oov.items()})
        # Test compatibility counts are recorded, never used to choose dimensions.
        if split != 'test':
            summary[split].update(length=stats(lengths), width_at_height48=stats(widths))
    with (args.output / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = dict(status='passed', data_root=str(args.data_root.resolve()),
                  dictionary_sha256=digest(args.dictionary), max_text_length=args.max_length,
                  normalization='NFC; terminal CR/LF only removed', seed=args.seed, counts=summary)
    fatal = any(x['reason'] not in ('oov', 'length', 'nrtr_length') for x in issues)
    report['status'] = 'failed' if fatal else 'passed'
    dump(args.output / 'validation_report.json', report)
    table(args.output / 'issues.csv', issues, ['image_path', 'split', 'reason', 'detail'])
    train = [r for r in rows if r['source_split'] == 'train' and r['trainable']]
    smoke = random.Random(args.seed).sample(train, min(64, len(train)))
    (args.output / 'train_smoke_list.txt').write_text(
        ''.join(r['image_path'] + '\t' + r['gt_train'] + '\n' for r in smoke), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if fatal:
        raise ValueError('Input errors: inspect validation_report.json and issues.csv')


def config(args):
    import yaml
    repo = args.repo.resolve()
    source = repo / 'configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml'
    cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
    report = read(args.data / 'validation_report.json')
    dict_path = repo / 'ppocr/utils/dict/ppocrv5_korean_dict.txt'
    if report['status'] != 'passed' or report['dictionary_sha256'] != digest(dict_path):
        raise ValueError('Preparation failed or dictionary changed')
    if report['max_text_length'] != cfg['Global']['max_text_length']:
        raise ValueError('Prepared max length differs from official architecture; no silent head change')
    if not args.weights.is_file():
        raise ValueError('Missing official pretrained .pdparams')
    g = cfg['Global']
    g.update(pretrained_model=args.weights.resolve().as_posix(), checkpoints=None,
             character_dict_path=dict_path.as_posix(), save_model_dir=args.output_dir.resolve().as_posix(),
             distributed=False, seed=42, save_epoch_step=1)
    for section, split in (('Train', 'train'), ('Eval', 'val')):
        cfg[section]['dataset'].update(data_dir=report['data_root'].replace('\\', '/'),
                                      label_file_list=[(args.data / f'{split}_list.txt').resolve().as_posix()])
    cfg['Eval']['loader']['shuffle'] = False
    cfg['Metric'].update(ignore_space=False, is_filter=False)
    if args.profile == 'rtx5060ti':
        g.update(epoch_num=30, eval_batch_step=[0, 100])
        cfg['Optimizer']['lr'].update(learning_rate=1e-5, warmup_epoch=1)
        cfg['Train']['sampler']['first_bs'] = 32
        cfg['Train']['loader'].update(batch_size_per_card=32, num_workers=4)
        cfg['Eval']['loader'].update(batch_size_per_card=32, num_workers=2)
    if args.smoke:
        g.update(epoch_num=2, eval_batch_step=[0, 10])
        cfg['Optimizer']['lr']['warmup_epoch'] = 1
        cfg['Train']['dataset']['label_file_list'] = [(args.data / 'train_smoke_list.txt').resolve().as_posix()]
        cfg['Train']['sampler']['first_bs'] = 4
        for section in ('Train', 'Eval'):
            cfg[section]['loader'].update(batch_size_per_card=4, num_workers=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f'Wrote {args.output}; profile={args.profile}, smoke={args.smoke}')


def verify(args):
    report = read(args.data/'validation_report.json')
    if report['status'] != 'passed':
        raise ValueError('Preparation did not pass')
    root = Path(report['data_root'])
    rows = [json.loads(s) for s in (args.data/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        for key in ('image', 'label'):
            if digest(inside(root, row[key+'_path'])) != row[key+'_sha256']:
                raise ValueError(f'Original changed: {row["image_path"]}')
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['source_split'] == split]
        if len(subset) != report['counts'][split]['images']:
            raise ValueError('Manifest count differs from preparation report')
        expected = ''.join(r['image_path']+'\t'+r['gt_train']+'\n' for r in subset if r['trainable'])
        if (args.data/f'{split}_list.txt').read_text(encoding='utf-8') != expected:
            raise ValueError(f'Changed derived list: {split}')
    print(f'Original hashes and derived lists verified: {len(rows)} samples')


def packages():
    result = {}
    for name in ('paddlepaddle', 'paddlepaddle-gpu', 'paddleocr', 'paddlex', 'Pillow'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def model_info(folder):
    required = ['inference.json', 'inference.pdiparams', 'inference.yml']
    return {name: digest(folder / name) for name in required}


def evaluate(args):
    import yaml
    configs = [yaml.safe_load((p / 'inference.yml').read_text(encoding='utf-8'))
               for p in (args.baseline, args.candidate)]
    for key in ('PreProcess', 'PostProcess'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Model {key} differs; paired evaluation requires identical policy/dictionary')
    protocol = dict(manifest_sha256=digest(args.manifest),
                    models={k: model_info(p) for k, p in [('baseline', args.baseline), ('candidate', args.candidate)]},
                    device=args.device, batch_size=1, normalization='NFC', packages=packages(),
                    amount_fields=sorted(args.amount_fields), enable_mkldnn=False, cpu_threads=4)
    if args.split == 'test':
        if not args.frozen or read(args.frozen)['protocol'] != protocol:
            raise ValueError('Test requires matching frozen protocol from completed full validation')
        # One final test per data directory, independent of run folder/model selection.
        with (args.manifest.parent / 'final_test_started.json').open('x', encoding='utf-8') as f:
            json.dump(protocol, f, ensure_ascii=False)
    rows = [json.loads(s) for s in args.manifest.read_text(encoding='utf-8').splitlines()]
    rows = sorted((r for r in rows if r['source_split'] == args.split), key=lambda r: r['image_path'])
    if not rows or len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate evaluation samples')
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', split=args.split, protocol=protocol, n=len(rows))
    dump(args.output / 'run_info.json', info)
    try:
        for row in rows:
            if 'invalid_input' in row['issues']:
                raise ValueError(f'Invalid input: {row["image_path"]}')
            for key in ('image', 'label'):
                if digest(inside(args.data_root, row[f'{key}_path'])) != row[f'{key}_sha256']:
                    raise ValueError(f'Changed {key}: {row["image_path"]}')
        from paddleocr import TextRecognition
        results, summaries = {}, {}
        for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
            started = time.monotonic()
            model = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(path.resolve()),
                                    device=args.device, enable_mkldnn=False, cpu_threads=4)
            scored = []
            for row in rows:
                output = list(model.predict(input=str(inside(args.data_root, row['image_path'])), batch_size=1))
                if len(output) != 1:
                    raise ValueError('Expected one recognition result per crop')
                payload = output[0].json
                if isinstance(payload, str):
                    payload = json.loads(payload)
                payload = payload.get('res', payload)
                prediction = payload['rec_text']
                if not isinstance(prediction, str) or not math.isfinite(float(payload['rec_score'])):
                    raise ValueError('Invalid prediction/confidence')
                scored.append(dict(image_path=row['image_path'], field=row['field'],
                                   ground_truth=row['gt_raw'], prediction=prediction,
                                   **measure(row['gt_raw'], prediction)))
                if len(scored) % 100 == 0:
                    print(f'{name}: {len(scored)}/{len(rows)}', flush=True)
            results[name] = scored
            amount = [r for r in scored if r['field'] in args.amount_fields]
            summaries[name] = dict(**aggregate(scored), amount_n=len(amount),
                                   amount_exact_pct=aggregate(amount)['exact_match_pct'],
                                   seconds=time.monotonic()-started)
            table(args.output / f'{name}_predictions.csv', scored, list(scored[0]))
            fields = [dict(field=f, **aggregate(r for r in scored if r['field'] == f))
                      for f in sorted({r['field'] for r in scored})]
            table(args.output / f'{name}_by_field.csv', fields, list(fields[0]))
            del model
        changes = [dict(image_path=b['image_path'], field=b['field'], ground_truth=b['ground_truth'],
                        baseline=b['prediction'], candidate=c['prediction'],
                        char_edits_delta=c['char_edits']-b['char_edits'])
                   for b, c in zip(results['baseline'], results['candidate'])]
        for name, predicate in [('improved', lambda d: d < 0), ('regressed', lambda d: d > 0)]:
            table(args.output / f'{name}.csv', [r for r in changes if predicate(r['char_edits_delta'])], list(changes[0]))
        dump(args.output / 'summary.json', summaries)
        info['status'] = 'completed'
        dump(args.output / 'run_info.json', info)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    except BaseException:
        info.update(status='incomplete', error=traceback.format_exc())
        dump(args.output / 'run_info.json', info)
        raise


def freeze(args):
    info = read(args.run / 'run_info.json')
    if info['status'] != 'completed' or info['split'] != 'val':
        raise ValueError('Only a completed full val comparison may freeze test settings')
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(dict(protocol=info['protocol'], source_run=str(args.run.resolve())), f, indent=2)


def report(args):
    base = args.root
    lines = ['# KORIE GPU 파인튜닝 실행 결과', '',
             '공식 한국어 v3.7.0 구조·사전, RTX 5060 Ti용 설정. 원본 split 유지.', '',
             '학습 설정: first_bs 32(멀티스케일 32/21/16), train workers 4, eval workers 2, '
             '최대 30 epochs, LR 1e-5, warmup 1. 선택 규칙: 내장 val acc 최고 checkpoint.', '']
    selection = base/'resource_probes/selected.json'
    if selection.exists():
        picked = read(selection)
        lines += [f'처리량 실측 후 최종 first_bs={picked["batch"]}, train workers={picked["workers"]}. '
                  f'probe 처리량 {picked["samples_per_second"]:.2f} samples/s, '
                  f'peak reserved {picked["peak_reserved_mb"]} MiB. '
                  '위 32/4 시작값보다 이 최종 선택값이 우선한다.', '']
    data_report = base/'data/validation_report.json'
    if data_report.exists():
        data = read(data_report)
        lines += ['| split | 원본 | 학습/내장평가 가능 | 보류 |', '|---|---:|---:|---:|']
        for split, c in data['counts'].items():
            lines.append(f'| {split} | {c["images"]} | {c["usable"]} | {c["held"]} |')
        lines += ['', '보류는 OOV/길이 등의 사유로 issues.csv에 기록. 외부 val/test는 보류 표본도 포함한다.', '']
    lines += ['| 평가 | 모델 | n | CER % ↓ | EM % ↑ | 숫자·기호 EM % | 금액 EM % |',
              '|---|---|---:|---:|---:|---:|---:|']
    for run in ('initial_val', 'selected_val', 'final_test'):
        folder = base/'evaluation'/run
        if not (folder/'run_info.json').exists() or read(folder/'run_info.json')['status'] != 'completed':
            lines.append(f'| {run} | 미완료/미실행 | | | | | |')
            continue
        for model, metric in read(folder/'summary.json').items():
            fmt = lambda v: '' if v is None else f'{v:.4f}'
            lines.append(f'| {run} | {model} | {metric["n"]} | {fmt(metric["cer_pct"])} | '
                         f'{fmt(metric["exact_match_pct"])} | {fmt(metric["numeric_symbol_only_exact_pct"])} | '
                         f'{fmt(metric["amount_exact_pct"])} |')
    lines += ['', 'initial_val: 기존 HF 모델 vs 공식 학습 가중치의 초기 export. '
              'selected_val/final_test: 초기 export vs 선택된 튜닝 export.', '',
              '금액은 Total/Subtotal/TotalTax/Item_Unit_Price/Item_Total_Price 문자열 EM이며, '
              '숫자·기호 지표에는 날짜·전화번호 등도 포함된다.', '',
              '전체 영수증 검출+인식 평가는 미실행: 고정 검출기와 대응 GT 확인 필요.', '',
              '증거: `records/`, `data/validation_report.json`, `evaluation/*/run_info.json`, '
              '`evaluation/*/summary.json`, `evaluation/*/*_predictions.csv`.']
    args.output.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-root', type=Path, default=ROOT / 'dataset/korie')
    p.add_argument('--dictionary', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT / 'ocr_baseline/models/korean_PP-OCRv5_mobile_rec/inference.yml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser('config')
    for name in ('repo', 'data', 'weights', 'output', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--profile', choices=['official', 'rtx5060ti'], default='rtx5060ti')
    p.set_defaults(func=config)
    p = sub.add_parser('verify')
    p.add_argument('--data', type=Path, required=True)
    p.set_defaults(func=verify)
    p = sub.add_parser('evaluate')
    for name in ('manifest', 'baseline', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=ROOT / 'dataset/korie')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--device', default='gpu:0')
    p.add_argument('--amount-fields', nargs='*', default=['Total', 'Subtotal', 'TotalTax', 'Item_Unit_Price', 'Item_Total_Price'])
    p.add_argument('--frozen', type=Path)
    p.set_defaults(func=evaluate)
    p = sub.add_parser('freeze')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=freeze)
    p = sub.add_parser('report')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

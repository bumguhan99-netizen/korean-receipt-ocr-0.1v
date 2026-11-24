"""Recognition-only paired evaluation. Any missing/failed prediction aborts the run."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from common import MODELS, DEFAULT_DATA, read_csv, sha256, write_csv, write_json
from metrics import aggregate, measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/official_manifest.csv'))
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/baseline'))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--require-local-models', action='store_true')
    parser.add_argument('--run-tag', help='Separate directory for a documented retry')
    parser.add_argument('--frozen-test-config', type=Path, help='Required for one final Test run; JSON with manifest_sha256 and settings')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error('--max-samples must be positive')
    settings = dict(device=args.device, batch_size=1, normalization='Unicode NFC only',
                    tokenization='str.split()', enable_mkldnn=False, cpu_threads=4,
                    preprocessing='unmodified source crop; model-native preprocessing', postprocessing='model-native decoding only',
                    models=list(MODELS))
    manifest_hash = sha256(args.manifest)
    if args.split == 'test':
        if not args.frozen_test_config or args.max_samples or not args.require_local_models:
            parser.error('Test requires frozen settings, local pinned models and full split; use Val to tune')
        frozen = json.loads(args.frozen_test_config.read_text(encoding='utf-8'))
        assert frozen['manifest_sha256'] == manifest_hash and frozen['settings'] == settings, 'Test settings mismatch'
    rows = sorted([r for r in read_csv(args.manifest) if r['eval_split'] == args.split], key=lambda r:r['image_path'])
    assert rows and len({r['image_path'] for r in rows}) == len(rows), 'Missing/duplicate samples'
    if args.max_samples:
        assert len(rows) >= args.max_samples, 'Requested smoke count exceeds split'
        rows = random.Random(args.seed).sample(rows, args.max_samples)
    tag = args.run_tag or (f'smoke_{args.max_samples}_seed{args.seed}' if args.max_samples else 'full')
    experiment = args.manifest.stem.removesuffix('_manifest')
    out = args.output_root / experiment / args.split / tag
    if args.split == 'test':
        # A different run tag must not silently allow repeated final-Test evaluations.
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent/'final_test_started.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(frozen_config=str(args.frozen_test_config), started=datetime.now(timezone.utc).isoformat()), stream)
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                python=sys.version, packages={k:importlib.metadata.version(k) for k in ('paddleocr','paddlepaddle','paddlex')},
                manifest_name=args.manifest.name, manifest_sha256=manifest_hash, eval_split=args.split,
                seed=args.seed, sample_count=len(rows), limited_to_n=args.max_samples, settings=settings,
                data_root=str(args.data_root.resolve()), models={})
    write_json(out/'run_info.json', info)
    try:
        # All sample hashes/labels are checked before any prediction, including stale manifest protection.
        for row in rows:
            path = (args.data_root / row['image_path']).resolve()
            assert path.is_relative_to(args.data_root.resolve()), 'Path outside DATA_ROOT'
            assert sha256(path) == row['image_sha256'], f'Changed image: {path}'
            with path.with_suffix('.txt').open(encoding='utf-8', newline='') as stream:
                assert stream.read().rstrip('\r\n') == row['gt_text'], f'Changed GT: {path}'
        write_csv(out/'samples.csv', rows)
        model_metadata = {}
        for name in MODELS:
            folder = args.model_root / name
            if args.require_local_models:
                metadata = json.loads((folder/'download_manifest.json').read_text(encoding='utf-8'))
                assert metadata['model'] == name
                for filename, expected in metadata['files'].items():
                    assert sha256(folder/filename) == expected, f'Model hash mismatch: {name}/{filename}'
                model_metadata[name] = metadata
        if args.require_local_models:
            assert len({m['files']['inference.pdiparams'] for m in model_metadata.values()}) == 2
            import yaml
            configs = [yaml.safe_load((args.model_root/name/'inference.yml').read_text(encoding='utf-8')) for name in MODELS]
            assert configs[0]['PreProcess'] == configs[1]['PreProcess'], 'Different preprocessing'
            assert configs[0]['PostProcess']['name'] == configs[1]['PostProcess']['name'], 'Different decoder algorithm'
            info['preprocessing_verified_equal'] = configs[0]['PreProcess']
            info['decoder'] = configs[0]['PostProcess']['name']
            info['native_dictionary_sizes'] = {name:len(config['PostProcess']['character_dict']) for name,config in zip(MODELS,configs)}
        info['model_downloads'] = model_metadata
        if args.split == 'test':
            assert frozen['model_downloads'] == model_metadata, 'Frozen Test model commit/hash mismatch'
        write_json(out/'run_info.json', info)
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        from paddleocr import TextRecognition
        summaries, identities = [], []
        for name in MODELS:
            started = time.perf_counter()
            info['models'][name] = dict(status='running')
            write_json(out/'run_info.json', info)
            kwargs = dict(model_name=name, device=args.device, enable_mkldnn=False, cpu_threads=4)
            if args.require_local_models:
                kwargs['model_dir'] = str((args.model_root/name).resolve())
            model = TextRecognition(**kwargs)
            predictions = []
            for index, row in enumerate(rows, 1):
                results = list(model.predict(input=str((args.data_root/row['image_path']).resolve()), batch_size=1))
                assert len(results) == 1, 'Expected exactly one recognition result'
                payload = results[0].json
                if isinstance(payload, str):
                    payload = json.loads(payload)
                data = payload.get('res', payload)
                prediction = data['rec_text']
                assert isinstance(prediction, str), 'rec_text must be a string'
                confidence = float(data['rec_score'])
                assert math.isfinite(confidence), 'Invalid recognition confidence'
                predictions.append(dict(model=name, **{k:v for k,v in row.items() if k!='gt_text'},
                    ground_truth=row['gt_text'], prediction=prediction, confidence=confidence, **measure(row['gt_text'], prediction)))
                if index == 1:
                    write_json(out/name/'first_result.json', payload)
                if index % 100 == 0 or index == len(rows):
                    print(f'{name}: {index}/{len(rows)}, {time.perf_counter()-started:.1f}s', flush=True)
            folder = out/name
            write_csv(folder/'predictions.csv', predictions)
            metric = aggregate(predictions)
            write_json(folder/'metrics.json', metric)
            grouped = defaultdict(list)
            for row in predictions:
                grouped[row['field']].append(row)
            write_csv(folder/'by_field.csv', [dict(field=field, **aggregate(group)) for field,group in sorted(grouped.items())])
            worst = sorted(predictions, key=lambda r:(r['char_edits']/max(1,r['gt_chars']),r['char_edits']), reverse=True)[:50]
            write_csv(folder/'worst_50.csv', worst)
            summaries.append(dict(model=name, **metric, eval_split=args.split, manifest_name=args.manifest.name,
                limited_to_n=args.max_samples, device=args.device, normalization='NFC', tokenization='whitespace'))
            identities.append([(r['image_path'],r['image_sha256'],r['ground_truth']) for r in predictions])
            info['models'][name] = dict(status='completed', n=len(predictions), seconds=time.perf_counter()-started)
            write_json(out/'run_info.json', info)
            del model
            gc.collect()
        assert identities[0] == identities[1], 'Model sample/GT sets differ'
        write_csv(out/'summary.csv', summaries)
        info['paired_samples_verified'] = True
        info['status'] = 'completed'
        info['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(out/'run_info.json', info)
        print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        info['status'] = 'incomplete'
        info['error'] = traceback.format_exc()
        write_json(out/'run_info.json', info)
        (out/'error.log').write_text(info['error'], encoding='utf-8')
        raise


if __name__ == '__main__':
    main()

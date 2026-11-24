"""Independently recompute saved scores and confirm paired sample identity."""
import argparse
import json
import math
from pathlib import Path
from common import MODELS, read_csv, sha256, write_json
from metrics import aggregate, measure


def verify(folder):
    info = json.loads((folder/'run_info.json').read_text(encoding='utf-8'))
    assert info['status'] == 'completed' and info['paired_samples_verified'], 'Incomplete run'
    samples = read_csv(folder/'samples.csv')
    expected = [(r['image_path'],r['image_sha256'],r['gt_text']) for r in samples]
    assert len(samples) == info['sample_count']
    summaries = {r['model']:r for r in read_csv(folder/'summary.csv')}
    assert set(summaries) == set(MODELS)
    scores = {}
    for name in MODELS:
        rows = read_csv(folder/name/'predictions.csv')
        assert [(r['image_path'],r['image_sha256'],r['ground_truth']) for r in rows] == expected
        for row in rows:
            assert row['model'] == name
            measured = measure(row['ground_truth'],row['prediction'])
            for key,value in measured.items():
                assert row[key] == str(value), (name,row['image_path'],key)
            assert math.isfinite(float(row['confidence']))
        scores[name] = aggregate(rows)
        stored = json.loads((folder/name/'metrics.json').read_text(encoding='utf-8'))
        assert scores[name] == stored
        for key,value in scores[name].items():
            assert summaries[name][key] == ('' if value is None else str(value)), (name,key)
        by_field = read_csv(folder/name/'by_field.csv')
        assert sum(int(r['n']) for r in by_field) == len(rows)
        assert {r['field'] for r in by_field} == {r['field'] for r in rows}
        for field_row in by_field:
            field_metrics = aggregate(r for r in rows if r['field']==field_row['field'])
            assert all(field_row[k] == ('' if v is None else str(v)) for k,v in field_metrics.items())
        assert info['models'][name]['status'] == 'completed'
    proof = dict(status='passed', paired_samples=len(samples), metrics_recomputed=True,
                 prediction_files_sha256={name:sha256(folder/name/'predictions.csv') for name in MODELS})
    write_json(folder/'verification.json', proof)
    print(f'{folder}: verified {len(samples)} paired samples and all saved metrics')
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    verify(args.run)

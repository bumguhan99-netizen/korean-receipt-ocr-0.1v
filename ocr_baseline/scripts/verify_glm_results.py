"""Recompute GLM scores; compare sample/GT identities with the saved Paddle run."""
import argparse
import json
from pathlib import Path
from common import MODELS, read_csv, sha256, write_csv, write_json
from metrics import aggregate, measure


def verify(run):
    info=json.loads((run/'run_info.json').read_text(encoding='utf-8'))
    assert info['status']=='completed'
    predictions=read_csv(run/'predictions.csv')
    samples=read_csv(run/'samples.csv')
    reference=Path(info['reference_run'])
    assert sha256(reference/'samples.csv')==info['reference_samples_sha256']
    reference_samples=read_csv(reference/'samples.csv')[:len(samples)]
    assert samples==reference_samples
    expected=[(r['image_path'],r['image_sha256'],r['gt_text']) for r in samples]
    assert [(r['image_path'],r['image_sha256'],r['ground_truth']) for r in predictions]==expected
    assert len(predictions)==info['sample_count']==info['completed_samples']
    for row in predictions:
        assert row['model']=='GLM-OCR' and row['confidence']==''
        calculated=measure(row['ground_truth'],row['prediction'])
        assert all(row[k]==str(v) for k,v in calculated.items())
    metrics=aggregate(predictions)
    assert metrics==json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    summary=read_csv(run/'summary.csv')
    assert len(summary)==1
    assert all(summary[0][k]==('' if v is None else str(v)) for k,v in metrics.items())
    field_rows=read_csv(run/'by_field.csv')
    assert sum(int(r['n']) for r in field_rows)==len(predictions)
    assert {r['field'] for r in field_rows}=={r['field'] for r in predictions}
    for row in field_rows:
        result=aggregate(r for r in predictions if r['field']==row['field'])
        assert all(row[k]==('' if v is None else str(v)) for k,v in result.items())
    partial=[json.loads(line) for line in (run/'predictions.partial.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [(r['image_path'],r['prediction']) for r in partial]==[(r['image_path'],r['prediction']) for r in predictions]
    proof=dict(status='passed',n=len(samples),reference_run=str(reference),
               prediction_sha256=sha256(run/'predictions.csv'),
               token_limit_reached=sum(int(r['token_limit_reached']) for r in predictions),
               metric_recomputation=True,identical_sample_order_gt_hash=True)
    if len(samples)==len(read_csv(reference/'samples.csv')):
        comparisons=read_csv(reference/'summary.csv')+summary
        write_csv(run/'comparison_summary.csv',comparisons)
        for name in MODELS:
            baseline=read_csv(reference/name/'predictions.csv')
            assert [(r['image_path'],r['image_sha256'],r['ground_truth']) for r in baseline]==expected
        proof['both_paddle_sample_lists_verified']=True
    write_json(run/'verification.json',proof)
    print(f'Verified GLM-OCR: {len(samples)} samples; metrics and Paddle sample identities match')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)

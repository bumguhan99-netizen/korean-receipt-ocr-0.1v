"""Linux training stages. Requires completed pre-training validation and real training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

PROJECT = Path(__file__).resolve().parents[1]
KORIE = PROJECT / 'training/korie.py'


def run(command, log, cwd=None):
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen([str(x) for x in command], cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
        if process.wait():
            raise RuntimeError(f'Command failed; inspect {log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'smoke', 'train', 'resume', 'export', 'validate', 'freeze', 'test'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--pre-eval', type=Path, help='Completed training/korie.py validation output directory')
    p.add_argument('--checkpoint', type=Path, help='Resume checkpoint prefix including matching .pdopt/.states')
    a = p.parse_args()
    a.run = a.run.resolve(); a.repo = a.repo.resolve(); a.data_root = a.data_root.resolve()
    if sys.platform != 'linux':
        p.error('Run inside the Linux GPU container')
    baseline = PROJECT / 'training/exports/selected'
    data = a.run / 'data'
    if a.stage == 'prepare':
        if not a.weights or a.weights.suffix != '.pdparams' or not a.weights.is_file():
            p.error('Provide verified first-round training .pdparams; inference.pdiparams cannot resume training')
        if not a.pre_eval:
            p.error('--pre-eval is required')
        info = json.loads((a.pre_eval / 'run_info.json').read_text())
        if info.get('status') != 'completed' or info.get('split') != 'val':
            p.error('Pre-training validation must be completed, not test')
        expected = {f:hashlib.sha256((baseline/f).read_bytes()).hexdigest()
                    for f in ('inference.json','inference.pdiparams','inference.yml')}
        if info['protocol']['models']['candidate'] != expected:
            p.error('Pre-evaluation candidate differs from first-round export')
        a.run.mkdir(parents=True, exist_ok=False)
        run([sys.executable,KORIE,'prepare','--data-root',a.data_root,'--dictionary',
             a.repo/'ppocr/utils/dict/ppocrv5_korean_dict.txt','--reference',baseline/'inference.yml',
             '--output',data],a.run/'prepare.log')
        for name in ('smoke','train'):
            cmd=[sys.executable,KORIE,'config','--repo',a.repo,'--data',data,'--weights',a.weights.resolve(),
                 '--output',a.run/f'{name}.yml','--output-dir',a.run/f'outputs_{name}']
            if name == 'smoke': cmd.append('--smoke')
            run(cmd,a.run/f'config_{name}.log')
        (a.run/'inputs.json').write_text(json.dumps(dict(weights=str(a.weights.resolve()),
            weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            pre_eval=str(a.pre_eval.resolve()), data_root=str(a.data_root),repo=str(a.repo)),indent=2))
        return
    recorded=json.loads((a.run/'inputs.json').read_text())
    if recorded['data_root'] != str(a.data_root) or recorded['repo'] != str(a.repo):
        p.error('Run paths differ from preparation')
    cfg=a.run/'train.yml'
    stamp=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log=a.run/f'{a.stage}_{stamp}.log'
    if a.stage in ('smoke','train','resume'):
        if a.stage != 'smoke' and not (a.run/'smoke_passed.json').exists():
            p.error('Successful smoke is required')
        if a.stage == 'train' and (a.run/'outputs_train').exists():
            p.error('Training output exists; use resume with an explicit checkpoint')
        cmd=[sys.executable,a.repo/'tools/train.py','-c',a.run/'smoke.yml' if a.stage=='smoke' else cfg]
        if a.stage=='resume':
            if not a.checkpoint or not all(Path(str(a.checkpoint)+s).is_file() for s in ('.pdparams','.pdopt','.states')):
                p.error('Resume requires checkpoint .pdparams/.pdopt/.states')
            cmd += ['-o',f'Global.checkpoints={a.checkpoint.resolve()}']
        run(cmd,log,a.repo)
        if a.stage=='smoke':
            content=log.read_text().lower()
            if any(s in content for s in ('not in loaded params','not matched','no kernel image','loss: nan','loss: inf')):
                raise RuntimeError('Smoke parameter/numerical mismatch; inspect log')
            if not (a.run/'outputs_smoke/latest.pdparams').exists():
                raise RuntimeError('Smoke checkpoint missing')
            (a.run/'smoke_passed.json').write_text(json.dumps({'log':str(log)}))
    elif a.stage=='export':
        target=a.run/'export'
        if target.exists(): p.error('Export exists; do not overwrite')
        ckpt=a.run/'outputs_train/best_accuracy'
        if not Path(str(ckpt)+'.pdparams').is_file(): p.error('Selected checkpoint missing')
        run([sys.executable,a.repo/'tools/export_model.py','-c',cfg,'-o',
             f'Global.pretrained_model={ckpt}',f'Global.save_inference_dir={target}'],log,a.repo)
    elif a.stage=='freeze':
        run([sys.executable,KORIE,'freeze','--run',a.run/'validation','--output',a.run/'frozen_test.json'],log)
    else:
        split='val' if a.stage=='validate' else 'test'
        cmd=[sys.executable,KORIE,'evaluate','--manifest',data/'manifest.jsonl','--data-root',a.data_root,
             '--baseline',baseline,'--candidate',a.run/'export','--split',split,
             '--output',a.run/('validation' if split=='val' else 'final_test')]
        if split=='test': cmd += ['--frozen',a.run/'frozen_test.json']
        run(cmd,log)


if __name__ == '__main__':
    main()

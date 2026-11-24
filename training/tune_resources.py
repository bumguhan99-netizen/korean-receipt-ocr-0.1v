"""Choose batch/workers by real training throughput, without looking at test accuracy."""
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import yaml

root = Path(__file__).resolve().parent
base = yaml.safe_load((root/'configs/official.yml').read_text())
lines = (root/'data/train_list.txt').read_text(encoding='utf-8').splitlines()
random.Random(42).shuffle(lines)
out = root/'resource_probes'
out.mkdir(exist_ok=False)
results = []


def probe(batch, workers):
    tag = f'b{batch}_w{workers}'
    cfg = copy.deepcopy(base)
    subset = out/f'{tag}.txt'
    subset.write_text('\n'.join(lines[:min(len(lines), batch*10)])+'\n', encoding='utf-8')
    cfg['Global'].update(epoch_num=1, print_batch_step=5, log_smooth_window=5,
                         eval_batch_step=[100000, 100000], save_model_dir=str(out/tag))
    cfg['Optimizer']['lr']['warmup_epoch'] = 0
    cfg['Train']['dataset']['label_file_list'] = [str(subset)]
    cfg['Train']['sampler']['first_bs'] = batch
    cfg['Train']['loader'].update(batch_size_per_card=batch, num_workers=workers)
    config = out/f'{tag}.yml'
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    log = out/f'{tag}.log'
    telemetry = []
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, 'tools/train.py', '-c', str(config)],
                                   cwd=root/'PaddleOCR', stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            query = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw,clocks.gr,clocks.mem,temperature.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
            telemetry.append(dict(time=time.time(), gpu=query.stdout.strip()))
            time.sleep(2)
    text = log.read_text(errors='replace')
    rates = [float(x) for x in re.findall(r'ips: ([0-9.]+) samples/s', text)]
    memory = [int(x) for x in re.findall(r'max_mem_reserved: (\d+) MB', text)]
    valid = (process.returncode == 0 and len(rates) >= 2 and memory and max(memory) <= 14000
             and not re.search(r'\b(?:nan|inf)\b|not in loaded params|shape.*not matched', text, re.I))
    result = dict(batch=batch, workers=workers, returncode=process.returncode,
                  accepted=bool(valid), samples_per_second=statistics.median(rates[1:]) if len(rates)>1 else None,
                  peak_reserved_mb=max(memory) if memory else None)
    (out/f'{tag}_telemetry.json').write_text(json.dumps(telemetry, indent=2))
    results.append(result)
    (out/'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
    return result


for batch in (32, 64, 128, 256, 512):
    result = probe(batch, 4)
    if not result['accepted']:
        break
eligible = [r for r in results if r['accepted']]
if not eligible:
    raise RuntimeError('No valid resource probe. Inspect logs before proceeding.')
best = max(eligible, key=lambda r: r['samples_per_second'])
for workers in (8, 12):
    if workers >= (os.cpu_count() or 1):
        continue
    probe(best['batch'], workers)
best = max((r for r in results if r['accepted']), key=lambda r: r['samples_per_second'])
base['Train']['sampler']['first_bs'] = best['batch']
base['Train']['loader'].update(batch_size_per_card=best['batch'], num_workers=best['workers'])
base['Eval']['loader'].update(batch_size_per_card=min(best['batch'], 128), num_workers=min(best['workers'], 4))
base['Global']['eval_batch_epoch'] = 1
(root/'configs/selected_resources.yml').write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False))
(out/'selected.json').write_text(json.dumps(best, indent=2))
print('Selected resource settings:', best, flush=True)

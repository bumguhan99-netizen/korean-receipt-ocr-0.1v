import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from common import MODELS, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--refresh', action='store_true', help='Explicitly select the current upstream commits')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    records = []
    for name in MODELS:
        repo = f'PaddlePaddle/{name}'
        folder = args.model_root / name
        lock = folder/'download_manifest.json'
        previous = json.loads(lock.read_text(encoding='utf-8')) if lock.exists() and not args.refresh else None
        if previous:
            assert previous['model'] == name and previous['repository'] == repo
        commit = previous['revision'] if previous else HfApi().model_info(repo).sha
        snapshot_download(repo, revision=commit, local_dir=folder,
                          allow_patterns=['inference.json', 'inference.pdiparams', 'inference.yml'])
        files = {f:sha256(folder/f) for f in ('inference.json', 'inference.pdiparams', 'inference.yml')}
        if previous:
            assert files == previous['files'], 'Pinned download differs from recorded hashes'
        records.append(dict(model=name, repository=repo, revision=commit, files=files))
        write_json(folder/'download_manifest.json', records[-1])
        print(f'{name}: {commit}, weights {files["inference.pdiparams"]}', flush=True)
    assert len({r['files']['inference.pdiparams'] for r in records}) == 2, 'Identical model weights'
    write_json(args.model_root/'download_manifest.json', records)


if __name__ == '__main__':
    main()

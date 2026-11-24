"""Pin official pretrained recognition reference and fallback detector by HF revision."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
FILES = ('inference.json', 'inference.pdiparams', 'inference.yml')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    session = requests.Session()
    for name in ('korean_PP-OCRv5_mobile_rec', 'PP-OCRv5_mobile_det'):
        folder = ROOT / 'models' / name
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / 'download_manifest.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding='utf-8'))
            assert previous['files'] == {f: digest(folder/f) for f in FILES}
            print(f'Already pinned: {name}', flush=True)
            continue
        response = session.get(f'https://huggingface.co/api/models/PaddlePaddle/{name}', timeout=60)
        response.raise_for_status()
        revision = response.json()['sha']
        for filename in FILES:
            destination = folder / filename
            if destination.exists():
                raise FileExistsError(f'Unrecorded model file: {destination}')
            url = f'https://huggingface.co/PaddlePaddle/{name}/resolve/{revision}/{filename}'
            with session.get(url, timeout=(30, 180), stream=True) as response:
                response.raise_for_status()
                temporary = destination.with_suffix(destination.suffix+'.partial')
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
                temporary.rename(destination)
        record = dict(repository=f'PaddlePaddle/{name}', revision=revision,
                      files={f:digest(folder/f) for f in FILES},
                      role='official pretrained recognition reference' if 'rec' in name else
                      'new fixed detector candidate; original detector was not found')
        with manifest.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

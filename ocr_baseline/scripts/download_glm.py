"""Download official GLM-OCR, pin its commit and SHA-256 every local asset."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from common import sha256, write_json

REPO = 'zai-org/GLM-OCR'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=Path('models/GLM-OCR'))
    args = parser.parse_args()
    manifest = args.model_dir/'download_manifest.json'
    previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
    revision = previous['revision'] if previous else HfApi().model_info(REPO).sha
    snapshot_download(REPO, revision=revision, local_dir=args.model_dir,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', 'README.md', 'LICENSE*'])
    files = {p.relative_to(args.model_dir).as_posix():sha256(p) for p in sorted(args.model_dir.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(args.model_dir).parts and p != manifest}
    assert any(name.endswith('.safetensors') for name in files), 'Missing model weights'
    assert 'config.json' in files and 'preprocessor_config.json' in files
    if previous:
        assert files == previous['files'], 'Pinned model files changed'
    write_json(manifest, dict(repository=REPO, revision=revision, files=files))
    print(f'Downloaded {REPO} revision {revision}; hashed {len(files)} files', flush=True)


if __name__ == '__main__':
    main()

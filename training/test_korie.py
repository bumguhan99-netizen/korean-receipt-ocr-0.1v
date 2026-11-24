"""Synthetic-only tests: never read/convert the real dataset or run OCR."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types
import yaml
from PIL import Image
import korie


class PipelineTests(unittest.TestCase):
    def test_preserve_splits_raw_text_and_report_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'original'
            for split in ('train', 'val', 'test'):
                (data / split).mkdir(parents=True)
                for i, text in enumerate([' 1,000-가 ', 'X', '11111111111111111111111111']):
                    path = data / split / f'IMG{i:05d}_Total.jpg'
                    Image.new('RGB', (80, 20)).save(path)
                    path.with_suffix('.txt').write_text(text+'\n', encoding='utf-8')
            chars = list('10,-가')
            dictionary = root / 'dict.txt'
            dictionary.write_text('\n'.join(chars)+'\n', encoding='utf-8')
            reference = root / 'inference.yml'
            reference.write_text(yaml.safe_dump({'PostProcess': {'character_dict': chars}}), encoding='utf-8')
            before = {str(p): korie.digest(p) for p in data.rglob('*') if p.is_file()}
            args = argparse.Namespace(data_root=data, dictionary=dictionary, reference=reference,
                                      output=root/'derived', max_length=25, seed=42)
            korie.prepare(args)
            report = korie.read(args.output/'validation_report.json')
            for split in ('train', 'val', 'test'):
                self.assertEqual(report['counts'][split]['usable'], 1)
                self.assertEqual(report['counts'][split]['held'], 2)
                self.assertEqual((args.output/f'{split}_list.txt').read_text(encoding='utf-8'),
                                 f'{split}/IMG00000_Total.jpg\t 1,000-가 \n')
            rows = [json.loads(s) for s in (args.output/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 9)  # Evaluation keeps OOV/long samples.
            self.assertEqual(before, {str(p): korie.digest(p) for p in data.rglob('*') if p.is_file()})
            korie.verify(argparse.Namespace(data=args.output))
            (args.output/'train_list.txt').write_text('tampered', encoding='utf-8')
            with self.assertRaises(ValueError):
                korie.verify(argparse.Namespace(data=args.output))
            with self.assertRaises(FileExistsError):
                korie.prepare(args)

    def test_metrics_preserve_punctuation_and_nfc(self):
        self.assertEqual(korie.measure('1,000', '1000')['exact_match'], 0)
        self.assertEqual(korie.measure(' 가 ', '가')['char_edits'], 2)
        self.assertEqual(korie.measure('가', '\u1100\u1161')['exact_match'], 1)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                korie.inside(root, '../outside.jpg')

    def test_freeze_rejects_incomplete_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for status, split in [('incomplete', 'val'), ('completed', 'test')]:
                korie.dump(root/'run_info.json', dict(status=status, split=split))
                with self.assertRaises(ValueError):
                    korie.freeze(argparse.Namespace(run=root, output=root/'frozen.json'))

    def test_paired_evaluation_keeps_oov_and_locks_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ('val', 'test'):
                image = root / f'{split}.jpg'
                label = root / f'{split}.txt'
                Image.new('RGB', (20, 10)).save(image)
                label.write_text('1,000X', encoding='utf-8')
                records.append(dict(source_split=split, image_path=image.name, label_path=label.name,
                                    image_sha256=korie.digest(image), label_sha256=korie.digest(label),
                                    gt_raw='1,000X', field='Total', issues=['oov'], trainable=False))
            manifest = root/'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            for name in ('baseline', 'candidate'):
                folder = root/name
                folder.mkdir()
                (folder/'inference.json').write_text('{}')
                (folder/'inference.pdiparams').write_bytes(name.encode())
                (folder/'inference.yml').write_text('PreProcess: {}\nPostProcess: {}\n')

            class FakeRecognition:
                def __init__(self, **kwargs):
                    self.text = '1,000X' if Path(kwargs['model_dir']).name == 'candidate' else '1000X'

                def predict(self, **kwargs):
                    return [types.SimpleNamespace(json={'res': {'rec_text': self.text, 'rec_score': 1.0}})]

            args = argparse.Namespace(manifest=manifest, baseline=root/'baseline', candidate=root/'candidate',
                                      data_root=root, output=root/'val_result', split='val', device='cpu',
                                      amount_fields=['Total'], frozen=None)
            with patch.dict('sys.modules', {'paddleocr': types.SimpleNamespace(TextRecognition=FakeRecognition)}):
                korie.evaluate(args)
                result = korie.read(args.output/'summary.json')
                self.assertEqual(result['candidate']['n'], 1)
                self.assertEqual(result['candidate']['exact_match_pct'], 100)
                self.assertEqual(result['baseline']['exact_match_pct'], 0)
                frozen = root/'frozen.json'
                korie.freeze(argparse.Namespace(run=args.output, output=frozen))
                args.split, args.frozen, args.output = 'test', frozen, root/'test_result'
                korie.evaluate(args)
                args.output = root/'second_test'
                with self.assertRaises(FileExistsError):
                    korie.evaluate(args)


if __name__ == '__main__':
    unittest.main()

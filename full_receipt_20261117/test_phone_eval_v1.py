import argparse
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from phone_receipt_eval_v1 import audit,subset,measure

class PhoneEvaluation(unittest.TestCase):
    def test_audit_preserves_verified_text_and_excludes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'data'
            for folder in ('labels','crops','aligned_images','verified_labels'):
                (data/folder).mkdir(parents=True)
            Image.new('RGB',(30,20)).save(data/'aligned_images/r001.png')
            Image.new('RGB',(20,10)).save(data/'crops/r001_001.png')
            text=' 1,000-원 '
            (data/'verified_labels/r001_001.txt').write_text(text,encoding='utf-8')
            receipt=dict(id='r001',source_sha256='source',lines=[dict(id='r001_001',status='verified',crop='crops/r001_001.png',coordinate_image='aligned_images/r001.png',bbox=[0,0,20,10],text=text),dict(id='r001_002',status='excluded')])
            (data/'annotations.json').write_text(json.dumps([receipt]))
            (data/'labels/r001.json').write_text(json.dumps(receipt))
            output=root/'output'
            audit(argparse.Namespace(data=data,output=output))
            row=json.loads((output/'manifest.jsonl').read_text())
            self.assertEqual(row['text'],text)
            self.assertEqual(json.loads((output/'audit.json').read_text())['verified'],1)
    def test_same_source_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'groups.json'
            path.write_text(json.dumps({'a':{'group_id':'x','split':'train'},'b':{'group_id':'y','split':'val'}}))
            with self.assertRaises(ValueError):
                subset([{'receipt_id':'a','source_sha256':'same'},{'receipt_id':'b','source_sha256':'same'}],path,'val')
    def test_missing_region_counts_all_characters(self):
        result=measure('1,000','')
        self.assertEqual(result['char_edits'],5)
    def test_val_requires_group_definition(self):
        with self.assertRaises(ValueError):
            subset([],None,'val')

if __name__=='__main__':
    unittest.main()

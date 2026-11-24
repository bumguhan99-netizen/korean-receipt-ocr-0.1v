import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import train_round2_v2 as subject

class Guards(unittest.TestCase):
    def run_case(self, stage, frozen=False, changed=False, checkpoint=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'weights.pdparams'
            weights.write_bytes(b'original')
            (root/'inputs.json').write_text(json.dumps(dict(repo=str(root),data_root=str(root),weights=str(weights),weights_sha256=hashlib.sha256(b'original').hexdigest())))
            (root/'smoke_passed.json').write_text('{}')
            if frozen:
                (root/'frozen_test.json').write_text('{}')
            if changed:
                weights.write_bytes(b'changed')
            argv = ['guard',stage,'--run',str(root),'--repo',str(root),'--data-root',str(root)]
            if checkpoint:
                argv += ['--checkpoint',str(root/'foreign/latest')]
            with patch('sys.argv',argv), patch.object(subject,'run') as execute:
                with self.assertRaises(SystemExit):
                    subject.main()
                execute.assert_not_called()
    def test_freeze_blocks_training(self):
        self.run_case('train',frozen=True)
    def test_changed_weights_block_training(self):
        self.run_case('train',changed=True)
    def test_foreign_resume_is_rejected(self):
        self.run_case('resume',checkpoint=True)

if __name__ == '__main__':
    unittest.main()

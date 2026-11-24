import unittest
from partial_matching_v1 import match

class Matching(unittest.TestCase):
    def test_duplicate_detection_not_double_counted(self):
        result=match([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]])
        self.assertEqual(len(result['pairs']),1)
        self.assertEqual(len(result['unscored_predictions']),1)
    def test_missing_gt_is_explicit(self):
        result=match([[0,0,10,10],[20,20,30,30]],[[0,0,10,10]])
        self.assertEqual(result['missing_gt'],[1])
    def test_empty_predictions(self):
        self.assertEqual(match([[0,0,10,10]],[])['missing_gt'],[0])
    def test_unmatched_prediction_is_unscored(self):
        self.assertEqual(match([],[[0,0,10,10]])['unscored_predictions'],[0])

if __name__=='__main__':
    unittest.main()

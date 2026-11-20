import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import _video_split_components
from datasets.common import CookingSample, ListDataset


def component(source):
    samples = [CookingSample('', '', ['answer'], [1.0], ['frame.jpg'], [0.0], source,
                              {'video_id': str(i)}) for i in range(20)]
    return ListDataset(samples, temporal_jitter={'enabled': True},
                       sample_weights=[1.0] * len(samples))


class VideoSplitTest(unittest.TestCase):
    def test_raw_videos_are_disjoint_across_components_and_repeatable(self):
        components = [component('instruction'), component('counterfactual')]
        splits = _video_split_components(components, [1, 2], [.8, .1, .1], 320, 42)
        train = splits['train']
        train_ids = {s.metadata['video_id'] for d in train.datasets for s in d.samples}
        val_ids = {splits['val'][i].metadata['video_id'] for i in range(len(splits['val']))}
        test_ids = {splits['test'][i].metadata['video_id'] for i in range(len(splits['test']))}
        self.assertFalse(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        self.assertEqual(len(train_ids | val_ids | test_ids), 20)
        self.assertEqual(len(train), 320)
        self.assertEqual(len(splits['val']), 4)
        self.assertNotIn('temporal_jitter', splits['val'][0].metadata)
        self.assertTrue(all(d.sample_weights is None for d in splits['val'].datasets))
        again = _video_split_components(components, [1, 2], [.8, .1, .1], 320, 42)
        self.assertEqual(train_ids, {s.metadata['video_id'] for d in again['train'].datasets for s in d.samples})

    def test_too_few_videos_fail(self):
        with self.assertRaises(ValueError):
            _video_split_components([ListDataset(component('x').samples[:2])], [1], [.8,.1,.1], 32, 42)


if __name__ == '__main__':
    unittest.main()

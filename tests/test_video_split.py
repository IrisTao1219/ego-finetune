import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import _video_split_components
from datasets.common import CookingSample, ListDataset, WeightedMixtureDataset


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

    def test_epoch_draws_change_and_are_repeatable(self):
        mix = WeightedMixtureDataset([component('a'), component('b')], [1, 2], 320, 42)
        first = [mix.sample_location(i) for i in range(len(mix))]
        mix.set_epoch(1)
        second = [mix.sample_location(i) for i in range(len(mix))]
        self.assertNotEqual(first, second)
        mix.set_epoch(0)
        self.assertEqual(first, [mix.sample_location(i) for i in range(len(mix))])
        replica = WeightedMixtureDataset([component('a'), component('b')], [1, 2], 320, 42)
        replica.set_epoch(1)
        self.assertEqual(second, [replica.sample_location(i) for i in range(len(replica))])

    def test_component_weights_applied_once(self):
        source = component('a')
        source.sample_weights = [0] * 19 + [1]
        mix = WeightedMixtureDataset([source], [1], 100, 42)
        self.assertTrue(all(mix[i].metadata['video_id'] == '19' for i in range(100)))
        self.assertIn('temporal_jitter', mix[0].metadata)

    def test_too_few_videos_fail(self):
        with self.assertRaises(ValueError):
            _video_split_components([ListDataset(component('x').samples[:2])], [1], [.8,.1,.1], 32, 42)


if __name__ == '__main__':
    unittest.main()

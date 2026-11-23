"""Retention regression tests; no CUDA or DeepSpeed required."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpointing import retain_epoch_checkpoints


class RetentionTests(unittest.TestCase):
    def save(self, root, epoch, loss, k):
        tag = f"epoch-{epoch}"
        (root / tag).mkdir()
        (root / tag / "rank0.pt").write_text("model shard")
        (root / tag / "rank1.pt").write_text("optimizer shard")
        (root / "latest").write_text(tag)
        retain_epoch_checkpoints(root, tag, loss, k)

    def test_actual_run_keeps_best_three_and_latest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for epoch, loss in enumerate([9.6455, 7.3107, 6.5966, 6.9539, 7.9652,
                                          8.7033, 12.6893, 17.6428, 21.2729, 23.7960], 1):
                self.save(root, epoch, loss, 3)
            self.assertEqual({p.name for p in root.glob('epoch-*')},
                             {'epoch-2', 'epoch-3', 'epoch-4', 'epoch-10'})
            self.assertEqual((root / 'best').read_text().strip(), 'epoch-3')
            self.assertEqual((root / 'latest').read_text(), 'epoch-10')
            # Each invocation reloads the index: simulates retention after restart.
            self.save(root, 11, 5.0, 3)
            self.assertEqual({p.name for p in root.glob('epoch-*')},
                             {'epoch-3', 'epoch-4', 'epoch-11'})
            self.assertEqual((root / 'best').read_text().strip(), 'epoch-11')

    def test_keep_all_zero_and_legacy(self):
        for k, expected in [(-1, {'epoch-1', 'epoch-2'}), (0, {'epoch-2'})]:
            with self.subTest(k=k), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / 'epoch-99').mkdir()  # Unknown legacy files must survive.
                (root / 'step-1').mkdir()
                self.save(root, 1, 1.0, k)
                self.save(root, 2, 2.0, k)
                self.assertEqual({p.name for p in root.glob('epoch-*')}, expected | {'epoch-99'})
                self.assertTrue((root / 'step-1').exists())
                if k == 0:
                    self.assertFalse((root / 'best').exists())

    def test_nonfinite_loss_does_not_displace_best(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.save(root, 1, 1.0, 1)
            self.save(root, 2, float('nan'), 1)
            self.assertEqual((root / 'best').read_text().strip(), 'epoch-1')
            self.assertTrue((root / 'epoch-2').exists())


if __name__ == '__main__':
    unittest.main()

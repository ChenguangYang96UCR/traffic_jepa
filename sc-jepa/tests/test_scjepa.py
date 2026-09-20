from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from data import TrafficWindows
from model import Config, SCJEPA


class TrafficTests(unittest.TestCase):
    def test_adaptation_transfer_cli(self):
        from train import verify_split
        from split_fremont_adaptation import create_split
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, count in [('train', 4), ('val', 4), ('test', 20)]:
                samples = np.array([dict(x_data=rng.normal(size=(12, 3, 1)).astype('float32'),
                                         y_data=rng.normal(size=(12, 3, 1)).astype('float32'))
                                    for _ in range(count)], dtype=object)
                np.save(root / f'incident_{split}.npy', samples)
            adaptation = root / 'adapt'
            create_split(root / 'incident_test.npy', adaptation, acknowledge_stored_order=True)
            verify_split(adaptation, root)
            script = str(Path(__file__).resolve().parents[1] / 'train.py')
            common = [sys.executable, script, '--data-dir', str(root), '--nodes', '3',
                      '--dim', '16', '--codes', '8', '--heads', '2', '--layers', '1',
                      '--encoder-hidden', '16', '--predictor-dim', '16',
                      '--predictor-heads', '2', '--predictor-layers', '1',
                      '--batch-size', '2', '--pretrain-epochs', '1', '--finetune-epochs', '1',
                      '--device', 'cpu']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       KMP_USE_SHM='0')
            def run(extra):
                result = subprocess.run(common + extra, capture_output=True, text=True, env=env, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            base, new = root / 'base', root / 'new'
            run(['--mode', 'pretrain', '--forecast-weight', '1', '--run-dir', str(base)])
            old_metrics = (base / 'pretrain/metrics.json').read_bytes()
            run(['--mode', 'transfer', '--checkpoint', str(base / 'pretrain/checkpoint.pt'),
                 '--adaptation-dir', str(adaptation), '--run-dir', str(new)])
            self.assertEqual((base / 'pretrain/metrics.json').read_bytes(), old_metrics)
            self.assertTrue((new / 'baseline/metrics.json').exists())
            for stage in ('probe', 'full'):
                metadata = json.loads((new / stage / 'run.json').read_text())
                self.assertEqual(metadata['split_sizes'], dict(train=12, val=4, test=4))
                ckpt = torch.load(new / stage / 'checkpoint.pt', weights_only=True)
                self.assertEqual(ckpt['data_contract']['root'], str(adaptation.resolve()))
                before = (new / stage / 'metrics.json').read_bytes()
                run(['--mode', 'eval', '--checkpoint', str(new / stage / 'checkpoint.pt'),
                     '--adaptation-dir', str(adaptation), '--run-dir', str(new)])
                self.assertEqual((new / stage / 'metrics.json').read_bytes(), before)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ema_freezing_and_round_trip(self):
        torch.manual_seed(2026)
        c = Config(dim=16, codes=8, heads=2, layers=1, encoder_hidden=16,
                   predictor_dim=16, predictor_heads=2, predictor_layers=1, dropout=.1)
        m = SCJEPA(c)
        m.configure('pretrain')
        m.train()
        x, y = torch.randn(2, 12, 3), torch.randn(2, 12, 3)
        self.assertEqual(m(x).shape, y.shape)
        target_before = {n: p.clone() for n, p in m.target_encoder.named_parameters()}
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
        loss, terms = m.objective(x, y)
        self.assertEqual(set(terms), {'fine_kl'})
        self.assertFalse(hasattr(m, 'reconstructor'))
        self.assertFalse(hasattr(m, 'coarse_predictor'))
        self.assertTrue(all(torch.isfinite(v) for v in terms.values()))
        self.assertEqual(set(m.objective(x, y, forecast_weight=1)[1]),
                         {'fine_kl', 'forecast_mse'})
        loss.backward()
        self.assertTrue(all(p.grad is None for p in m.target_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in m.forecast_head.parameters()))
        opt.step()
        m.update_ema()
        for (n, target), online in zip(m.target_encoder.named_parameters(), m.encoder.parameters()):
            torch.testing.assert_close(target, target_before[n] * c.ema + online * (1 - c.ema))
        for stage in ('probe', 'full'):
            m.configure(stage)
            m.train()
            frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
            if stage == 'probe':
                self.assertEqual({n for n, p in m.named_parameters() if p.requires_grad},
                                 {'forecast_head.weight', 'forecast_head.bias'})
                torch.testing.assert_close(m(x), m(x), atol=0, rtol=0)
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
            opt.zero_grad()
            (m(x) - y).square().mean().backward()
            opt.step()
            for n, p in m.named_parameters():
                if n in frozen:
                    torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)
        m.eval()
        restored = SCJEPA(Config(**asdict(c)))
        restored.load_state_dict(m.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(m(x), restored(x), atol=0, rtol=0)
        # Constant inputs remain numerically valid; the retained upstream fine
        # predictor requires equal context and target patch counts.
        other = SCJEPA(Config(dim=16, codes=8, heads=2, layers=1,
                             encoder_hidden=16, predictor_dim=16, predictor_heads=2,
                             predictor_layers=1))
        self.assertTrue(torch.isfinite(other.objective(torch.ones(2, 12, 3), torch.ones(2, 12, 3))[0]))
        with self.assertRaises(ValueError):
            Config(seq_len=12, pred_len=6)

    def test_dataset_and_cli(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                samples = []
                for i in range(4):
                    x = rng.normal(size=(12, 3, 3)).astype('float32')
                    y = rng.normal(size=(12, 3, 3)).astype('float32')
                    samples.append(dict(x_data=x, y_data=y, incident_features='ignored'))
                np.save(root / f'incident_{split}.npy', np.array(samples, dtype=object))
            ds = TrafficWindows(root, 'test', nodes=3)
            np.testing.assert_array_equal(ds[1][0], samples[1]['x_data'][:, :, 0])
            np.testing.assert_array_equal(ds[1][1], samples[1]['y_data'][:, :, 0])
            with self.assertRaises(ValueError):
                TrafficWindows(root, 'test', nodes=4)
            script = str(Path(__file__).resolve().parents[1] / 'train.py')
            common = [sys.executable, script, '--data-dir', str(root), '--nodes', '3',
                      '--dim', '16', '--codes', '8', '--heads', '2', '--layers', '1',
                      '--encoder-hidden', '16', '--predictor-dim', '16',
                      '--predictor-heads', '2', '--predictor-layers', '1',
                      '--batch-size', '2', '--pretrain-epochs', '1', '--finetune-epochs', '1',
                      '--device', 'cpu']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       KMP_USE_SHM='0')

            def run(extra):
                result = subprocess.run(common + extra, capture_output=True, text=True, env=env, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            run_dir = root / 'ssl'
            run(['--run-dir', str(run_dir)])
            self.assertFalse((run_dir / 'pretrain/metrics.json').exists())
            for stage in ('probe', 'full'):
                metrics = json.loads((run_dir / stage / 'metrics.json').read_text())
                self.assertAlmostEqual(metrics['RMSE'] ** 2, metrics['MSE'])
                run(['--mode', 'eval', '--checkpoint', str(run_dir / stage / 'checkpoint.pt')])
                after = json.loads((run_dir / stage / 'metrics.json').read_text())
                self.assertEqual(metrics, after)
            p = torch.load(run_dir / 'probe/checkpoint.pt', weights_only=True)
            f = torch.load(run_dir / 'full/checkpoint.pt', weights_only=True)
            self.assertEqual(p['source_sha256'], f['source_sha256'])
            result = subprocess.run(common + ['--run-dir', str(run_dir)], capture_output=True,
                                    text=True, env=env, timeout=90)
            self.assertNotEqual(result.returncode, 0)  # refuses overwrite
            run(['--mode', 'pretrain', '--forecast-weight', '1', '--run-dir', str(root / 'joint')])
            self.assertTrue((root / 'joint/pretrain/metrics.json').exists())


if __name__ == '__main__':
    unittest.main()

"""Small CPU integration run using generated traffic, never the real datasets."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class AdaptationCLI(unittest.TestCase):
    def test_pretrain_baseline_and_full(self):
        code_root = Path(__file__).resolve().parents[1]
        # These optional imports belong to unrelated attention implementations.
        # If absent, fail if anything attempts to call them on the tested path.
        launcher = '''
import sys, types, runpy, torch
torch.set_num_threads(1)
for name, attr in [('reformer_pytorch', 'LSHSelfAttention'), ('einops', 'rearrange')]:
    try: __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        def unused(*a, **k): raise AssertionError('Unused optional path invoked')
        setattr(module, attr, unused)
        sys.modules[name] = module
script = sys.argv.pop(1)
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
runpy.run_path(script, run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(8)
            for split, count in [('train', 4), ('val', 2), ('test', 10)]:
                samples = np.array([dict(x_data=rng.normal(size=(12, 3, 3)).astype('float32'),
                                         y_data=rng.normal(size=(12, 3, 3)).astype('float32'))
                                    for _ in range(count)], dtype=object)
                np.save(root / f'incident_{split}.npy', samples)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_USE_SHM='0',
                       MPLCONFIGDIR=str(root / 'mpl'))
            split = subprocess.run([sys.executable, str(code_root.parent / 'Preprocess/split_fremont_adaptation.py'),
                '--source', str(root / 'incident_test.npy'), '--output', str(root / 'adapt'),
                '--acknowledge-stored-order'], capture_output=True, text=True, env=env)
            self.assertEqual(split.returncode, 0, split.stderr)
            common = [sys.executable, '-c', launcher, str(code_root / 'run.py'),
                '--is_training', '1', '--model_id', 'tiny', '--model', 'TopoJEPA',
                '--data', 'Fremont', '--root_path', str(root),
                '--fremont_adaptation_root', str(root / 'adapt'),
                '--checkpoints', str(root / 'checkpoints'), '--seq_len', '12', '--pred_len', '12',
                '--label_len', '6', '--enc_in', '3', '--dec_in', '3', '--c_out', '3',
                '--d_model', '16', '--n_heads', '2', '--e_layers', '2', '--d_ff', '32',
                '--batch_size', '2', '--num_workers', '0', '--train_epochs', '1',
                '--model_variant', 'jepa', '--topo_weight', '0', '--text_weight', '0']
            def run(extra):
                result = subprocess.run(common + extra, cwd=root, capture_output=True, text=True,
                                        env=env, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            output = run(['--training_stage', 'pretrain', '--experiment_tag', 'tiny_pre', '--jepa_weight', '.1'])
            self.assertIn('train 4', output)
            checkpoint = root / 'checkpoints/tiny_pre_0/checkpoint.pth'
            output = run(['--training_stage', 'finetune', '--experiment_tag', 'tiny_full',
                         '--finetune_strategy', 'full', '--pretrained_checkpoint', str(checkpoint), '--jepa_weight', '0'])
            self.assertIn('train 6', output)
            self.assertIn('val 2', output)
            self.assertEqual(output.count('test 2'), 1)
            run(['--is_training', '0', '--training_stage', 'pretrain', '--experiment_tag', 'tiny_base',
                 '--eval_checkpoint', str(checkpoint), '--jepa_weight', '.1'])
            self.assertTrue((root / 'results/tiny_base_0/evaluation.json').exists())

            objective_args = [
                '--pretrain_objective', 'forecast_reconstruction',
                '--pretrain_mask_strategy', 'temporal',
                '--pretrain_mask_ratio', '.25',
                '--reconstruction_weight', '.1']
            output = run(['--training_stage', 'pretrain',
                          '--experiment_tag', 'tiny_pre_reconstruction',
                          '--jepa_weight', '.1'] + objective_args)
            self.assertIn('Masked reconstruction MSE', output)
            reconstruction_checkpoint = (
                root / 'checkpoints/tiny_pre_reconstruction_0/checkpoint.pth')
            output = run([
                '--training_stage', 'finetune',
                '--experiment_tag', 'tiny_full_reconstruction',
                '--finetune_strategy', 'full',
                '--pretrained_checkpoint', str(reconstruction_checkpoint),
                '--jepa_weight', '0'] + objective_args)
            self.assertIn('Loaded JEPA checkpoint', output)


if __name__ == '__main__':
    unittest.main()

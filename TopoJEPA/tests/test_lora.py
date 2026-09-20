"""Run from TopoJEPA: python -m unittest discover -s tests -p test_lora.py"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.TopoJEPA import Model
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as Exp


class LoRATest(unittest.TestCase):
    def test_training_and_evaluation_checkpoint_lifecycle(self):
        torch.manual_seed(7)
        args = SimpleNamespace(
            seq_len=12, pred_len=12, output_attention=False, use_norm=True,
            model_variant='jepa', jepa_weight=0., topo_weight=0., text_weight=0.,
            alignment_weight=0., use_gnn=False, incident=False, embed='timeF',
            freq='t', dropout=.1, class_strategy='projection', factor=1,
            d_model=16, n_heads=2, d_ff=32, activation='gelu', e_layers=2,
            learning_rate=.001, training_stage='finetune', finetune_strategy='lora',
            lora_rank=4, lora_alpha=8., lora_dropout=0., lora_lr_scale=1.,
            encoder_lr_scale=.1, partial_unfreeze_layers=1, is_training=1,
            model='TopoJEPA', use_multi_gpu=False, use_gpu=False)
        exp = Exp.__new__(Exp)
        exp.args, exp.model, exp.device = args, Model(args), torch.device('cpu')
        x, y = torch.randn(2, 12, 93), torch.randn(2, 12, 93)
        exp.model.eval()
        original = exp.model(x, None, None, None).detach()
        with tempfile.TemporaryDirectory() as tmp:
            args.pretrained_checkpoint = str(Path(tmp) / 'pretrain.pth')
            torch.save(exp.model.state_dict(), args.pretrained_checkpoint)
            exp._load_pretrained_for_finetuning()
            exp.model.eval()
            torch.testing.assert_close(exp.model(x, None, None, None), original,
                                       rtol=0, atol=0)
            frozen = {n: p.detach().clone() for n, p in exp.model.named_parameters()
                      if not p.requires_grad}
            trainable = {n: p.detach().clone() for n, p in exp.model.named_parameters()
                         if p.requires_grad}
            self.assertTrue(all('lora_' in n or n.startswith('projector.')
                                for n in trainable))
            self.assertEqual(sum('lora_' in n for n in trainable), 8)
            optimizer = exp._select_optimizer()
            self.assertTrue(all(g['lr'] == args.learning_rate for g in optimizer.param_groups))
            exp.model.train()
            self.assertFalse(exp.model.encoder.training)
            self.assertFalse(exp.model.target_encoder.training)
            for _ in range(2):
                optimizer.zero_grad()
                (exp.model(x, None, None, None) - y).square().mean().backward()
                optimizer.step()
            for n, p in exp.model.named_parameters():
                if n in frozen:
                    torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
                else:
                    self.assertFalse(torch.equal(p, trainable[n]), n)
            exp.model.eval()
            expected = exp.model(x, None, None, None).detach()
            path = Path(tmp) / 'lora.pth'
            torch.save(exp.model.state_dict(), path)
            args.is_training = 0
            import model.TopoJEPA as model_module
            exp.model_dict = {'TopoJEPA': model_module}
            restored = exp._build_model()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            restored.eval()
            torch.testing.assert_close(restored(x, None, None, None), expected,
                                       rtol=0, atol=0)
            args.lora_rank = 2
            with self.assertRaises(RuntimeError):
                exp._build_model().load_state_dict(torch.load(path, weights_only=True))


if __name__ == '__main__':
    unittest.main()

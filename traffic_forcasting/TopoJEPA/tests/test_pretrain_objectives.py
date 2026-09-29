"""JEPA-fixed auxiliary pretraining objective tests."""
import sys
import types
from types import SimpleNamespace
import unittest

import torch

try:
    import reformer_pytorch  # noqa: F401
except ImportError:
    module = types.ModuleType('reformer_pytorch')
    module.LSHSelfAttention = object
    sys.modules['reformer_pytorch'] = module
try:
    import einops  # noqa: F401
except ImportError:
    module = types.ModuleType('einops')
    module.rearrange = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('Unused optional path invoked'))
    sys.modules['einops'] = module

from traffic_forcasting.TopoJEPA.model.TopoJEPA import Model
from traffic_forcasting.TopoJEPA.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as Exp


def model_args(objective='forecast_reconstruction'):
    return SimpleNamespace(
        seq_len=12, pred_len=12, output_attention=False, use_norm=True,
        model_variant='jepa', jepa_weight=.1, topo_weight=0., text_weight=0.,
        alignment_weight=0., use_gnn=False, incident=False, embed='timeF',
        freq='t', dropout=.1, class_strategy='projection', factor=1,
        d_model=16, n_heads=2, d_ff=32, activation='gelu', e_layers=1,
        forecast_mask=False, pretrain_objective=objective)


class PretrainObjectiveTests(unittest.TestCase):
    def test_mask_strategies_have_expected_shape(self):
        x = torch.randn(4, 12, 7)
        for strategy in ('random', 'temporal', 'sensor', 'block'):
            with self.subTest(strategy=strategy):
                exp = Exp.__new__(Exp)
                exp.args = SimpleNamespace(
                    pretrain_mask_strategy=strategy,
                    pretrain_mask_ratio=.25)
                generator = torch.Generator().manual_seed(13)
                mask = exp._make_pretrain_input_mask(x, generator)
                self.assertEqual(mask.shape, x.shape)
                self.assertEqual(mask.dtype, torch.bool)
                self.assertGreater(mask.sum().item(), 0)
                self.assertLess(mask.sum().item(), mask.numel())

    def test_reconstruction_auxiliary_and_freezing(self):
        torch.manual_seed(7)
        args = model_args()
        model = Model(args)
        model.configure_pretraining()
        x, target = torch.randn(2, 12, 7), torch.randn(2, 12, 7)
        input_mask = torch.zeros_like(x, dtype=torch.bool)
        input_mask[:, 3:6] = True
        outputs = model.forward_with_jepa(
            x, None, None, None, target, None,
            input_mask=input_mask, return_pretrain_aux=True)
        forecast, jepa_loss, aux = outputs[0], outputs[1], outputs[-1]
        self.assertEqual(forecast.shape, target.shape)
        self.assertEqual(aux['reconstruction'].shape, x.shape)
        self.assertIsNone(aux['forecast_mask'])

        exp = Exp.__new__(Exp)
        normalized = model._normalized_view(x)
        reconstruction_loss = exp._masked_reconstruction_loss(
            aux['reconstruction'], normalized, input_mask)
        loss = jepa_loss + reconstruction_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None
                            for p in model.reconstruction_head.parameters()))

        model.configure_finetuning('full')
        self.assertTrue(all(not p.requires_grad
                            for p in model.reconstruction_head.parameters()))

    def test_jepa_only_does_not_train_forecast_head(self):
        model = Model(model_args('jepa'))
        model.configure_pretraining()
        self.assertTrue(all(not p.requires_grad for p in model.projector.parameters()))
        self.assertFalse(hasattr(model, 'reconstruction_head'))


if __name__ == '__main__':
    unittest.main()

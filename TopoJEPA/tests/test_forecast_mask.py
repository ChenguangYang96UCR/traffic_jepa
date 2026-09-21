"""Learned time/sensor forecast-loss mask tests."""
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

from model.TopoJEPA import Model
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as Exp


class ForecastMaskTests(unittest.TestCase):
    def test_mask_shapes_loss_and_finetune_freezing(self):
        torch.manual_seed(5)
        args = SimpleNamespace(
            seq_len=12, pred_len=12, output_attention=False, use_norm=True,
            model_variant='jepa', jepa_weight=.1, topo_weight=0., text_weight=0.,
            alignment_weight=0., use_gnn=False, incident=False, embed='timeF',
            freq='t', dropout=.1, class_strategy='projection', factor=1,
            d_model=16, n_heads=2, d_ff=32, activation='gelu', e_layers=1,
            forecast_mask=True, forecast_mask_floor=.1,
            forecast_mask_target=.5, forecast_mask_budget_weight=.1,
            forecast_mask_entropy_weight=.01, forecast_mask_smooth_weight=.01)
        model = Model(args)
        model.configure_pretraining()
        x, target = torch.randn(2, 12, 7), torch.randn(2, 12, 7)
        outputs = model.forward_with_jepa(
            x, None, None, None, target, None,
            return_forecast_mask=True)
        forecast, mask_info = outputs[0], outputs[-1]
        self.assertEqual(forecast.shape, target.shape)
        self.assertEqual(mask_info['weights'].shape, target.shape)
        self.assertTrue(torch.all(mask_info['weights'] >= args.forecast_mask_floor))
        self.assertTrue(torch.all(mask_info['weights'] <= 1.0))

        exp = Exp.__new__(Exp)
        exp.args = args
        loss, metrics = exp._forecast_pretrain_loss(
            forecast, target, mask_info)
        self.assertEqual(set(metrics), {
            'forecast', 'forecast_weighted', 'mask_budget',
            'mask_neg_entropy', 'mask_smooth', 'mask_mean',
            'mask_min', 'mask_max'})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.forecast_mask_head.parameters()))

        model.configure_finetuning('full')
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in model.forecast_mask_head.parameters()))


if __name__ == '__main__':
    unittest.main()

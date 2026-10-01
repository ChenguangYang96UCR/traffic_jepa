import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staeformer_jepa.model import FutureJEPA, STAEformerEncoder, STAEformerForecast


class ModelTests(unittest.TestCase):
    def make_encoder(self):
        return STAEformerEncoder(
            num_nodes=7,
            max_steps=12,
            input_dim=3,
            input_embedding_dim=8,
            tod_embedding_dim=4,
            dow_embedding_dim=4,
            adaptive_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )

    def test_future_jepa_shapes_and_stop_gradient(self):
        encoder = self.make_encoder()
        model = FutureJEPA(encoder, 12, 3, heads=4, predictor_layers=1, dropout=0.0)
        history = torch.randn(2, 12, 7, 3)
        future = torch.randn(2, 3, 7, 3)
        history[..., 1] = torch.rand(2, 12, 7)
        future[..., 1] = torch.rand(2, 3, 7)
        history[..., 2] = torch.randint(0, 7, (2, 12, 7)).float()
        future[..., 2] = torch.randint(0, 7, (2, 3, 7)).float()
        prediction, target = model(history, future)
        self.assertEqual(prediction.shape, (2, 3, 7, encoder.model_dim))
        self.assertEqual(target.shape, prediction.shape)
        torch.nn.functional.l1_loss(prediction, target).backward()
        self.assertTrue(any(p.grad is not None for p in model.online_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_forecast_shape(self):
        encoder = self.make_encoder()
        model = STAEformerForecast(encoder, input_steps=12, pred_steps=3)
        history = torch.randn(2, 12, 7, 3)
        history[..., 1] = torch.rand(2, 12, 7)
        history[..., 2] = torch.randint(0, 7, (2, 12, 7)).float()
        output = model(history)
        self.assertEqual(output.shape, (2, 3, 7, 1))


if __name__ == "__main__":
    unittest.main()

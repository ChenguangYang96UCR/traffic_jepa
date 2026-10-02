import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from staeformer_jepa.model import (
    FutureJEPA,
    MaskedFutureForecast,
    MaskedStepJEPA,
    STAEformerEncoder,
    STAEformerForecast,
)
from staeformer_jepa.data import FremontForecastDataset
from run_transfer import load_cross_city_encoder


class ModelTests(unittest.TestCase):
    def make_encoder(self):
        return STAEformerEncoder(
            num_nodes=7,
            max_steps=12,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )

    def test_future_jepa_shapes_and_stop_gradient(self):
        encoder = self.make_encoder()
        model = FutureJEPA(encoder, 12, 3, heads=4, predictor_layers=1, dropout=0.0)
        history = torch.randn(2, 12, 7, 1)
        future = torch.randn(2, 3, 7, 1)
        prediction, target = model(history, future)
        self.assertEqual(prediction.shape, (2, 3, 7, encoder.model_dim))
        self.assertEqual(target.shape, prediction.shape)
        torch.nn.functional.l1_loss(prediction, target).backward()
        self.assertTrue(any(p.grad is not None for p in model.online_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_forecast_shape(self):
        encoder = self.make_encoder()
        model = STAEformerForecast(encoder, input_steps=12, pred_steps=3)
        history = torch.randn(2, 12, 7, 1)
        output = model(history)
        self.assertEqual(output.shape, (2, 3, 7, 1))

    def test_random_step_jepa_and_future_mask(self):
        encoder = STAEformerEncoder(
            num_nodes=7,
            max_steps=15,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        model = MaskedStepJEPA(
            encoder, input_steps=12, pred_steps=3, masked_steps=3, dropout=0.0
        )
        history = torch.randn(4, 12, 7, 1)
        future = torch.randn(4, 3, 7, 1)
        prediction, target, random_positions = model(history, future)
        self.assertEqual(prediction.shape, (4, 3, 7, encoder.model_dim))
        self.assertEqual(target.shape, prediction.shape)
        self.assertTrue(torch.all(random_positions[:, 1:] > random_positions[:, :-1]))
        suffix = model.future_mask_positions(4, history.device)
        self.assertTrue(torch.equal(suffix, torch.tensor([[12, 13, 14]]).expand(4, -1)))

    def test_masked_future_forecast_shape(self):
        encoder = STAEformerEncoder(
            num_nodes=7,
            max_steps=15,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        model = MaskedFutureForecast(encoder, input_steps=12, pred_steps=3)
        output = model(torch.randn(2, 12, 7, 1))
        self.assertEqual(output.shape, (2, 3, 7, 1))

    def test_twelve_step_causal_future_mask(self):
        encoder = STAEformerEncoder(
            num_nodes=5,
            max_steps=24,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        jepa = MaskedStepJEPA(
            encoder, input_steps=12, pred_steps=12, masked_steps=12, dropout=0.0
        )
        history = torch.randn(2, 12, 5, 1)
        future = torch.randn(2, 12, 5, 1)
        positions = jepa.future_mask_positions(2, history.device)
        expected = torch.arange(12, 24)[None].expand(2, -1)
        self.assertTrue(torch.equal(positions, expected))
        prediction, target, used = jepa(history, future, positions)
        self.assertEqual(prediction.shape, (2, 12, 5, encoder.model_dim))
        self.assertTrue(torch.equal(used, expected))
        forecaster = MaskedFutureForecast(encoder, input_steps=12, pred_steps=12)
        self.assertEqual(tuple(forecaster(history).shape), (2, 12, 5, 1))

    def test_cross_city_load_reinitializes_sensor_embedding(self):
        source = self.make_encoder()
        target = STAEformerEncoder(
            num_nodes=5,
            max_steps=12,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=16,
            feed_forward_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        initial_sensor = target.sensor_embedding.detach().clone()
        report = load_cross_city_encoder(target, source.state_dict())
        self.assertIn("sensor_embedding", report["reinitialized_city_specific"])
        self.assertTrue(torch.equal(target.sensor_embedding, initial_sensor))
        self.assertTrue(
            torch.equal(target.relative_step_embedding, source.relative_step_embedding)
        )

    def test_traffic_only_object_array_loading(self):
        samples = np.empty(2, dtype=object)
        for index in range(2):
            samples[index] = {
                "x_data": np.random.randn(12, 5, 3).astype(np.float32),
                "y_data": np.random.randn(12, 5, 3).astype(np.float32),
                "incident_features": np.zeros((1,), dtype=np.float32),
            }
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "incident_train.npy", samples, allow_pickle=True)
            dataset = FremontForecastDataset(
                directory, "train", input_steps=12, pred_steps=3, traffic_feature=0
            )
        history, future, labels = dataset[0]
        self.assertEqual(tuple(history.shape), (12, 5, 1))
        self.assertEqual(tuple(future.shape), (3, 5, 1))
        self.assertTrue(torch.equal(future, labels))


if __name__ == "__main__":
    unittest.main()

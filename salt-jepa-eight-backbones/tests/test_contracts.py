import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from summarize import mean_std

try:
    import torch
    from torch import nn
    from salt_backbones.models import BACKBONES, build_backbone
    from salt_backbones.training import (
        BackboneReconstructionTeacher, Distiller, SpatioTemporalBlockMasker,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    BACKBONES = ()
    build_backbone = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this verification environment")
class EncoderContractTest(unittest.TestCase):
    def test_inventory(self):
        self.assertEqual(
            BACKBONES,
            ("pdformer", "flashst", "patchstg", "testam", "patchtst",
             "staeformer", "stgormer", "tsformer"),
        )

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in
            ("torchinfo",)),
        "Install requirements.txt to exercise all official backbones",
    )
    def test_shapes(self):
        nodes = 8
        history = torch.randn(2, 12, nodes, 1)
        train_x = np.random.default_rng(2026).normal(
            size=(32, 12, nodes, 1)
        ).astype(np.float32)
        adjacency = np.eye(nodes, dtype=np.float32)
        adjacency += np.roll(np.eye(nodes, dtype=np.float32), 1, axis=0)
        for horizon in (6, 9, 12):
            for name in BACKBONES:
                with self.subTest(name=name, horizon=horizon):
                    model = build_backbone(
                        name, nodes, input_steps=12, pred_steps=horizon,
                        adj=adjacency, train_x=train_x,
                    )
                    latent = model.encode_history(history)
                    forecast = model(history)
                    self.assertEqual(tuple(forecast.shape), (2, horizon, nodes, 1))
                    self.assertEqual(latent.shape[0], 2)
                    self.assertEqual(latent.shape[2], nodes)
                    self.assertEqual(latent.shape[3], model.latent_dim)

    def test_future_biased_mask_contract(self):
        masker = SpatioTemporalBlockMasker(
            num_blocks=8, future_start=12, future_block_ratio=0.5,
        )
        mask = masker(3, 18, 17, torch.device("cpu"),
                      torch.Generator().manual_seed(2026))
        self.assertEqual(tuple(mask.shape), (3, 18, 17))
        self.assertTrue(bool(mask[:, :12].any()))
        self.assertTrue(bool(mask[:, 12:].any()))

    def test_matched_teacher_student_contract(self):
        class DummyBackbone(nn.Module):
            latent_dim = 8

            def __init__(self):
                super().__init__()
                self.input_steps, self.pred_steps = 12, 6
                self.future_token = nn.Parameter(torch.zeros(1))
                self.projection = nn.Linear(1, self.latent_dim)

            def encode_sequence(self, sequence):
                return self.projection(sequence)

            def encode_history(self, history):
                future = self.future_token.expand(
                    history.shape[0], self.pred_steps, history.shape[2], 1
                )
                return self.encode_sequence(torch.cat((history, future), 1))

            def encoder_parameters(self):
                yield self.future_token
                yield from self.projection.parameters()

        teacher = DummyBackbone()
        student = DummyBackbone()
        history = torch.randn(2, 12, 5, 1)
        future = torch.randn(2, 6, 5, 1)
        prediction, target = Distiller(teacher, student, 0.0)(history, future)
        self.assertEqual(tuple(prediction.shape), (2, 18, 5, 8))
        self.assertEqual(prediction.shape, target.shape)
        mask = SpatioTemporalBlockMasker()(2, 18, 5, torch.device("cpu"))
        reconstructed = BackboneReconstructionTeacher(teacher)(
            torch.cat((history, future), 1), mask
        )
        self.assertEqual(tuple(reconstructed.shape), (2, 18, 5, 1))


class SummaryStatisticsTest(unittest.TestCase):
    def test_sample_mean_and_std(self):
        mean, std = mean_std([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
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
    from torch.utils.data import DataLoader, TensorDataset
    from salt_backbones.models import BACKBONES, build_backbone
    from salt_backbones.shared_teacher import SharedTrafficTeacher
    from salt_backbones.training import (
        Distiller, SpatioTemporalBlockMasker, fit_shared_teacher,
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

    def test_shared_teacher_multiple_student_contract(self):
        class DummyBackbone(nn.Module):
            def __init__(self, latent_dim, patch=False):
                super().__init__()
                self.latent_dim = latent_dim
                self.input_steps, self.pred_steps = 12, 6
                self.patch = patch
                self.future_token = nn.Parameter(torch.zeros(1))
                self.projection = nn.Linear(1, self.latent_dim)

            def encode_sequence(self, sequence):
                hidden = self.projection(sequence)
                if self.patch:
                    hidden = hidden.view(
                        hidden.shape[0], 6, 3, hidden.shape[2], hidden.shape[3]
                    ).mean(dim=2)
                return hidden

            def encode_history(self, history):
                future = self.future_token.expand(
                    history.shape[0], self.pred_steps, history.shape[2], 1
                )
                return self.encode_sequence(torch.cat((history, future), 1))

            def encoder_parameters(self):
                yield self.future_token
                yield from self.projection.parameters()

            def align_teacher(self, hidden):
                if not self.patch:
                    return hidden
                return hidden.view(
                    hidden.shape[0], 6, 3, hidden.shape[2], hidden.shape[3]
                ).mean(dim=2)

        adjacency = np.eye(5, dtype=np.float32)
        teacher = SharedTrafficTeacher(
            adjacency, input_steps=12, pred_steps=6, latent_dim=16,
            layers=1, heads=4, ff_dim=32, graph_pe_dim=4, dropout=0.0,
        )
        history = torch.randn(2, 12, 5, 1)
        future = torch.randn(2, 6, 5, 1)
        mask = SpatioTemporalBlockMasker()(2, 18, 5, torch.device("cpu"))
        reconstructed = teacher.reconstruct(torch.cat((history, future), 1), mask)
        self.assertEqual(tuple(reconstructed.shape), (2, 18, 5, 1))
        for student in (DummyBackbone(8), DummyBackbone(12, patch=True)):
            prediction, target = Distiller(teacher, student, 0.0)(history, future)
            self.assertEqual(prediction.shape, target.shape)
            self.assertEqual(prediction.shape[-1], 16)
        self.assertEqual(tuple(Distiller(teacher, DummyBackbone(8), 0.0)(
            history, future
        )[0].shape), (2, 18, 5, 16))
        self.assertEqual(tuple(Distiller(teacher, DummyBackbone(12, True), 0.0)(
            history, future
        )[0].shape), (2, 6, 5, 16))

    def test_shared_teacher_checkpoint_contract(self):
        nodes = 5
        adjacency = np.eye(nodes, dtype=np.float32)
        model = SharedTrafficTeacher(
            adjacency, input_steps=12, pred_steps=6, latent_dim=16,
            layers=1, heads=4, ff_dim=32, graph_pe_dim=4, dropout=0.0,
        )
        dataset = TensorDataset(
            torch.randn(4, 12, nodes, 1), torch.randn(4, 6, nodes, 1)
        )
        loaders = {
            split: DataLoader(dataset, batch_size=2, shuffle=False)
            for split in ("train", "val", "test")
        }
        masker = SpatioTemporalBlockMasker(
            num_blocks=2, min_time=1, max_time=2,
            min_sensor_ratio=0.2, max_sensor_ratio=0.4,
            future_start=12, future_block_ratio=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            saved = fit_shared_teacher(
                model, loaders, torch.device("cpu"), Path(directory),
                epochs=1, lr=1e-3, patience=1, masker=masker,
                validation_seed=2026,
                checkpoint_metadata={
                    "mask_blocks": 2, "future_block_ratio": 0.5,
                    "mask_min_time": 1, "mask_max_time": 2,
                    "mask_min_sensor_ratio": 0.2,
                    "mask_max_sensor_ratio": 0.4,
                },
            )
            self.assertTrue(Path(directory, "best_teacher.pt").exists())
            self.assertEqual(saved["metadata"]["architecture"], model.architecture)
            self.assertIn("teacher", saved)


class SummaryStatisticsTest(unittest.TestCase):
    def test_sample_mean_and_std(self):
        mean, std = mean_std([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)


if __name__ == "__main__":
    unittest.main()

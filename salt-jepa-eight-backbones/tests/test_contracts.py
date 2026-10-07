import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from summarize import mean_std

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from salt_backbones.models import BACKBONES, build_backbone
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
            ("torchinfo", "positional_encodings")),
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
        for name in BACKBONES:
            with self.subTest(name=name):
                model = build_backbone(
                    name, nodes, adj=adjacency, train_x=train_x
                )
                latent = model.encode_history(history)
                forecast = model(history)
                self.assertEqual(tuple(forecast.shape), (2, 12, nodes, 1))
                self.assertEqual(latent.shape[0], 2)
                self.assertEqual(latent.shape[2], nodes)
                self.assertEqual(latent.shape[3], model.latent_dim)


class SummaryStatisticsTest(unittest.TestCase):
    def test_sample_mean_and_std(self):
        mean, std = mean_std([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)


if __name__ == "__main__":
    unittest.main()

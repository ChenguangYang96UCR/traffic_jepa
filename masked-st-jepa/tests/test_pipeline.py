import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_st_jepa.graph import normalize_adjacency
from masked_st_jepa.masking import SensorMasker
from masked_st_jepa.model import MaskedSTJEPA
from masked_st_jepa.train_utils import masked_losses


class PipelineTest(unittest.TestCase):
    def test_shapes_mask_count_and_backward(self):
        batch, steps, nodes = 2, 12, 7
        raw = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes)
        adjacency = normalize_adjacency(raw)
        masker = SensorMasker(nodes, 2, "random", raw)
        mask = masker(batch, steps, torch.device("cpu"), torch.Generator().manual_seed(4))
        self.assertEqual(mask.shape, (batch, steps, nodes))
        self.assertTrue(torch.all(mask.sum(dim=-1) == 2))

        model = MaskedSTJEPA(nodes, steps, adjacency, dim=16, heads=4, layers=1)
        values = torch.randn(batch, steps, nodes, 1)
        prediction, latent, target = model(values, mask)
        self.assertEqual(prediction.shape, values.shape)
        self.assertEqual(latent.shape, (batch, steps, nodes, 16))
        losses = masked_losses(prediction, latent, target, values, mask)
        (losses["value"] + 0.1 * losses["latent"]).backward()
        self.assertIsNotNone(model.value_head[-1].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.target_encoder.parameters()))

    def test_persistent_mask(self):
        nodes = 6
        raw = np.eye(nodes, dtype=np.float32)
        mask = SensorMasker(nodes, 2, "persistent", raw)(
            3, 5, torch.device("cpu"), torch.Generator().manual_seed(5)
        )
        self.assertTrue(torch.equal(mask[:, :1].expand_as(mask), mask))


if __name__ == "__main__":
    unittest.main()

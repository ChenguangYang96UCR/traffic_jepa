import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import configure_downstream, load_cross_city_encoder
from staeformer_salt.distillation import (
    ReconstructionTeacher,
    SALTDistiller,
    SpatioTemporalBlockMasker,
)
from staeformer_salt.model import MaskedFutureForecast, STAEformerEncoder


def encoder(nodes: int, width: int = 1):
    base = 4 * width
    return STAEformerEncoder(
        num_nodes=nodes,
        max_steps=24,
        input_dim=1,
        input_embedding_dim=base,
        step_embedding_dim=base,
        sensor_embedding_dim=2 * base,
        feed_forward_dim=8 * base,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


class SALTTests(unittest.TestCase):
    def test_block_mask_and_reconstruction(self):
        sequence = torch.randn(2, 24, 5, 1)
        mask = SpatioTemporalBlockMasker(2, 2, 4, 0.2, 0.5)(
            2, 24, 5, sequence.device
        )
        self.assertEqual(tuple(mask.shape), (2, 24, 5))
        self.assertTrue(mask.any())
        self.assertEqual(
            tuple(ReconstructionTeacher(encoder(5))(sequence, mask).shape),
            tuple(sequence.shape),
        )

    def test_static_teacher_receives_no_gradient(self):
        model = SALTDistiller(encoder(5), encoder(5, 2), 12, 12, 0.0)
        prediction, target = model(
            torch.randn(2, 12, 5, 1), torch.randn(2, 12, 5, 1)
        )
        torch.nn.functional.l1_loss(prediction, target).backward()
        self.assertTrue(any(p.grad is not None for p in model.student.parameters()))
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_shared_teacher_supports_city_sensor_subset(self):
        model = SALTDistiller(
            encoder(7),
            encoder(3, 2),
            12,
            12,
            0.0,
            teacher_node_indices=torch.tensor([1, 3, 6]),
        )
        prediction, target = model(
            torch.randn(2, 12, 3, 1), torch.randn(2, 12, 3, 1)
        )
        self.assertEqual(tuple(prediction.shape[:3]), (2, 12, 3))
        self.assertEqual(tuple(target.shape), tuple(prediction.shape))

    def test_node_agnostic_oakland_teacher_accepts_fremont_student(self):
        teacher = STAEformerEncoder(
            num_nodes=5,
            max_steps=24,
            input_dim=1,
            input_embedding_dim=8,
            step_embedding_dim=8,
            sensor_embedding_dim=0,
            feed_forward_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        model = SALTDistiller(teacher, encoder(3, 2), 12, 12, 0.0)
        prediction, target = model(
            torch.randn(2, 12, 3, 1), torch.randn(2, 12, 3, 1)
        )
        self.assertEqual(tuple(prediction.shape[:3]), (2, 12, 3))
        self.assertEqual(tuple(target.shape), tuple(prediction.shape))

    def test_cross_city_skips_sensor_table(self):
        source, target = encoder(5), encoder(3)
        initial = target.sensor_embedding.detach().clone()
        report = load_cross_city_encoder(target, source.state_dict())
        self.assertIn("sensor_embedding", report["reinitialized_city_specific"])
        self.assertTrue(torch.equal(target.sensor_embedding, initial))

    def test_forecast_shape(self):
        model = MaskedFutureForecast(encoder(5), 12, 12)
        self.assertEqual(tuple(model(torch.randn(2, 12, 5, 1)).shape), (2, 12, 5, 1))


if __name__ == "__main__":
    unittest.main()

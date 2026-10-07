import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from salt_backbones.models import build_backbone
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    build_backbone = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this local verification environment")
class EncoderContractTest(unittest.TestCase):
    def test_shapes(self):
        history = torch.randn(2, 12, 5, 1)
        for name in ("patchtst", "stgformer", "stlaformer"):
            with self.subTest(name=name):
                model = build_backbone(name, 5)
                latent = model.encode_history(history)
                forecast = model(history)
                self.assertEqual(tuple(forecast.shape), (2, 12, 5, 1))
                self.assertEqual(latent.shape[0], 2)
                self.assertEqual(latent.shape[2], 5)
                self.assertEqual(latent.shape[3], model.latent_dim)


if __name__ == "__main__":
    unittest.main()

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer_benchmark.gnn import TrafficGNN, transferable_state
from transfer_benchmark.tsjepa import batched_gather, random_patch_split


class PipelineTests(unittest.TestCase):
    def test_ts_jepa_patch_mask_contract(self):
        tokens = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        indices = torch.tensor([[3, 1], [0, 2]])
        selected = batched_gather(tokens, indices)
        self.assertEqual(tuple(selected.shape), (2, 2, 3))
        self.assertTrue(torch.equal(selected[0, 0], tokens[0, 3]))
        masked, visible = random_patch_split(
            batch=5, patches=4, mask_ratio=0.7, device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(2026),
        )
        self.assertEqual(tuple(masked.shape), (5, 2))
        self.assertEqual(tuple(visible.shape), (5, 2))
        for row in range(5):
            self.assertEqual(
                sorted(torch.cat((masked[row], visible[row])).tolist()),
                [0, 1, 2, 3],
            )

    def test_gnn_contract_and_cross_city_transfer(self):
        for kind in ("dcrnn", "agcrn"):
            source = TrafficGNN(5, 12, 6, np.eye(5, dtype=np.float32), kind)
            target = TrafficGNN(3, 12, 6, np.eye(3, dtype=np.float32), kind)
            report = transferable_state(target, source.state_dict())
            self.assertTrue(report["copied"])
            prediction = target(torch.randn(2, 12, 3, 1))
            self.assertEqual(tuple(prediction.shape), (2, 6, 3, 1))

    def test_summarizer_writes_avg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "city", "source_city", "method", "horizon", "history",
                    "mae", "mse", "rmse", "seed", "transfer",
                ))
                writer.writeheader()
                for horizon in (6, 9, 12):
                    writer.writerow({
                        "city": "Berkeley", "source_city": "Oakland",
                        "method": "ours", "horizon": horizon, "history": 12,
                        "mae": horizon / 100, "mse": horizon / 1000,
                        "rmse": .1, "seed": 2026, "transfer": "full_finetune",
                    })
            subprocess.run(
                [sys.executable, str(ROOT / "summarize.py"), "--root", str(root)],
                check=True,
            )
            text = (root / "paper_table_mae_mse.csv").read_text()
            self.assertIn("Avg", text)
            self.assertIn("Ours (STAEformer+SALT)", text)


if __name__ == "__main__":
    unittest.main()

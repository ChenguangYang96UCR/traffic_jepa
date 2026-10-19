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
from transfer_benchmark.recent import (
    DSTFGCNForecast,
    SDGFormerForecast,
    recent_encoder_transfer,
)
from transfer_benchmark.tsjepa import batched_gather, random_patch_split
from prepare_county_cities import prepare


class PipelineTests(unittest.TestCase):
    def test_county_city_views_keep_sensor_graph_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            county, output = root / "Orange", root / "cities"
            county.mkdir()
            output.mkdir()
            with (county / "sensors.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("ID", "City"))
                writer.writeheader()
                for index, city in enumerate(("Irvine", "Other", "Irvine", "Target")):
                    writer.writerow({"ID": index, "City": city})
            adjacency = np.arange(16, dtype=np.float32).reshape(4, 4)
            np.save(county / "adj_matrix.npy", adjacency)
            for split in ("train", "val", "test"):
                samples = np.asarray([{
                    "x_data": np.arange(12 * 4, dtype=np.float32).reshape(12, 4, 1),
                    "y_data": np.arange(12 * 4, dtype=np.float32).reshape(12, 4, 1),
                    "incident_distances": np.arange(4, dtype=np.float32),
                }], dtype=object)
                np.save(county / f"incident_{split}.npy", samples, allow_pickle=True)
            prepare(county, output, ["Irvine", "Target"])
            irvine = np.load(output / "Irvine/incident_train.npy", allow_pickle=True)
            self.assertEqual(irvine[0]["x_data"].shape, (12, 2, 1))
            self.assertEqual(irvine[0]["incident_distances"].tolist(), [0.0, 2.0])
            expected = adjacency[np.ix_([0, 2], [0, 2])]
            self.assertTrue(np.array_equal(np.load(output / "Irvine/adj_matrix.npy"), expected))
            manifest = (output / "Irvine/city_view_manifest.json").read_text()
            self.assertIn('"sensor_indices": [', manifest)

    def test_recent_model_contracts_and_cross_city_transfer(self):
        for model_class in (SDGFormerForecast, DSTFGCNForecast):
            source = model_class(7, 12, 6, np.eye(7, dtype=np.float32), **(
                {"train_samples": 16} if model_class is SDGFormerForecast else {}
            ))
            target = model_class(5, 12, 6, np.eye(5, dtype=np.float32), **(
                {"train_samples": 16} if model_class is SDGFormerForecast else {}
            ))
            report = recent_encoder_transfer(target, source.state_dict())
            self.assertTrue(report["copied"])
            prediction = target(torch.randn(2, 12, 5, 1))
            self.assertEqual(tuple(prediction.shape), (2, 6, 5, 1))

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

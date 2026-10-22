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
from transfer_benchmark.tsjepa import (
    TSJEPAForecast,
    batched_gather,
    random_patch_split,
)
from transfer_benchmark.salt_native import NativeSALTAdapter, SALT_TABLE_BACKBONES
from prepare_county_cities import prepare
from run import restrict_target_train
from salt_backbones.training import (
    MatchedRandomPointMasker,
    SpatioTemporalBlockMasker,
    TopologyAwareSpatioTemporalBlockMasker,
)
from run_mask_ablation_multicounty import configurations


class PipelineTests(unittest.TestCase):
    def test_multicounty_mask_ablation_inventory_and_random_point_count(self):
        configs = configurations()
        self.assertEqual(len(configs), 9)
        self.assertEqual(
            {(item["mask_mode"], item["history_future_ratio"]) for item in configs},
            {
                (mode, ratio)
                for ratio in ("1:9", "3:7", "5:5", "8:2")
                for mode in ("random_block", "topology_block")
            } | {("random_point", "n/a")},
        )
        reference = SpatioTemporalBlockMasker(
            num_blocks=8, min_time=2, max_time=4,
            min_sensor_ratio=0.2, max_sensor_ratio=0.4,
            future_start=12, future_block_ratio=0.5,
        )
        points = MatchedRandomPointMasker(reference)
        first_generator = torch.Generator().manual_seed(19)
        second_generator = torch.Generator().manual_seed(19)
        block_mask = reference(3, 18, 10, torch.device("cpu"), first_generator)
        point_mask = points(3, 18, 10, torch.device("cpu"), second_generator)
        self.assertTrue(torch.equal(
            block_mask.flatten(1).sum(1), point_mask.flatten(1).sum(1)
        ))

    def test_salt_table_backbone_inventory_and_native_latent_contract(self):
        self.assertEqual(SALT_TABLE_BACKBONES, (
            "pdformer", "flashst", "testam", "patchstg", "scjepa",
            "ts_jepa", "dcrnn", "agcrn", "sdgformer", "dstfgcn",
        ))
        source_native = TrafficGNN(5, 12, 6, np.eye(5, dtype=np.float32), "dcrnn")
        target_native = TrafficGNN(3, 12, 6, np.eye(3, dtype=np.float32), "dcrnn")
        source = NativeSALTAdapter(source_native, 64, 12, 6)
        target = NativeSALTAdapter(target_native, 64, 12, 6)
        from salt_backbones.models import compatible_encoder_transfer
        report = compatible_encoder_transfer(target, source.state_dict())
        self.assertTrue(report["copied"])
        self.assertIn("native.fixed_support", report["skipped"])
        history = torch.randn(2, 12, 3, 1)
        self.assertEqual(tuple(target.encode_history(history).shape), (2, 18, 3, 128))
        self.assertEqual(tuple(target(history).shape), (2, 6, 3, 1))

    def test_topology_mask_selects_a_connected_sensor_block(self):
        adjacency = np.zeros((5, 5), dtype=np.float32)
        for node in range(4):
            adjacency[node, node + 1] = adjacency[node + 1, node] = 1
        masker = TopologyAwareSpatioTemporalBlockMasker(
            adjacency, num_blocks=1, min_time=2, max_time=2,
            min_sensor_ratio=0.6, max_sensor_ratio=0.6,
            future_start=4, future_block_ratio=0.0,
        )
        mask = masker(
            1, 8, 5, torch.device("cpu"),
            torch.Generator().manual_seed(2026),
        )
        selected = torch.nonzero(mask[0].any(dim=0), as_tuple=False).flatten().tolist()
        self.assertEqual(len(selected), 3)
        reached = {selected[0]}
        while True:
            expanded = reached | {
                node for node in selected
                if any(adjacency[node, other] > 0 for other in reached)
            }
            if expanded == reached:
                break
            reached = expanded
        self.assertEqual(reached, set(selected))

    def test_few_shot_subset_is_train_only_and_deterministic(self):
        class Windows:
            def __init__(self, count):
                self.history = torch.arange(count * 2).reshape(count, 2, 1, 1)
                self.future = self.history.clone()

            def __len__(self):
                return len(self.history)

        datasets = {"train": Windows(20), "val": Windows(7), "test": Windows(9)}
        first, indices_a = restrict_target_train(datasets, 0.10, 2026, "random")
        second, indices_b = restrict_target_train(datasets, 0.10, 2026, "random")
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(len(first["train"]), 2)
        self.assertIs(first["val"], datasets["val"])
        self.assertIs(first["test"], datasets["test"])
        self.assertTrue(torch.equal(first["train"].history, second["train"].history))

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

    def test_ts_jepa_patch_tensor_is_contiguous_for_official_tokenizer(self):
        dummy = type("Dummy", (), {
            "input_steps": 12, "num_patches": 4, "patch_size": 3,
        })()
        history = torch.randn(2, 12, 5, 1)
        patches = TSJEPAForecast.patches(dummy, history)
        self.assertEqual(tuple(patches.shape), (10, 4, 3))
        self.assertTrue(patches.is_contiguous())

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

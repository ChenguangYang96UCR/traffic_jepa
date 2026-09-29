from __future__ import annotations

import numpy as np
import torch


class SensorMasker:
    """Generate exactly k masked nodes per time step."""

    def __init__(self, nodes: int, k: int, mode: str, adjacency: np.ndarray):
        if not 1 <= k < nodes:
            raise ValueError(f"mask_k must be in [1,{nodes - 1}], got {k}")
        self.nodes = nodes
        self.k = k
        self.mode = mode
        self.neighbors = [
            np.flatnonzero((adjacency[node] != 0) | (adjacency[:, node] != 0)).tolist()
            for node in range(nodes)
        ]

    def _spatial_block(self, generator: torch.Generator) -> list[int]:
        start = int(torch.randint(self.nodes, (1,), generator=generator).item())
        chosen = [start]
        frontier = [start]
        while frontier and len(chosen) < self.k:
            node = frontier.pop(0)
            candidates = self.neighbors[node]
            if candidates:
                order = torch.randperm(len(candidates), generator=generator).tolist()
                for index in order:
                    candidate = candidates[index]
                    if candidate not in chosen:
                        chosen.append(candidate)
                        frontier.append(candidate)
                    if len(chosen) == self.k:
                        break
        if len(chosen) < self.k:
            remaining = [node for node in range(self.nodes) if node not in chosen]
            order = torch.randperm(len(remaining), generator=generator).tolist()
            chosen.extend(remaining[index] for index in order[: self.k - len(chosen)])
        return chosen

    def __call__(
        self,
        batch: int,
        steps: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        generator = generator or torch.Generator()
        mask = torch.zeros(batch, steps, self.nodes, dtype=torch.bool)
        for sample in range(batch):
            if self.mode == "persistent":
                selected = torch.randperm(self.nodes, generator=generator)[: self.k]
                mask[sample, :, selected] = True
            elif self.mode == "spatial_block":
                for step in range(steps):
                    mask[sample, step, self._spatial_block(generator)] = True
            elif self.mode == "random":
                scores = torch.rand(steps, self.nodes, generator=generator)
                selected = scores.topk(self.k, dim=-1).indices
                mask[sample].scatter_(1, selected, True)
            else:
                raise ValueError(f"Unknown mask mode: {self.mode}")
        return mask.to(device)


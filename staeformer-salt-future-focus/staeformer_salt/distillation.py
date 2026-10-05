from __future__ import annotations

import torch
from torch import nn

from .model import STAEformerEncoder


class SpatioTemporalBlockMasker:
    """Spatio-temporal block masking with optional future-biased allocation.

    With ``future_start=None`` blocks are sampled over the complete sequence,
    matching the original implementation. Otherwise a fixed fraction of blocks
    is sampled wholly inside the future region and the remainder wholly inside
    history. This guarantees the intended allocation instead of relying on a
    per-block Bernoulli draw.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        min_time: int = 2,
        max_time: int = 6,
        min_sensor_ratio: float = 0.10,
        max_sensor_ratio: float = 0.30,
        future_start: int | None = None,
        future_block_ratio: float = 0.75,
    ):
        if not 0 < min_sensor_ratio <= max_sensor_ratio <= 1:
            raise ValueError("Sensor ratios must satisfy 0 < min <= max <= 1")
        if not 0 < min_time <= max_time:
            raise ValueError("Time block sizes must satisfy 0 < min <= max")
        if future_start is not None and future_start <= 0:
            raise ValueError("future_start must leave at least one history step")
        if not 0 <= future_block_ratio <= 1:
            raise ValueError("future_block_ratio must be in [0, 1]")
        self.num_blocks = num_blocks
        self.min_time = min_time
        self.max_time = max_time
        self.min_sensor_ratio = min_sensor_ratio
        self.max_sensor_ratio = max_sensor_ratio
        self.future_start = future_start
        self.future_block_ratio = future_block_ratio

    def _regions(self, steps: int) -> list[tuple[int, int]]:
        if self.future_start is None:
            return [(0, steps)] * self.num_blocks
        if self.future_start >= steps:
            raise ValueError(
                f"future_start={self.future_start} must be smaller than steps={steps}"
            )
        future_blocks = int(self.num_blocks * self.future_block_ratio + 0.5)
        future_blocks = min(self.num_blocks, max(0, future_blocks))
        history_blocks = self.num_blocks - future_blocks
        return (
            [(self.future_start, steps)] * future_blocks
            + [(0, self.future_start)] * history_blocks
        )

    def __call__(
        self,
        batch: int,
        steps: int,
        nodes: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch, steps, nodes, dtype=torch.bool, device=device)
        min_nodes = max(1, round(nodes * self.min_sensor_ratio))
        max_nodes = max(min_nodes, round(nodes * self.max_sensor_ratio))
        regions = self._regions(steps)
        for sample in range(batch):
            for region_start, region_end in regions:
                region_steps = region_end - region_start
                max_time = min(self.max_time, region_steps)
                min_time = min(self.min_time, max_time)
                length = int(
                    torch.randint(
                        min_time, max_time + 1, (1,), device=device, generator=generator
                    ).item()
                )
                local_start = int(
                    torch.randint(
                        0,
                        region_steps - length + 1,
                        (1,),
                        device=device,
                        generator=generator,
                    ).item()
                )
                start = region_start + local_start
                count = int(
                    torch.randint(
                        min_nodes, max_nodes + 1, (1,), device=device, generator=generator
                    ).item()
                )
                sensors = torch.randperm(nodes, device=device, generator=generator)[:count]
                mask[sample, start : start + length, sensors] = True
            if not mask[sample].any():
                mask[sample, 0, 0] = True
        return mask


class ReconstructionTeacher(nn.Module):
    """Stage-1 teacher trained against fixed raw traffic targets."""

    def __init__(self, encoder: STAEformerEncoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.model_dim, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(sequence, step_mask=mask))


class SALTPredictor(nn.Module):
    """Asymmetric projection from student width to frozen-teacher width."""

    def __init__(self, student_dim: int, teacher_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, 2 * student_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class SALTDistiller(nn.Module):
    """Stage-2 static-teacher asymmetric latent distillation."""

    def __init__(
        self,
        frozen_teacher: STAEformerEncoder,
        student: STAEformerEncoder,
        input_steps: int,
        pred_steps: int,
        dropout: float,
        teacher_node_indices: torch.Tensor | None = None,
        distill_scope: str = "future",
    ):
        super().__init__()
        if (
            teacher_node_indices is None
            and frozen_teacher.sensor_embedding_dim != 0
            and frozen_teacher.num_nodes != student.num_nodes
        ):
            raise ValueError("Different teacher/student node counts require teacher_node_indices")
        if teacher_node_indices is not None:
            teacher_node_indices = torch.as_tensor(teacher_node_indices, dtype=torch.long)
            if teacher_node_indices.numel() != student.num_nodes:
                raise ValueError("teacher_node_indices length must equal student node count")
            if teacher_node_indices.unique().numel() != teacher_node_indices.numel():
                raise ValueError("teacher_node_indices must be unique")
        if frozen_teacher.max_steps < input_steps + pred_steps:
            raise ValueError("Teacher does not cover the complete sequence")
        if student.max_steps < input_steps + pred_steps:
            raise ValueError("Student does not cover the complete sequence")
        if distill_scope not in ("all", "future"):
            raise ValueError("distill_scope must be 'all' or 'future'")
        self.teacher = frozen_teacher
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.student = student
        self.predictor = SALTPredictor(
            student.model_dim, frozen_teacher.model_dim, dropout
        )
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.distill_scope = distill_scope
        self.register_buffer(
            "teacher_node_indices", teacher_node_indices, persistent=True
        )

    def student_hidden(self, history: torch.Tensor) -> torch.Tensor:
        """Encode history plus future mask tokens without target leakage."""
        batch, history_steps, nodes, channels = history.shape
        if history_steps != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} history steps, got {history_steps}"
            )
        sequence = torch.cat(
            (
                history,
                history.new_zeros(batch, self.pred_steps, nodes, channels),
            ),
            dim=1,
        )
        mask = torch.zeros(
            batch,
            self.input_steps + self.pred_steps,
            dtype=torch.bool,
            device=history.device,
        )
        mask[:, self.input_steps :] = True
        return self.student(sequence, step_mask=mask)

    def teacher_hidden(
        self, history: torch.Tensor, future: torch.Tensor
    ) -> torch.Tensor:
        """Encode the complete sequence with the frozen Teacher."""
        sequence = torch.cat((history, future), dim=1)
        with torch.no_grad():
            target = self.teacher(sequence, node_indices=self.teacher_node_indices)
        return target.detach()

    def latent_triplet(self, history: torch.Tensor, future: torch.Tensor):
        """Return predicted latents, Teacher targets, and Student hidden states."""
        student_hidden = self.student_hidden(history)
        predicted = self.predictor(student_hidden)
        target = self.teacher_hidden(history, future)
        return predicted, target, student_hidden

    def latent_pairs(self, history: torch.Tensor, future: torch.Tensor):
        """Return predictor and frozen-Teacher latents for every time step."""
        predicted, target, _ = self.latent_triplet(history, future)
        return predicted, target

    def select_scope(self, predicted: torch.Tensor, target: torch.Tensor):
        if self.distill_scope == "future":
            return (
                predicted[:, self.input_steps :],
                target[:, self.input_steps :],
            )
        return predicted, target

    def forward(self, history: torch.Tensor, future: torch.Tensor):
        return self.select_scope(*self.latent_pairs(history, future))

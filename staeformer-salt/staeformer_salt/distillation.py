from __future__ import annotations

import torch
from torch import nn

from .model import STAEformerEncoder


class SpatioTemporalBlockMasker:
    """Spatio-temporal multi-block masking for teacher reconstruction."""

    def __init__(
        self,
        num_blocks: int = 3,
        min_time: int = 2,
        max_time: int = 6,
        min_sensor_ratio: float = 0.10,
        max_sensor_ratio: float = 0.30,
    ):
        if not 0 < min_sensor_ratio <= max_sensor_ratio <= 1:
            raise ValueError("Sensor ratios must satisfy 0 < min <= max <= 1")
        if not 0 < min_time <= max_time:
            raise ValueError("Time block sizes must satisfy 0 < min <= max")
        self.num_blocks = num_blocks
        self.min_time = min_time
        self.max_time = max_time
        self.min_sensor_ratio = min_sensor_ratio
        self.max_sensor_ratio = max_sensor_ratio

    def __call__(
        self,
        batch: int,
        steps: int,
        nodes: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch, steps, nodes, dtype=torch.bool, device=device)
        max_time = min(self.max_time, steps)
        min_time = min(self.min_time, max_time)
        min_nodes = max(1, round(nodes * self.min_sensor_ratio))
        max_nodes = max(min_nodes, round(nodes * self.max_sensor_ratio))
        for sample in range(batch):
            for _ in range(self.num_blocks):
                length = int(
                    torch.randint(
                        min_time, max_time + 1, (1,), device=device, generator=generator
                    ).item()
                )
                start = int(
                    torch.randint(
                        0, steps - length + 1, (1,), device=device, generator=generator
                    ).item()
                )
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
        self.teacher = frozen_teacher
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.student = student
        self.predictor = SALTPredictor(
            student.model_dim, frozen_teacher.model_dim, dropout
        )
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.register_buffer(
            "teacher_node_indices", teacher_node_indices, persistent=True
        )

    def forward(self, history: torch.Tensor, future: torch.Tensor):
        sequence = torch.cat((history, future), dim=1)
        batch = sequence.shape[0]
        mask = torch.zeros(
            batch,
            self.input_steps + self.pred_steps,
            dtype=torch.bool,
            device=sequence.device,
        )
        mask[:, self.input_steps :] = True
        student_hidden = self.student(sequence, step_mask=mask)
        predicted = self.predictor(student_hidden[:, self.input_steps :])
        with torch.no_grad():
            target = self.teacher(
                sequence, node_indices=self.teacher_node_indices
            )[:, self.input_steps :]
        return predicted, target.detach()

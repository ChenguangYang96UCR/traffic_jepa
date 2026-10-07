"""SALT distillation adapters for traffic Transformer encoders."""

from .models import build_backbone
from .teacher import load_frozen_teacher

__all__ = ["build_backbone", "load_frozen_teacher"]

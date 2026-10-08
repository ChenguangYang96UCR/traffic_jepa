"""Shared-Teacher SALT adapters for traffic Transformer encoders."""

from .models import build_backbone
from .shared_teacher import SharedTrafficTeacher

__all__ = ["build_backbone", "SharedTrafficTeacher"]

"""CPU-only ECS pose shared by material rendering and physics adapters."""

from dataclasses import field

from .ECS import component
from .Math import Vec3, Vec4


@component
class MaterialTransform:
	position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
	rotation: Vec4 = field(default_factory=lambda: Vec4(0, 0, 0, 1))
	scale: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))

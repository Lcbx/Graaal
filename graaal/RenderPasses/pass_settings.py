"""CPU settings shared by transparency composition and postprocessing effects."""
from typing import Annotated

from .settings_validation import checked_dataclass, float_range, int_range


@checked_dataclass(frozen=True)
class AntialiasingSettings:
	lxaa_edge_threshold: Annotated[float, float_range(0, 1).__metadata__[0]] = 0.375
	lxaa_line_check: bool = True
	faaa_edge_threshold: Annotated[float, float_range(1, 9).__metadata__[0]] = 5.0
	faaa_quality: Annotated[int, int_range(1, 9).__metadata__[0]] = 5
	ddaa2_strength: Annotated[float, float_range(0, 10).__metadata__[0]] = 3.0
	ddaa2_threshold_min: Annotated[float, float_range(0, 1).__metadata__[0]] = 0.0625
	ddaa2_threshold_max: Annotated[float, float_range(0, 1).__metadata__[0]] = 0.166
	ddaa2_edge_steps: Annotated[int, int_range(0, 5).__metadata__[0]] = 5
	show_edges: bool = False
	hybrid_luma_chroma: bool = False
	chroma_weight: Annotated[float, float_range(0, 5).__metadata__[0]] = 0.5


@checked_dataclass(frozen=True)
class AmbientOcclusionSettings:
	"""Sampling and strength controls for screen-space ambient occlusion."""
	radius: Annotated[float, float_range().__metadata__[0]] = 0.25
	sample_count: Annotated[int, int_range(1, 64).__metadata__[0]] = 8
	turns: Annotated[float, float_range().__metadata__[0]] = 5.0
	bias: Annotated[float, float_range().__metadata__[0]] = 0.05
	falloff: Annotated[float, float_range().__metadata__[0]] = 1.0
	strength: Annotated[float, float_range().__metadata__[0]] = 3.0

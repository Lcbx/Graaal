"""CPU settings shared by transparency composition and postprocessing effects."""
from .settings_validation import checked_dataclass, float_range, int_range


@checked_dataclass(frozen=True)
class AntialiasingSettings:
	lxaa_edge_threshold: float_range(0, 1) = 0.375
	lxaa_line_check: bool = True
	faaa_edge_threshold: float_range(1, 9) = 5.0
	faaa_quality: int_range(1, 9) = 5
	ddaa2_strength: float_range(0, 10) = 3.0
	ddaa2_threshold_min: float_range(0, 1) = 0.0625
	ddaa2_threshold_max: float_range(0, 1) = 0.166
	ddaa2_edge_steps: int_range(0, 5) = 5
	show_edges: bool = False
	hybrid_luma_chroma: bool = False
	chroma_weight: float_range(0, 5) = 0.5


@checked_dataclass(frozen=True)
class AmbientOcclusionSettings:
	"""Sampling and strength controls for screen-space ambient occlusion."""
	radius: float_range() = 0.25
	sample_count: int_range(1, 64) = 8
	turns: float_range() = 5.0
	bias: float_range() = 0.05
	falloff: float_range() = 1.0
	strength: float_range() = 3.0

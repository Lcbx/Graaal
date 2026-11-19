from dataclasses import replace

import pytest

from graaal.postprocessing_settings import AmbientOcclusionSettings
from graaal.settings_validation import checked_dataclass, float_range, int_range


@checked_dataclass(frozen=True)
class ExampleSettings:
	weight: float_range(0, 1) = 0.5
	count: int_range(1, 3) = 2
	enabled: bool = True

	def __post_init__(self):
		if self.enabled and self.weight == 0: raise ValueError("enabled requires weight")


def test_checked_dataclass_validates_fields_and_preserves_custom_checks():
	settings = ExampleSettings()
	assert replace(settings, weight=1, count=3).weight == 1
	for changes in ({"weight": -1}, {"weight": float("nan")}, {"weight": float("inf")}, {"count": 0}, {"count": True}, {"enabled": 1}, {"weight": 0}):
		with pytest.raises(ValueError): replace(settings, **changes)
	assert replace(settings, enabled=False, weight=0).weight == 0


@pytest.mark.parametrize("changes", [
	{"sample_count": 0}, {"sample_count": 65}, {"sample_count": True},
	{"radius": -1}, {"turns": float("inf")}, {"bias": float("nan")},
	{"falloff": -0.1}, {"strength": float("inf")},
])
def test_ambient_occlusion_settings_checks(changes):
	with pytest.raises(ValueError): replace(AmbientOcclusionSettings(), **changes)

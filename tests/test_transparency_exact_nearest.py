"""Exact nearest contribution and an ID-excluded banded tail."""
import numpy as np
import pytest

from transparency_helpers import factory, read, render, encoded


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("alpha", [0.001, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("passes", [0, 1, 3])
def test_single_layer_is_exact_and_tail_is_empty(factory, samples, alpha, passes):
	renderer = factory(sample_count=samples, alpha=alpha, kawase_passes=passes)
	image = render(renderer, (31, 23))
	oit = renderer.transparency
	np.testing.assert_allclose(read(oit.nearest_color), np.broadcast_to((1, 0, 0, 1), (samples, 23, 31, 4)), atol=0.001)
	np.testing.assert_array_equal(read(oit.winner_color), 0)
	expected = encoded(alpha*np.array([1, 0, 0])+(1-alpha)*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)
	if samples == 1:
		np.testing.assert_array_equal(read(oit.transmittance), 1)
		assert np.all(read(oit.nearest_payload, np.uint32)[..., 0] == 1)


@pytest.mark.parametrize("samples", [1, 4])
def test_two_half_layers_preserve_tail_and_compose_exact_mean(factory, samples):
	renderer = factory(sample_count=samples, alpha=0.5, layers=2)
	image = render(renderer, (63, 47))
	oit = renderer.transparency
	# Complementary selection of the one tail layer gives exactly half blue.
	np.testing.assert_allclose(read(oit.winner_color), np.broadcast_to((0, 0, 0.5, 0.5), (samples, 47, 63, 4)), atol=0.001)
	expected = encoded(np.array([0.5, 0, 0.25])+0.25*np.array([0.035, 0.05, 0.08]))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize("front_alpha", [0.5, 1.0])
def test_rgb_nearest_and_coplanar_tail_are_not_double_counted(factory, samples, front_alpha):
	renderer = factory(sample_count=samples, alpha=0.5, layers=2)
	renderer.instances[0].material["base_color"] = (1, 1, 1, front_alpha)
	renderer.instances[0].material["opacity_rgb"] = (1, 0.5, 0.25)
	image = render(renderer, (41, 27))
	a1 = front_alpha*np.array([1, 0.5, 0.25])
	expected = encoded(a1*np.array([1, 0, 0])+(1-a1)*(0.5*np.array([0, 0, 1])+0.5*np.array([0.035, 0.05, 0.08])))
	np.testing.assert_allclose(image[0, ..., :3], np.broadcast_to(expected, image[0, ..., :3].shape), atol=1)
	# Equal-depth fragments with different IDs must leave one layer in the tail.
	renderer.instances[0].material["opacity_rgb"] = (1, 1, 1)
	renderer.instances[0].material["base_color"] = (1, 1, 1, 0.5)
	renderer.instances[1].position = renderer.instances[0].position
	render(renderer, (41, 27))
	assert np.all(read(renderer.transparency.winner_color)[..., 3] == 0.5)


def test_owned_nearest_targets_validate_and_diagnostics_follow_option(factory):
	renderer = factory()
	render(renderer, (33, 25))
	oit = renderer.transparency
	targets = oit._targets
	from dataclasses import replace
	with pytest.raises(ValueError, match="nearest_payload"):
		oit.resize((33, 25), targets=replace(targets, nearest_payload=None))
	assert not oit.resize((33, 25), targets=targets)
	from graaal.UI.rendering_settings import FilterSettings
	assert {"nearest", "nearest_mask", "tail_transmittance"} <= set(FilterSettings.available_debug_views(oit))
	for view in ("nearest", "nearest_mask", "tail_transmittance", "transmittance", "reconstruction", "winner"):
		oit.set_debug_view(view)
		assert np.isfinite(render(renderer, (33, 25))).all()

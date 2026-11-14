"""Helpers for recording transparency passes."""


def record_transparency(commands, transparency, geometry, draw):
	"""Record banded transparency using draw(render, pass_name)."""
	with transparency.nearest_pass(commands) as render:
		draw(render, "nearest")
	transparency.shade_nearest(commands, geometry)
	with transparency.transmittance_pass(commands) as render:
		draw(render, "transmittance")
	if transparency.mixed_tail:
		with transparency.tail_statistics_pass(commands) as render:
			draw(render, "tail_statistics")
	with transparency.banded_pass(commands) as render:
		draw(render, "bands")
	transparency.shade_winner(commands, geometry)

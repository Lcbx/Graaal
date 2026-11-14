from contextlib import contextmanager

import pytest

from graaal.transparency_utils import record_transparency


@pytest.mark.parametrize("mixed", [False, True])
def test_record_transparency_sequence(mixed):
	events = []
	geometry = object()
	commands = object()
	@contextmanager
	def render_pass(*args, **kwargs):
		assert args == (commands,)
		yield object()
	class Pass:
		mixed_tail = mixed
		nearest_pass = transmittance_pass = tail_statistics_pass = banded_pass = staticmethod(render_pass)
		def shade_nearest(self, command, supplied):
			assert command is commands and supplied is geometry
			events.append("shade_nearest")
		def shade_winner(self, command, supplied):
			assert command is commands and supplied is geometry
			events.append("shade")
	record_transparency(commands, Pass(), geometry, lambda render, name: events.append(name))
	assert events == ["nearest", "shade_nearest", "transmittance", *(["tail_statistics"] if mixed else []), "bands", "shade"]

"""Offscreen sampling checks for scaled scene presentation."""

import numpy as np
import pytest

from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.RenderPasses.render_resolution import ResolutionPresenter


@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb"])
def test_resolution_presenter_samples_scene_into_larger_output(format):
	RenderContext.setup_graphics_backend(True)
	presenter = ResolutionPresenter(target_format=format)
	assert presenter.resize((2, 2))
	assert not presenter.resize((2, 2))
	pixels = np.array([[(255, 0, 0, 255), (0, 255, 0, 255)],
		[(0, 0, 255, 255), (255, 255, 255, 255)]], dtype=np.uint8)
	RenderContext.device.queue.write_texture({"texture": presenter.color.handle}, pixels,
		{"bytes_per_row": 8, "rows_per_image": 2}, (2, 2, 1))
	output = Texture((4, 4), format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands()
	presenter.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 16}, (4, 4, 1))
	image = np.frombuffer(data, dtype=np.uint8).reshape(4, 4, 4)
	np.testing.assert_array_equal(image[0, 0], pixels[0, 0])
	np.testing.assert_array_equal(image[0, -1], pixels[0, -1])
	np.testing.assert_array_equal(image[-1, 0], pixels[-1, 0])
	np.testing.assert_array_equal(image[-1, -1], pixels[-1, -1])

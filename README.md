# Graaal

Graaal is a Python game engine built around **NumPy** and **wgpu**. It provides a
NumPy-backed entity-component system (ECS), GPU drawing and culling, materials,
transparency, post-processing, and utilities for building interactive scenes.

Its tools are composable: the ECS is optional for rendering, and CPU-side tools
can be used without initializing a graphics backend. ECS-to-renderer integration
lives in separate adapters.

For the module/API overview and rendering examples, see the
**[detailed engine guide](graaal/README.md)**.

## Install and run

From the repository root (Python 3.11+):

```sh
pip install -e .
python -m graaal
```

The base installation needs only NumPy. For rendering, assets, UI, bundled
scenes, and tests:

```sh
pip install -e ".[render,assets,ui,examples,test]"
python -m graaal --offscreen --frames 1
python main.py scenes/msaa_example.py
python main.py scenes/transparency_example.py
```

The graphical commands require a working WGPU backend. See the
[scene guide](scenes/README.md) for more examples and controls.

## ECS example

```python
import numpy as np
from graaal.ECS import ECS, array_field, columns, component

@component
class Position:
	value: np.ndarray = array_field(np.float32, shape=(3,))

world = ECS()
positions, = world.register(Position)
entities = world.create(2)
world.add(entities, columns(Position, value=[[0, 0, 0], [1, 0, 0]]))
positions[entities].assign(value=np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32))
print(positions[entities].value)
```

Components store typed NumPy columns. See the [ECS API](graaal/ECS_API.md) for
queries, dense edits, versions, and scheduling.

## Rendering example

A minimal triangle using the procedural rendering API:

```python
from graaal.RenderContext import RenderContext, RenderPipeline, Shader

RenderContext.init_window(640, 480, "Graaal")
try:
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let p = array<vec2f, 3>(vec2f(-0.6, -0.6), vec2f(0.6, -0.6), vec2f(0.0, 0.6));
			return vec4f(p[i], 0.0, 1.0);
		}
		@fragment fn fs() -> @location(0) vec4f {
			return vec4f(1.0, 0.6, 0.2, 1.0);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False)
	while RenderContext.window_loop():
		commands = RenderContext.commands("triangle")
		with commands.render_pass(color=RenderContext.screen(clear=(0, 0, 0, 1))) as render:
			render.set_pipeline(pipeline)
			render.draw(3)
		RenderContext.submit(commands.finish())
finally:
	RenderContext.cleanup()
```

For materials, indirect draws, and transparency, start with the
[detailed guide](graaal/README.md) or [interactive scenes](scenes/README.md).

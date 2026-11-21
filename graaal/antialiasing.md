# Antialiasing

Implementation: [PostprocessingPass.py](RenderPasses/PostprocessingPass.py) and
[antialiasing.shaderlib](shaders/antialiasing.shaderlib). Example controls:
[rendering_settings.py](UI/rendering_settings.py). Regression tests:
[test_antialiasing.py](../tests/test_antialiasing.py). Run from the repository
root with `python -m pytest tests/test_antialiasing.py -q` (GPU backend required
for rendering tests).

`AntialiasingEffect` applies LXAA, FAAA or DDAA2 to a single-sample color target.
`AntialiasingSettings` controls each algorithm. Set `show_edges=True` to
display detected edges in green.
It runs through [`PostprocessingPass`](README.md#postprocessing), alone or with
other effects.
Set `hybrid_luma_chroma=True` to include color differences in edge detection;
`chroma_weight` controls their contribution from 0 to 5 and defaults to 0.5.

```python
from graaal.RenderPasses.PostprocessingPass import AntialiasingEffect, PostprocessingPass
from graaal.RenderPasses.pass_settings import AntialiasingSettings

aa = AntialiasingEffect("faaa", AntialiasingSettings(faaa_quality=5))
post = PostprocessingPass([aa], target_format="rgba8unorm")
post.resize((width, height))
scene_output = post.color_attachment()
# Render the scene to scene_output, then record the antialiasing pass.
post.record(commands, output)  # "lxaa" and "ddaa2" are also available
```

Call `aa.set_settings(settings)` to change the parameters and `post.resize(size)`
when output dimensions change. `record` writes to a single-sample color
attachment with the same format as the pass.

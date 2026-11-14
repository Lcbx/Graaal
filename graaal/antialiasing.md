# Antialiasing

`AntialiasingPass` applies LXAA, FAAA or DDAA2 to a single-sample color target.
`AntialiasingSettings` controls each algorithm. Set `show_edges=True` to
display detected edges in green.
Set `hybrid_luma_chroma=True` to include color differences in edge detection;
`chroma_weight` controls their contribution from 0 to 5 and defaults to 0.5.

```python
from graaal.AntialiasingPass import AntialiasingPass, AntialiasingSettings

aa = AntialiasingPass(target_format="rgba8unorm", settings=AntialiasingSettings(faaa_quality=5))
aa.resize((width, height))
scene_output = aa.color_attachment()
# Render the scene to scene_output, then record the antialiasing pass.
aa.record(commands, output, "faaa")  # "lxaa" and "ddaa2" are also available
```

Call `set_settings(settings)` to change the parameters and `resize(size)` when
the output dimensions change. `record` writes to a single-sample color
attachment with the same format as the pass.

from types import SimpleNamespace

from graaal.Materials.MaterialRenderer import record_material_frame


def test_material_frame_uses_renderer_culler_and_accepts_override(monkeypatch):
	events = []
	class Culler:
		def __init__(self, name): self.name = name
		def record(self, commands, visibility, **kwargs): events.append((self.name, kwargs))
	class Transparency:
		has_external_opaque_inputs = True
		def resize(self, size): events.append(("resize", size))
		def update_camera(self, camera, **kwargs): return "view_projection"
		def compose(self, commands, output): events.append(("compose", output))
	renderer = SimpleNamespace(sync=lambda: events.append(("sync",)), transparency=Transparency(),
		culler=Culler("default"), visibility=object(), draws=SimpleNamespace(draw_batches=[]), geometry=[None, object()])
	camera = SimpleNamespace(position=(1, 2, 3), perspective=True, fovy_deg=60)
	monkeypatch.setattr("graaal.Materials.MaterialRenderer.record_transparency", lambda *args: events.append(("transparent",)))
	record_material_frame(renderer, object(), (10, 20), camera, "output")
	assert [event[0] for event in events] == ["sync", "resize", "default", "transparent", "compose"]
	assert events[2][1] == dict(camera_position=camera.position, view_projection="view_projection", fovy_deg=60)
	events.clear()
	record_material_frame(renderer, object(), (10, 20), camera, "output", culler=Culler("override"))
	assert events[2][0] == "override"

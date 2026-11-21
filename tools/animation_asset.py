"""Load source animation channels and supported runtime clips for desktop tools."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graaal.AnimationEvents import EventTimeline, EventTrack
from graaal.Assets import read_gltf_accessor
from .animation_preview import load_preview


@dataclass(frozen=True)
class Track:
	label: str
	interpolation: str
	times: np.ndarray
	values: np.ndarray
	node: int | None = None
	path: str = ""
	tangents: np.ndarray | None = None


@dataclass
class Animation:
	name: str
	duration: float
	tracks: tuple[Track, ...]
	events: EventTimeline | None
	preview: object = None
	clip: object = None
	sync_markers: EventTimeline | None = None

	def __post_init__(self):
		duration = self.clip.duration if self.clip is not None else self.duration
		if self.sync_markers is None and duration > 0: self.sync_markers = EventTimeline(EventTrack(duration, [], []))


def load_animations(path):
	"""Read source channels and load supported runtime clips for the viewport."""
	from pygltflib import GLTF2

	path = Path(path)
	gltf = GLTF2().load(str(path))
	preview = load_preview(gltf, path)
	animations = []
	for animation_id, animation in enumerate(gltf.animations or ()):
		tracks = []
		for channel in animation.channels or ():
			if channel.sampler is None or not 0 <= channel.sampler < len(animation.samplers): raise ValueError("Animation references an invalid sampler")
			sampler = animation.samplers[channel.sampler]
			if sampler.input is None or sampler.output is None or not 0 <= sampler.input < len(gltf.accessors) or not 0 <= sampler.output < len(gltf.accessors):
				raise ValueError("Animation sampler references an invalid accessor")
			times = read_gltf_accessor(gltf, sampler.input, path.parent)
			values = read_gltf_accessor(gltf, sampler.output, path.parent)
			if times.ndim != 2 or times.shape[1] != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times[:, 0]) <= 0):
				raise ValueError("Animation key times must be finite, nonnegative and strictly increasing")
			interpolation = sampler.interpolation or "LINEAR"
			factor = 3 if interpolation == "CUBICSPLINE" else 1
			if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE") or values.size == 0 or values.size % (len(times) * factor) or not np.isfinite(values).all():
				raise ValueError("Invalid animation sampler values or interpolation")
			samples = values.reshape(len(times), factor, -1)
			tangents = samples[:, (0, 2)].copy() if factor == 3 else None
			values = samples[:, 1 if factor == 3 else 0].copy()
			times = times[:, 0].astype(np.float64, copy=True)
			times.flags.writeable = False
			values.flags.writeable = False
			target = channel.target
			if target is None: raise ValueError("Animation channel has no target")
			node = target.node
			if node is None: node_name = "Unbound target"
			elif not 0 <= node < len(gltf.nodes): raise ValueError("Animation references an invalid node")
			else: node_name = f"{gltf.nodes[node].name or 'Node'} [{node}]"
			label = f"{node_name} / {target.path or 'unknown'}"
			if target.path in ("translation", "scale", "rotation") and values.shape[1] != (4 if target.path == "rotation" else 3): raise ValueError("Invalid transform channel width")
			tracks.append(Track(label, interpolation, times, values, node, target.path or "", tangents))
		duration = max((float(track.times[-1]) for track in tracks), default=0.0)
		events = EventTimeline(EventTrack(duration, [], [])) if duration > 0 else None
		clip = preview.asset.clips[animation_id][1] if preview.asset is not None else None
		animations.append(Animation(animation.name or f"Animation {animation_id}", duration, tuple(tracks), events, preview, clip))
	return animations

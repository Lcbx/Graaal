"""Native rig settings and passive joint controls for the ragdoll tuner."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from graaal.PhysicsRagdoll import ragdoll_limit_parameters

if TYPE_CHECKING:
	import pyjoltphysics as pj


SHAPE_FIELDS = {
	"Box": ("Width, height, depth, rounding (m)", 4),
	"Sphere": ("Radius (m)", 1),
	"Capsule": ("Radius, cylinder length (m)", 2),
	"Cylinder": ("Radius, height (m)", 2),
}
GAINS = ("stiffness", "damping", "max_torque")


def _finite_rows(values, shape, name):
	rows = np.asarray(values, dtype=np.float64)
	if rows.shape != shape or not np.isfinite(rows).all():
		raise ValueError(f"{name} requires finite values with shape {shape}")
	return rows


def _unit_turns(values, count, name):
	rows = _finite_rows(values, (count, 4), name)
	if not np.allclose(np.linalg.norm(rows, axis=1), 1, rtol=0, atol=1e-6):
		raise ValueError(f"{name} requires unit XYZW quaternions")
	return rows


def _controls(values, fields, count):
	if not isinstance(values, dict) or values.keys() != set(fields):
		raise ValueError("joint controls require " + ", ".join(fields))
	result = {name: _finite_rows(values[name], (count,), name) for name in fields}
	if any(np.any(row < 0) for row in result.values()):
		raise ValueError("joint gains and torque caps must be nonnegative")
	if "margin" in result and np.any(result["margin"] == 0):
		raise ValueError("soft limit margins must be positive")
	return result


@dataclass(frozen=True)
class RagdollProject:
	settings: "pj.RagdollSettings"
	neutral: np.ndarray
	spring: dict[str, np.ndarray]
	soft_limit: dict[str, np.ndarray]

	def __post_init__(self):
		count = len(self.settings)
		parents = self.settings.parents
		if not count or parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, count)):
			raise ValueError("the ragdoll needs one root and parent-before-child bone order")
		_finite_rows(self.settings.rest_translations, (count, 3), "rest translations")
		_unit_turns(self.settings.rest_rotations, count, "rest rotations")
		_finite_rows([bone.shape_offset for bone in self.settings.bones], (count, 3), "collider offsets")
		_unit_turns([bone.shape_rotation for bone in self.settings.bones], count, "collider rotations")
		for bone in self.settings.bones:
			if not np.isfinite(bone.mass) or not 0 < bone.mass <= np.finfo(np.float32).max:
				raise ValueError(f"{bone.name}: mass must be positive and fit float32")
			kind = bone.shape.kind.name.capitalize()
			if kind not in SHAPE_FIELDS:
				raise ValueError(f"{bone.name}: use a box, sphere, capsule or cylinder")
			params = _finite_rows(bone.shape.params, (4,), f"{bone.name} collider")
			used = SHAPE_FIELDS[kind][1]
			dimensions = params[:3] if kind == "Box" else params[:used]
			if np.any(dimensions <= 0) or np.any(dimensions > np.finfo(np.float32).max):
				raise ValueError(f"{bone.name}: collider dimensions must be positive and fit float32")
			if kind == "Box" and not 0 <= params[3] <= dimensions.min() / 4:
				raise ValueError(f"{bone.name}: box rounding must be between zero and a quarter of the shortest side")
			if not np.isfinite((bone.friction, bone.strength)).all() or min(bone.friction, bone.strength) < 0:
				raise ValueError(f"{bone.name}: joint friction and motor torque must be finite and nonnegative")
		frames = ragdoll_limit_parameters(self.settings)
		twist, swing = frames["twist"], frames["swing"]
		_finite_rows(twist, (count, 2), "twist limits")
		_finite_rows(swing, (count, 2), "swing half angles")
		if np.any(twist[:, 0] > twist[:, 1]) or np.any(np.abs(twist) > np.pi + 1e-6):
			raise ValueError("twist limits must be ordered and between −180° and 180°")
		if np.any(swing < 0) or np.any(swing > np.pi + 1e-6):
			raise ValueError("swing half angles must be between 0° and 180°")
		object.__setattr__(self, "neutral", _unit_turns(self.neutral, count, "neutral rotations"))
		object.__setattr__(self, "spring", _controls(self.spring, GAINS, count))
		object.__setattr__(self, "soft_limit", _controls(self.soft_limit, ("margin", *GAINS), count))

	def limit_parameters(self):
		return ragdoll_limit_parameters(self.settings) | self.soft_limit


def default_project():
	"""Use the authored passive humanoid from the example."""
	from scenes.passive_ragdoll_example import make_humanoid

	settings, neutral, spring, limit = make_humanoid()
	soft = {key: np.broadcast_to(limit[key], (len(settings),)).copy() for key in ("margin", *GAINS)}
	return RagdollProject(settings, neutral, spring, soft)


def open_project(path):
	"""Return the project and whether its source was plain native settings."""
	import pyjoltphysics as pj

	data = json.loads(Path(path).read_text(encoding="utf-8"))
	if not isinstance(data, dict): raise ValueError("expected a JSON object for a ragdoll project")
	try:
		if "bones" in data:
			settings = pj.RagdollSettings.from_dict(data)
			spring = {key: np.zeros(len(settings)) for key in GAINS}
			soft = spring | {"margin": np.full(len(settings), 0.08)}
			return RagdollProject(settings, settings.rest_rotations, spring, soft), True
		if data.get("version") != 1 or data.keys() != {"version", "settings", "neutral", "spring", "soft_limit"}:
			raise ValueError("expected a version 1 ragdoll tuner project")
		return RagdollProject(pj.RagdollSettings.from_dict(data["settings"]), data["neutral"], data["spring"], data["soft_limit"]), False
	except (AttributeError, IndexError, KeyError, OverflowError, TypeError) as exc:
		raise ValueError(f"invalid ragdoll project: {exc}") from exc


def read_project(path):
	"""Read a tuner project or import plain RagdollSettings.as_dict() JSON."""
	return open_project(path)[0]


def write_project(path, project):
	"""Atomically save native settings and passive controls as one JSON project."""
	path = Path(path)
	if path.suffix.lower() != ".json": raise ValueError("ragdoll projects must use a .json file")
	data = {
		"version": 1,
		"settings": project.settings.as_dict(),
		"neutral": project.neutral.tolist(),
		"spring": {key: value.tolist() for key, value in project.spring.items()},
		"soft_limit": {key: value.tolist() for key, value in project.soft_limit.items()},
	}
	content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
			temporary = Path(output.name)
			output.write(content)
		os.replace(temporary, path)
	finally:
		if temporary is not None: temporary.unlink(missing_ok=True)

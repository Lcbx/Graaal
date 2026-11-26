"""Export the authored ragdoll as an approximate Genesis MJCF rig."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from graaal.Math import Quaternion
from tools.ragdoll_project import default_project, read_project
from tools.ragdoll_standing import compact_axes


def _numbers(values):
	return " ".join(f"{float(value):.12g}" for value in values)


def _quaternion(xyzw):
	return _numbers(np.asarray(xyzw)[[3, 0, 1, 2]])


def export_rig(project):
	"""Build MJCF and ordered action metadata for an approximate hinge rig.

	The hinge limits bound each axis independently; Jolt's bent swing cone is
	not equivalent. Collision filtering, motor gains and passive torques are not
	encoded here and must be validated separately before policy transfer.
	"""
	settings = project.settings
	limits = project.limit_parameters()
	bones, axes = compact_axes(limits)
	frames = Quaternion.matrix_batch(limits["child_frames"])
	root = ET.Element("mujoco", model="graaal_ragdoll")
	ET.SubElement(root, "compiler", angle="radian")
	ET.SubElement(root, "option", timestep="0.008333333333333333", gravity="0 -9.81 0")
	world = ET.SubElement(root, "worldbody")
	bodies = []
	actions = []
	for index, bone in enumerate(settings.bones):
		parent = world if index == 0 else bodies[int(settings.parents[index])]
		body = ET.SubElement(parent, "body", name=bone.name, pos=_numbers(settings.rest_translations[index]), quat=_quaternion(settings.rest_rotations[index]))
		bodies.append(body)
		if index == 0:
			ET.SubElement(body, "freejoint", name="root")
		else:
			for action_index in np.flatnonzero(bones == index):
				axis = axes[action_index]
				axis_index = int(np.argmax(np.abs(frames[index].T @ axis)))
				bounds = limits["twist"][index] if axis_index == 0 else np.array([-1, 1]) * limits["swing"][index, axis_index - 1]
				name = f"{bone.name}_{axis_index}"
				ET.SubElement(body, "joint", name=name, type="hinge", axis=_numbers(axis), limited="true", range=_numbers(bounds))
				actions.append(dict(name=name, bone=bone.name, axis=axis.tolist(), axis_index=axis_index, limits=bounds.tolist()))
		kind = bone.shape.kind.name
		params = np.asarray(bone.shape.params)
		shape_rotation = np.asarray(bone.shape_rotation)
		if kind == "BOX":
			shape, size = "box", params[:3] / 2
		elif kind == "SPHERE":
			shape, size = "sphere", params[:1]
		elif kind in ("CAPSULE", "CYLINDER"):
			shape, size = kind.lower(), [params[0], params[1] / 2]
			shape_rotation = Quaternion.multiply_batch(shape_rotation, Quaternion.from_x_rotation(np.pi / 2))
		else:
			raise ValueError(f"unsupported collider {kind} on {bone.name}")
		ET.SubElement(body, "geom", name=f"{bone.name}_shape", type=shape, size=_numbers(size), pos=_numbers(bone.shape_offset), quat=_quaternion(shape_rotation), mass=str(float(bone.mass)))
	return ET.tostring(root, encoding="unicode"), actions


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("output", type=Path)
	parser.add_argument("--tuning", type=Path)
	args = parser.parse_args()
	project = default_project() if args.tuning is None else read_project(args.tuning)
	xml, actions = export_rig(project)
	args.output.write_text(xml + "\n", encoding="utf-8")
	print(f"Wrote {len(actions)} hinge actions to {args.output}")


if __name__ == "__main__":
	main()

"""Retarget the example FBX clips with Blender's background Python runner."""

import json
from pathlib import Path
from struct import unpack

import bpy
from mathutils import Quaternion, Vector
import numpy as np


RESOURCES = Path(__file__).resolve().parents[1] / "scenes" / "resources"
SPINE = ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")
ARM = ("Shoulder", "Arm", "ForeArm", "Hand")
FINGERS = tuple(f"Hand{finger}{joint}" for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for joint in range(1, 4))
LEG = ("UpLeg", "Leg", "Foot", "ToeBase")
SOURCE_BONES = SPINE + tuple(side + bone for group in (ARM + FINGERS, LEG) for side in ("Left", "Right") for bone in group)


def target_rest():
	with (RESOURCES / "dummy_rigged.glb").open("rb") as file:
		file.seek(12)
		length, chunk_type = unpack("<I4s", file.read(8))
		if chunk_type != b"JSON": raise ValueError("Expected a GLB JSON chunk")
		gltf = json.loads(file.read(length))
	nodes = gltf["nodes"]
	joints = gltf["skins"][0]["joints"]
	names = [f"bone_{i}" for i in range(len(SOURCE_BONES))]
	if [nodes[j]["name"] for j in joints] != names:
		raise ValueError("Dummy joint order changed")
	parents = {child: parent for parent in joints for child in nodes[parent].get("children", ())}
	local, model = {}, {}
	for joint in joints:
		x, y, z, w = nodes[joint].get("rotation", (0, 0, 0, 1))
		local[joint] = Quaternion((w, x, y, z))
		model[joint] = model[parents[joint]] @ local[joint] if joint in parents else local[joint]
	return names, [local[j] for j in joints], [model[j] for j in joints], Vector(nodes[joints[0]]["translation"])


def bake(filename, local_rest, model_rest, hips_rest):
	bpy.ops.wm.read_factory_settings(use_empty=True)
	bpy.ops.import_scene.fbx(filepath=str(RESOURCES / filename))
	armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
	scene = bpy.context.scene
	start, stop = map(round, armature.animation_data.action.frame_range)
	to_gltf = Quaternion((1, 0, 0), -np.pi / 2)
	bones = [armature.pose.bones[f"mixamorig1:{name}"] for name in SOURCE_BONES]
	source_world = to_gltf @ armature.matrix_world.to_quaternion()
	# Convert each target bone's rest axes to the corresponding FBX bone's axes.
	bases = [(source_world @ bone.bone.matrix_local.to_quaternion()).inverted() @ target for bone, target in zip(bones, model_rest)]
	hips = bones[0]
	source_rest = to_gltf @ (armature.matrix_world @ hips.bone.head_local)
	height_ratio = hips_rest.y / source_rest.y
	rotation = np.empty((stop - start + 1, len(bones), 4), dtype=np.float32)
	translation = np.empty((len(rotation), 3), dtype=np.float32)
	for row, frame in enumerate(range(start, stop + 1)):
		scene.frame_set(frame)
		translation[row] = hips_rest + height_ratio * (to_gltf @ (armature.matrix_world @ hips.head) - source_rest)
		for index, (bone, rest, basis) in enumerate(zip(bones, local_rest, bases)):
			q = rest @ basis.inverted() @ bone.matrix_basis.to_quaternion() @ basis
			rotation[row, index] = (q.x, q.y, q.z, q.w)
	return translation, rotation, np.float32(scene.render.fps / scene.render.fps_base)


def main():
	names, *rest = target_rest()
	data = {"joints": names}
	for name in ("Walking", "Running"):
		for key, array in zip(("hips", "rotation", "fps"), bake(f"{name}.fbx", *rest)):
			data[f"{name}_{key}"] = array
	np.savez_compressed(RESOURCES / "dummy_animations.npz", **data)


if __name__ == "__main__":
	main()

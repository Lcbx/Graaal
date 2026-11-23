"""Fixed-tick native preview and CPU collider drawing for the ragdoll tuner."""

import numpy as np

from graaal.Math import Camera, Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques
from scenes.passive_ragdoll_example import terrain_boxes
from scenes.physics_geometry import collider_bounds, collider_poses, cube_vertices, cylinder_vertices, rounded_vertices
from .ragdoll_preview_raster import PreviewRaster


TICK_HZ = 120
SCENARIOS = {"Drop": "drop", "Zero gravity": "zero_gravity", "Stairs": "stairs"}


def _surface(cube, shape):
	kind = shape.kind.name
	if kind == "BOX":
		return cube["position"].reshape(-1, 3, 3) * shape.params[:3]
	radius, length = shape.params[:2]
	if kind == "CYLINDER":
		return cylinder_vertices(radius, length, segments=10)["position"].reshape(-1, 3, 3)
	return rounded_vertices(cube, radius, length if kind == "CAPSULE" else 0, segments=10, rings=2)["position"].reshape(-1, 3, 3)


class RagdollPreview:
	"""Own one native world. Rebuild from the same starting pose to compare edits."""

	def __init__(self, project, scenario):
		import pyjoltphysics as pj
		from pyjoltphysics.ragdolls import forward_kinematics

		self.project = project
		self.parents = project.settings.parents
		self.limits = project.limit_parameters()
		self.scenario = scenario = SCENARIOS[scenario]
		settings = project.settings
		origin, identity = (0, 0, 0), (0, 0, 0, 1)
		rest_points, rest_turns = forward_kinematics(settings, None, origin, identity)
		span = float(np.ptp(rest_points, axis=0).max())
		if span == 0:
			lower, upper = collider_bounds(settings.bones, rest_points, rest_turns)
			span = float((upper.max(axis=0) - lower.min(axis=0)).max())
		self.scale = max(span, 0.1) / 1.8
		positions, sizes = terrain_boxes(scenario, self.scale)
		self.terrain = positions, sizes
		turn = Quaternion.from_z_rotation({"drop": 0.35, "stairs": 0.7, "zero_gravity": 0}[scenario])
		points, turns = forward_kinematics(settings, None, origin, turn)
		lower, upper = collider_bounds(settings.bones, points, turns)
		spawn = np.zeros(3)
		if scenario != "zero_gravity":
			tops = positions[:, 1] + sizes[:, 1] / 2
			top_step = int(np.argmax(tops))
			lowest = int(np.argmin(lower[:, 1]))
			if scenario == "stairs":
				spawn = positions[top_step] - (lower[lowest] + upper[lowest]) / 2
			else:
				spawn[0] = -(lower[:, 0].min() + upper[:, 0].max()) / 2
			spawn[1] = tops[top_step] - lower[lowest, 1] + (0.85 if scenario == "drop" else 0.02) * self.scale
		gravity = (0, 0, 0) if scenario == "zero_gravity" else (0, -9.81, 0)
		self.world = pj.World(pj.WorldSettings(tick_hz=TICK_HZ, threads=1, gravity=gravity, allow_sleeping=False))
		try:
			if len(positions):
				self.world.add_bodies(pj.BodyBatch.boxes(positions, size=sizes, motion=pj.MotionType.STATIC, friction=0.8))
			self.handle = self.world.add_ragdoll(settings, spawn, turn, mode=pj.RagdollMode.LIMP)
			self.uids = self.handle.bones
			if scenario == "stairs":
				self.world.set_velocities(self.uids, linear=np.broadcast_to([-1.2 * self.scale, 0, 0], (len(self.uids), 3)))
			self.world.flush()
			self.state = self._read_state()
		except BaseException:
			self.world.stop()
			raise

	def _read_state(self):
		state = self.world.bodies_state(self.uids)
		if not state.found.all() or not all(np.isfinite(value).all() for value in (state.position, state.orientation, state.angular_velocity, state.linear_velocity)):
			raise RuntimeError("preview lost a bone or received nonfinite physics state")
		return state

	def step(self, ticks=1, *, passive=True):
		for _ in range(ticks):
			if passive:
				turns, angular = self.state.orientation[None], self.state.angular_velocity[None]
				torque = joint_spring_torques(self.parents, turns, angular, self.project.neutral, **self.project.spring)
				torque += joint_limit_torques(orientations=turns, angular_velocity=angular, **self.limits)
				self.world.add_torques(self.uids, torque[0])
			self.world.run_ticks(1)
			self.world.drain_events()
			self.state = self._read_state()

	def push(self, bone, impulse):
		self.handle.hit(bone, impulse)
		self.world.flush()
		self.state = self._read_state()

	def bounds(self):
		lower, upper = collider_bounds(self.project.settings.bones, self.state.position, self.state.orientation)
		return lower.min(axis=0), upper.max(axis=0)

	def close(self):
		self.world.stop()


class ColliderViewport:
	"""Draw collider faces and terrain with one shared depth and picking buffer."""

	def __init__(self, preview):
		cube = cube_vertices()
		triangles, owners = [], []
		bones = preview.project.settings.bones
		self.offsets = np.array([bone.shape_offset for bone in bones])
		self.shape_rotations = np.array([bone.shape_rotation for bone in bones])
		for i, bone in enumerate(bones):
			faces = _surface(cube, bone.shape)
			triangles.append(faces)
			owners.extend([i] * len(faces))
		self.faces = np.concatenate(triangles)
		self.owners = np.array(owners, dtype=np.intp)
		terrain_positions, terrain_sizes = preview.terrain
		self.stairs = (cube["position"].reshape(-1, 3, 3) * terrain_sizes[1:, None, None, :] + terrain_positions[1:, None, None, :]).reshape(-1, 3, 3)
		self.links = np.column_stack((preview.parents[1:], np.arange(1, len(preview.parents))))
		self.raster = PreviewRaster()
		self.frame(preview, include_ground=True)

	def frame(self, preview, *, include_ground=False):
		lower, upper = preview.bounds()
		if include_ground and len(preview.terrain[0]):
			lower[1] = min(0, lower[1])
			if len(self.stairs):
				lower[0] = min(lower[0], self.stairs[..., 0].min())
				upper[0] = max(upper[0], self.stairs[..., 0].max())
		self.target = (lower + upper) / 2
		self.radius = max(float(np.linalg.norm(upper - lower)) / 2, preview.scale * 0.3)
		x, y, z = (-0.35, 0.45, 1.68) if preview.scenario == "stairs" else (1.1, 0.7, 2.0)
		self.yaw, self.pitch, self.zoom = np.arctan2(x, z), np.arctan2(y, np.hypot(x, z)), 1.0

	def render(self, preview, selected, width, height):
		direction = np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw) * np.cos(self.pitch)])
		framing = self.radius * 3 / min(width / height, 1)
		camera = self.target + direction * framing * self.zoom
		basis = Camera.calc_view(camera, self.target, (0, 1, 0))[:3, :3] * [1, 1, -1]
		self.raster.clear(width, height, near=0.02 * preview.scale)
		positions, turns = preview.state.position, preview.state.orientation
		collider_centers, rotations = collider_poses(positions, turns, self.offsets, self.shape_rotations)
		axes = Quaternion.matrix_batch(rotations)
		body_faces = self.faces @ axes[self.owners].transpose(0, 2, 1) + collider_centers[self.owners, None]
		faces = np.concatenate((body_faces, self.stairs))
		centers = faces.mean(axis=1)
		normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
		length = np.linalg.norm(normal, axis=1)
		visible = (length > 1e-12) & (np.sum(normal * (camera - centers), axis=1) > 0)
		indices = np.flatnonzero(visible)
		light = np.array([0.34, 0.82, 0.46])
		brightness = 0.45 + 0.55 * np.maximum(0, normal[indices] @ light / length[indices])
		owners = np.concatenate((self.owners, np.full(len(self.stairs), -1)))[indices]
		color = np.where(owners[:, None] < 0, [85, 101, 114], np.where(owners[:, None] == selected, [255, 189, 99], [109, 184, 211]))
		if len(preview.terrain[0]):
			self.raster.lines((self._floor(preview.scale) - camera) @ basis, (34, 49, 58))
		self.raster.triangles((faces[indices] - camera) @ basis, (color * brightness[:, None]).astype(np.uint8), owners)
		joints = (positions - camera) @ basis
		self.raster.lines(joints[self.links], (162, 186, 197), joints=self.links)
		self.raster.joint_marker(joints[selected], selected)
		return self.raster.rgb

	def bone_at(self, x, y):
		width, height = self.raster.size
		if 0 <= x < width and 0 <= y < height:
			owner = int(self.raster.owners[y, x])
			if owner >= 0: return owner
		return None

	def _floor(self, scale):
		spacing = scale / 2
		extent = int(np.clip(np.ceil(self.radius / spacing), 7, 32))
		steps = (np.arange(-extent, extent + 1) + np.round(self.target[[0, 2], None] / spacing)) * spacing
		x, z = steps
		along_z = np.stack((np.repeat(x[:, None], 2, axis=1), np.zeros((len(x), 2)), np.tile(z[[0, -1]], (len(x), 1))), axis=-1)
		along_x = np.stack((np.tile(x[[0, -1]], (len(z), 1)), np.zeros((len(z), 2)), np.repeat(z[:, None], 2, axis=1)), axis=-1)
		return np.concatenate((along_z, along_x))

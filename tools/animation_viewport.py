"""Shared CPU viewport drawing for the timeline and graph editors."""
import numpy as np

from .animation_preview import camera_from_bounds, mesh_triangles, pose_matrices, project


def draw_animation_viewport(canvas, animation, time, bounds, *, joint_model=None, show_mesh=True, show_bones=True):
	canvas.delete("all")
	if animation is None or not len(animation.preview.nodes): return None
	width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
	try:
		model = pose_matrices(animation, time, joint_model=joint_model)
		camera_position, camera_target = camera_from_bounds(bounds, width / height)
		points = model[:, :3, 3]
		screen, depth = project(points, camera_position, camera_target, width, height)
		if show_mesh:
			triangles = mesh_triangles(animation.preview, model)
			if len(triangles):
				vertices, distances = project(triangles.reshape(-1, 3), camera_position, camera_target, width, height)
				vertices, distances = vertices.reshape(-1, 6), distances.reshape(-1, 3)
				normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
				normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
				brightness = 0.35 + 0.65 * np.abs(normal @ (np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])))
				for face in np.argsort(distances.mean(axis=1))[::-1]:
					if np.any(distances[face] <= 0): continue
					color = tuple((np.array([100, 185, 125]) * brightness[face]).astype(int))
					canvas.create_polygon(*vertices[face], fill="#%02x%02x%02x" % color, outline="")
		if show_bones:
			bones = set(map(int, animation.preview.bones))
			for bone in bones:
				parent = int(animation.preview.parents[bone])
				while parent != -1 and parent not in bones: parent = int(animation.preview.parents[parent])
				if parent != -1 and min(depth[bone], depth[parent]) > 0: canvas.create_line(*screen[parent], *screen[bone], fill="#f4c96a", width=2)
				if depth[bone] > 0:
					x, y = screen[bone]
					canvas.create_oval(x-3, y-3, x+3, y+3, fill="#ffe3a3", outline="")
		canvas.create_text(8, 8, anchor="nw", fill="#bbc6d5", width=max(width-16, 1), text=animation.preview.message)
		return camera_position, camera_target
	except (ValueError, IndexError, FloatingPointError) as exc:
		canvas.create_text(8, 8, anchor="nw", fill="#ff9aa5", text=f"Preview unavailable: {exc}", width=max(width-16, 1))
		return None

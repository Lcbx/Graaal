"""Depth-tested CPU drawing for the ragdoll tuner's Tk viewport."""

import numpy as np


def _clip_triangle(points, near):
	outline = []
	for a, b in zip(np.roll(points, 1, axis=0), points):
		a_inside, b_inside = a[2] >= near, b[2] >= near
		if a_inside != b_inside:
			outline.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
		if b_inside:
			outline.append(b)
	return [(outline[0], outline[i], outline[i + 1]) for i in range(1, len(outline) - 1)]


class PreviewRaster:
	def __init__(self):
		self.size = (0, 0)

	def clear(self, width, height, near):
		if self.size != (width, height):
			self.size = (width, height)
			self.rgb = np.empty((height, width, 3), dtype=np.uint8)
			self.depth = np.empty((height, width), dtype=np.float32)
			self.owners = np.empty((height, width), dtype=np.int32)
			self.x, self.y = np.arange(width) + 0.5, np.arange(height)[:, None] + 0.5
			self.focal = height * 0.5 / np.tan(np.deg2rad(22.5))
			self.center = np.array(self.size) * 0.5
		self.near = near
		self.rgb[:] = (19, 30, 37)
		self.depth.fill(0)
		self.owners.fill(-1)

	def project(self, points):
		return points[..., :2] / np.maximum(points[..., 2:3], self.near) * [self.focal, -self.focal] + self.center

	def triangles(self, faces, colors, owners):
		inside = faces[..., 2] >= self.near
		for i in np.flatnonzero(inside.any(axis=1) & ~inside.all(axis=1)):
			clipped = np.asarray(_clip_triangle(faces[i], self.near))
			self._fill(self.project(clipped), 1 / clipped[..., 2], np.broadcast_to(colors[i], (len(clipped), 3)), np.full(len(clipped), owners[i]))
		complete = inside.all(axis=1)
		faces = faces[complete]
		self._fill(self.project(faces), 1 / faces[..., 2], colors[complete], owners[complete])

	def _fill(self, triangles, depths, colors, owners):
		lower = np.clip(np.ceil(triangles.min(axis=1) - 0.5), 0, self.size).astype(np.intp)
		upper = np.clip(np.floor(triangles.max(axis=1) + 0.5), 0, self.size).astype(np.intp)
		edges = triangles[:, 1:] - triangles[:, :1]
		ab, ac = edges[:, 0], edges[:, 1]
		area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
		active = np.flatnonzero((upper > lower).all(axis=1) & (np.abs(area) > 1e-10))
		for i in active:
			x0, y0 = lower[i]
			x1, y1 = upper[i]
			origin = triangles[i, 0]
			dx = self.x[x0:x1] - origin[0]
			dy = self.y[y0:y1] - origin[1]
			u = (ac[i, 1] * dx - ac[i, 0] * dy) / area[i]
			v = (ab[i, 0] * dy - ab[i, 1] * dx) / area[i]
			# Reciprocal view depth is linear across the projected triangle.
			z0, z1, z2 = depths[i]
			depth = (z0 + (z1 - z0) * u + (z2 - z0) * v).astype(np.float32)
			box = np.s_[y0:y1, x0:x1]
			pixels = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (depth > self.depth[box])
			self.depth[box][pixels] = depth[pixels]
			self.rgb[box][pixels] = colors[i]
			self.owners[box][pixels] = owners[i]

	def lines(self, lines, color, *, joints=None):
		for i in np.flatnonzero((lines[..., 2] >= self.near).any(axis=1)):
			a, b = lines[i]
			if a[2] < self.near: a = a + (b - a) * ((self.near - a[2]) / (b[2] - a[2]))
			if b[2] < self.near: b = b + (a - b) * ((self.near - b[2]) / (a[2] - b[2]))
			start, end = self.project(np.stack((a, b)))
			delta = end - start
			axis = int(np.argmax(np.abs(delta)))
			if abs(delta[axis]) < 1e-8: continue
			lo, hi = np.clip(np.floor(sorted((start[axis], end[axis]))) + [0, 1], 0, self.size[axis]).astype(np.intp)
			t = np.clip((np.arange(lo, hi) + 0.5 - start[axis]) / delta[axis], 0, 1)
			x, y = np.floor(start + t[:, None] * delta).astype(np.intp).T
			valid = (x >= 0) & (x < self.size[0]) & (y >= 0) & (y < self.size[1])
			x, y, t = x[valid], y[valid], t[valid]
			depth = (1 - t) / a[2] + t / b[2]
			visible = depth > self.depth[y, x]
			if joints is None:
				self.depth[y[visible], x[visible]] = depth[visible]
			else:
				parent, child = joints[i]
				owners = self.owners[y, x]
				visible = (visible | (owners == parent) | (owners == child)) & (np.arange(len(t)) % 6 < 3)
			self.rgb[y[visible], x[visible]] = color

	def joint_marker(self, point, owner):
		if point[2] < self.near: return
		x, y = self.project(point)
		if not (0 <= x < self.size[0] and 0 <= y < self.size[1]): return
		x0, y0 = np.maximum((0, 0), (int(x) - 4, int(y) - 4))
		x1, y1 = np.minimum(self.size, (int(x) + 5, int(y) + 5))
		radius_sq = (self.x[x0:x1] - x)**2 + (self.y[y0:y1] - y)**2
		box = np.s_[y0:y1, x0:x1]
		visible = (self.depth[box] < 1 / point[2]) | (self.owners[box] == owner)
		self.rgb[box][(radius_sq >= 9) & (radius_sq <= 20) & visible] = (255, 242, 195)

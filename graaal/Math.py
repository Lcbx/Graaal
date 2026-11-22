from __future__ import annotations

from typing import Any, Self, ClassVar

from numpy.typing import ArrayLike, DTypeLike
import numpy as np

from .utils import cache_last


class Quaternion(np.ndarray):
	"""An xyzw quaternion, with single-value and vectorized operations."""
	def __new__(cls, value: ArrayLike | None=None, dtype: DTypeLike | None=None) -> Self:
		if value is None: value = [0.0, 0.0, 0.0, 1.0]
		array = np.asarray(value, dtype=dtype)
		if array.shape in ((3, 3), (4, 4)):
			array = cls.from_matrices_batch(array[None])[0]
		if array.shape != (4,): raise ValueError("quaternion must have four components")
		if array.dtype.kind != "f": array = array.astype(np.float64)
		return np.array(array, copy=True).view(cls)

	x = property(lambda self: self[0], lambda self, value: self.__setitem__(0, value))
	y = property(lambda self: self[1], lambda self, value: self.__setitem__(1, value))
	z = property(lambda self: self[2], lambda self, value: self.__setitem__(2, value))
	w = property(lambda self: self[3], lambda self, value: self.__setitem__(3, value))
	xy = property(lambda self: self[:2], lambda self, value: self.__setitem__(slice(0, 2), value))
	xyz = property(lambda self: self[:3], lambda self, value: self.__setitem__(slice(0, 3), value))
	xyzw = property(lambda self: self[:], lambda self, value: self.__setitem__(slice(None), value))
	xz = property(lambda self: self[[0, 2]], lambda self, value: self.__setitem__([0, 2], value))
	xw = property(lambda self: self[[0, 3]], lambda self, value: self.__setitem__([0, 3], value))
	xyw = property(lambda self: self[[0, 1, 3]], lambda self, value: self.__setitem__([0, 1, 3], value))
	xzw = property(lambda self: self[[0, 2, 3]], lambda self, value: self.__setitem__([0, 2, 3], value))

	@classmethod
	def from_axis_rotation(cls, axis: ArrayLike, theta: float, dtype: DTypeLike | None=None) -> Self:
		axis = np.asarray(axis, dtype=dtype or np.float64)
		if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(theta): raise ValueError("axis and angle must be finite")
		length = np.linalg.norm(axis)
		if length == 0: raise ValueError("axis must be nonzero")
		return cls(np.append(axis / length * np.sin(theta / 2), np.cos(theta / 2)), dtype=dtype)

	@classmethod
	def from_axis(cls, axis: ArrayLike, dtype: DTypeLike | None=None) -> Self:
		axis = np.asarray(axis, dtype=dtype or np.float64)
		length = np.linalg.norm(axis)
		return cls(dtype=dtype) if length == 0 else cls.from_axis_rotation(axis, length, dtype=dtype)

	@classmethod
	def from_x_rotation(cls, theta: float, dtype: DTypeLike | None=None) -> Self: return cls.from_axis_rotation((1, 0, 0), theta, dtype=dtype)
	@classmethod
	def from_y_rotation(cls, theta: float, dtype: DTypeLike | None=None) -> Self: return cls.from_axis_rotation((0, 1, 0), theta, dtype=dtype)
	@classmethod
	def from_z_rotation(cls, theta: float, dtype: DTypeLike | None=None) -> Self: return cls.from_axis_rotation((0, 0, 1), theta, dtype=dtype)
	@classmethod
	def from_matrix(cls, matrix: ArrayLike, dtype: DTypeLike | None=None) -> Self: return cls(cls.from_matrices_batch(np.asarray(matrix)[None])[0], dtype=dtype)

	@classmethod
	def from_eulers(cls, eulers: ArrayLike, dtype: DTypeLike | None=None) -> Self:
		roll, pitch, yaw = np.asarray(eulers, dtype=np.float64) / 2
		sr, sp, sy = np.sin([roll, pitch, yaw])
		cr, cp, cy = np.cos([roll, pitch, yaw])
		return cls([sr*cp*cy + cr*sp*sy, cr*sp*cy - sr*cp*sy, cr*cp*sy + sr*sp*cy, cr*cp*cy - sr*sp*sy], dtype=dtype)

	@classmethod
	def from_inverse_of_eulers(cls, eulers: ArrayLike, dtype: DTypeLike | None=None) -> Self:
		roll, pitch, yaw = np.asarray(eulers, dtype=np.float64) / 2
		sr, sp, sy = np.sin([roll, pitch, yaw])
		cr, cp, cy = np.cos([roll, pitch, yaw])
		return cls([cy*sp*cr + sy*cp*sr, -cy*sp*sr + sy*cp*cr, -sy*sp*cr + cy*cp*sr, cy*cp*cr + sy*sp*sr], dtype=dtype)

	@property
	def length(self) -> float: return np.linalg.norm(np.asarray(self))
	def normalize(self) -> None: self[:] = self.normalized
	def normalise(self) -> None: self.normalize()
	@property
	def normalized(self) -> Quaternion:
		if self.length == 0: raise ValueError("cannot normalize a zero quaternion")
		return Quaternion(np.asarray(self) / self.length)
	@property
	def normalised(self) -> Quaternion: return self.normalized
	@property
	# Quaternions expose conjugation as a value rather than ndarray's method.
	def conjugate(self) -> Quaternion: return Quaternion(np.asarray(self) * [-1, -1, -1, 1])  # type: ignore[override]
	@property
	def inverse(self) -> Quaternion:
		length = self.length
		if length == 0: raise ValueError("cannot invert a zero quaternion")
		return Quaternion(np.asarray(self.conjugate) / (length * length))
	@property
	def negative(self) -> Quaternion: return Quaternion(-np.asarray(self))
	@property
	def angle(self) -> float: return float(2 * np.arccos(np.clip(self[3], -1, 1)))
	@property
	def axis(self) -> np.ndarray:
		denominator = np.sqrt(max(0.0, 1 - float(self[3]) ** 2))
		return np.asarray(self[:3]) / denominator if denominator else np.array([0, 0, -1], dtype=self.dtype)
	@property
	def is_identity(self) -> bool: return bool(np.allclose(np.asarray(self), [0, 0, 0, 1]))
	@property
	def matrix33(self) -> np.ndarray: return self.matrix_batch(np.asarray(self.normalized))
	@property
	def matrix44(self) -> np.ndarray:
		result = np.eye(4, dtype=self.dtype)
		result[:3, :3] = self.matrix33
		return result
	def cross(self, other: Any) -> Quaternion: return Quaternion(self.multiply_batch(np.asarray(self), np.asarray(other)))
	# This quaternion operation has no ndarray output-buffer argument.
	def dot(self, other: Any) -> Any: return np.dot(np.asarray(self), np.asarray(other))  # type: ignore[override]
	def lerp(self, other: Any, t: float) -> Quaternion:
		t = np.clip(t, 0, 1)
		return Quaternion(np.asarray(self) * (1 - t) + np.asarray(other) * t).normalized
	def slerp(self, other: Any, t: float) -> Quaternion: return Quaternion(self.slerp_batch(np.asarray(self), np.asarray(other), np.clip(t, 0, 1)))
	def __mul__(self, other: Any) -> Any:
		value = np.asarray(other)
		if value.shape == (4,): return self.cross(value)
		if value.shape == (3,):
			rotated = self.matrix33 @ value
			return Vec3(rotated) if isinstance(other, Vec3) else rotated
		return NotImplemented
	def __or__(self, other: Any) -> Any: return self.dot(other)
	def __invert__(self) -> Quaternion: return self.conjugate
	def __eq__(self, other: Any) -> Any:
		other = np.asarray(other)
		if self.shape != (4,) or other.shape != (4,): return np.asarray(self) == other
		if np.array_equal(np.asarray(self), other): return True
		return bool(other.dtype.kind in "iufc" and np.array_equal(np.asarray(self), -other))
	def __ne__(self, other: Any) -> Any:
		result = self.__eq__(other)
		return not result if isinstance(result, (bool, np.bool_)) else np.logical_not(result)
	def exp(self) -> Quaternion:
		vector = np.asarray(self[:3])
		length = np.linalg.norm(vector)
		factor = np.sin(length) / length if length else 1.0
		return Quaternion(np.exp(self[3]) * np.append(vector * factor, np.cos(length)))
	def power(self, exponent: float) -> Quaternion:
		unit = self.normalized
		angle = np.arccos(np.clip(unit[3], -1, 1))
		sine = np.sin(angle)
		if abs(sine) < 1e-12: return Quaternion()
		return Quaternion(np.append(np.asarray(unit[:3]) * np.sin(angle * exponent) / sine, np.cos(angle * exponent)))
	@staticmethod
	def multiply_batch(a: ArrayLike, b: ArrayLike) -> np.ndarray:
		"""Multiply broadcastable arrays of xyzw quaternions."""
		a, b = np.asarray(a), np.asarray(b)
		if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != 4 or b.shape[-1] != 4: raise ValueError("quaternions must have four components")
		vector = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3])
		scalar = a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True)
		return np.concatenate((vector, scalar), axis=-1)

	@staticmethod
	def nlerp_batch(a: ArrayLike, b: ArrayLike, weight: ArrayLike, *, out: np.ndarray | None=None) -> np.ndarray:
		"""Shortest-path normalized interpolation of batched unit quaternions."""
		a, b = np.asarray(a), np.asarray(b)
		if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != 4 or b.shape[-1] != 4: raise ValueError("quaternions must have four components")
		dtype = np.result_type(a.dtype, b.dtype, np.float32)
		a, b, weight = np.broadcast_arrays(a.astype(dtype, copy=False), b.astype(dtype, copy=False), np.asarray(weight))
		b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -b, b)
		if out is None: out = np.empty_like(a)
		np.multiply(a, 1 - weight, out=out)
		out += b * weight
		out /= np.linalg.norm(out, axis=-1, keepdims=True)
		return out

	@staticmethod
	def slerp_batch(a: ArrayLike, b: ArrayLike, weight: ArrayLike) -> np.ndarray:
		"""Shortest-path spherical interpolation of batched unit quaternions."""
		a, b = np.asarray(a), np.asarray(b)
		if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != 4 or b.shape[-1] != 4: raise ValueError("quaternions must have four components")
		a, b, weight = np.broadcast_arrays(a, b, np.asarray(weight))
		dot = np.sum(a * b, axis=-1, keepdims=True)
		b = np.where(dot < 0, -b, b)
		angle = np.arccos(np.clip(np.abs(dot), 0, 1))
		sine = np.sin(angle)
		denominator = np.where(sine > 1e-6, sine, 1)
		result = np.where(sine > 1e-6, np.sin((1 - weight) * angle) / denominator * a + np.sin(weight * angle) / denominator * b, a * (1 - weight) + b * weight)
		return result / np.linalg.norm(result, axis=-1, keepdims=True)

	@staticmethod
	def matrix_batch(rotation: ArrayLike) -> np.ndarray:
		"""Convert batched unit xyzw quaternions to column-vector 3x3 matrices."""
		rotation = np.asarray(rotation)
		if rotation.ndim == 0 or rotation.shape[-1] != 4: raise ValueError("quaternions must have four components")
		rotation = rotation.astype(np.result_type(rotation.dtype, np.float32), copy=False)
		x, y, z, w = np.moveaxis(rotation, -1, 0)
		result = np.empty(rotation.shape[:-1] + (3, 3), dtype=rotation.dtype)
		result[..., 0, 0] = 1 - 2 * (y*y + z*z)
		result[..., 0, 1] = 2 * (x*y - z*w)
		result[..., 0, 2] = 2 * (x*z + y*w)
		result[..., 1, 0] = 2 * (x*y + z*w)
		result[..., 1, 1] = 1 - 2 * (x*x + z*z)
		result[..., 1, 2] = 2 * (y*z - x*w)
		result[..., 2, 0] = 2 * (x*z - y*w)
		result[..., 2, 1] = 2 * (y*z + x*w)
		result[..., 2, 2] = 1 - 2 * (x*x + y*y)
		return result

	@staticmethod
	def from_matrices_batch(matrices: ArrayLike) -> np.ndarray:
		"""Convert a batch of column-vector rotation matrices to xyzw arrays."""
		matrices = np.asarray(matrices, dtype=np.float64)
		if matrices.ndim != 3 or matrices.shape[1:] not in ((3, 3), (4, 4)):
			raise ValueError("matrices must have shape (instances, 3, 3) or (instances, 4, 4)")
		m = matrices[..., :3, :3]
		diagonal = np.stack((1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]), axis=1)
		choice = np.argmax(diagonal, axis=1)
		result = np.empty((len(m), 4), dtype=np.float64)
		for branch in range(4):
			selected = choice == branch
			part = m[selected]
			s = 2 * np.sqrt(np.maximum(diagonal[selected, branch], 0))
			if branch == 0:
				values = ((part[:, 2, 1] - part[:, 1, 2]) / s, (part[:, 0, 2] - part[:, 2, 0]) / s, (part[:, 1, 0] - part[:, 0, 1]) / s, s / 4)
			elif branch == 1:
				values = (s / 4, (part[:, 0, 1] + part[:, 1, 0]) / s, (part[:, 0, 2] + part[:, 2, 0]) / s, (part[:, 2, 1] - part[:, 1, 2]) / s)
			elif branch == 2:
				values = ((part[:, 0, 1] + part[:, 1, 0]) / s, s / 4, (part[:, 1, 2] + part[:, 2, 1]) / s, (part[:, 0, 2] - part[:, 2, 0]) / s)
			else:
				values = ((part[:, 0, 2] + part[:, 2, 0]) / s, (part[:, 1, 2] + part[:, 2, 1]) / s, s / 4, (part[:, 1, 0] - part[:, 0, 1]) / s)
			result[selected] = np.stack(values, axis=1)
		return result / np.linalg.norm(result, axis=1, keepdims=True)


def _array_accessor(index:int=0) -> property:
	return property(
		lambda v: v[index],
		lambda v, y: v.__setitem__(index, y),
	)


class _baseVector(np.ndarray):
	_count: ClassVar[int]
	_base_dtype: ClassVar[np.dtype] = np.dtype(np.float32)

	def __new__(cls, *args: Any) -> Self:
		values = args[0] if len(args) == 1 and not np.isscalar(args[0]) else args
		if len(values) != cls._count:
			raise ValueError(f'got passed {len(values)} when {cls._count} were expected')
		return np.asarray(values, dtype=cls._base_dtype).view(cls)

	@classmethod
	def storage_dtype(cls) -> np.dtype:
		return np.dtype((cls._base_dtype, (cls._count,)))


class Vec2(_baseVector):
	_count = 2
	x = _array_accessor(0)
	y = _array_accessor(1)


class Vec3(_baseVector):
	_count = 3
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)


class Vec4(_baseVector):
	_count = 4
	x = _array_accessor(0)
	y = _array_accessor(1)
	z = _array_accessor(2)
	w = _array_accessor(3)


class Camera:
	def __init__(self, position: tuple, target: tuple, up: tuple, fovy_deg:float, near:float=0.1, far:float=1000.0, perspective:bool=True) -> None:
		self.position = Vec3(position)
		self.target = Vec3(target)
		self.up = Vec3(up)
		self.fovy_deg = fovy_deg
		self.near = near
		self.far = far
		self.perspective = perspective

	@cache_last(lambda self: (tuple(self.position), tuple(self.target), tuple(self.up)) )
	def view(self) -> np.ndarray:
		return Camera.calc_view(self.position, self.target, self.up)
	
	@staticmethod
	def calc_view(position: ArrayLike, target: ArrayLike, up: ArrayLike) -> np.ndarray:
		position, target, up = (np.asarray(value, dtype=np.float64) for value in (position, target, up))
		forward = target - position
		if not np.all(np.isfinite([position, target, up])) or np.linalg.norm(forward) == 0:
			raise ValueError("Expected finite camera values with distinct position and target")
		forward /= np.linalg.norm(forward)
		side = np.cross(forward, up)
		if np.linalg.norm(side) == 0:
			raise ValueError("Camera up must not be parallel to the view direction")
		side /= np.linalg.norm(side)
		up = np.cross(side, forward)
		matrix = np.eye(4, dtype=np.float32)
		matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = side, up, -forward
		matrix[3, :3] = (-np.dot(side, position), -np.dot(up, position), np.dot(forward, position))
		return matrix
	
	# NOTE: we could split this class into orth and perspective cameras	
	@cache_last(lambda self, aspect: (self.perspective, self.fovy_deg, aspect, self.near, self.far))
	def projection(self, aspect:float) -> np.ndarray:
		if self.perspective: return Camera.perspective_projection(self.fovy_deg, aspect, self.near, self.far )
		top = self.fovy_deg * 0.5
		right = top*aspect
		left = -right
		bottom = -top
		return Camera.orthogonal_projection(left, right, bottom, top, self.near, self.far)

	@cache_last(lambda self: (tuple(self.position), tuple(self.target)) )
	def direction(self) -> np.ndarray:
		l = self.position - self.target
		return l / np.linalg.norm(l)
	
	# Projection matrices use WGPU depth conventions.
	# opengl has ndc -1->1, wgpu is 0->1

	@staticmethod
	def perspective_projection(fovy_deg:float, aspect:float, near:float, far:float) -> np.ndarray:
		f = 1.0/np.tan(fovy_deg*3.14159/180.0 *0.5)
		far_factor = far/(near-far)
		mat = np.array([
			[f/aspect, 0.0,            0.0,  0.0],
			[0.0,        f,            0.0,  0.0],
			[0.0,      0.0,     far_factor, -1.0],
			[0.0,      0.0, far_factor*near, 0.0],
		], dtype=np.float32)
		#return np.ascontiguousarray(mat, dtype=np.float32)
		return mat

	@staticmethod
	def orthogonal_projection(left:float, right:float, bottom:float, top:float, near:float, far:float) -> np.ndarray:

		rml = right - left
		rpl = right + left
		tmb = top - bottom
		tpb = top + bottom
		fn = far - near

		mat = np.array([
			[2.0/rml,       0.0,      0.0, 0.0],
			[0.0,       2.0/tmb,      0.0, 0.0],
			[0.0,           0.0,  -1.0/fn, 0.0],
			[-rpl/rml, -tpb/tmb, -near/fn, 1.0],
		], dtype=np.float32)

		return mat


def extract_frustum_planes(view_projection: ArrayLike) -> np.ndarray:
	vp = np.asarray(view_projection)
	planes = np.zeros((6, 4), dtype=np.float32)
	planes[0], planes[1] = vp[:, 3] + vp[:, 0], vp[:, 3] - vp[:, 0]
	planes[2], planes[3] = vp[:, 3] + vp[:, 1], vp[:, 3] - vp[:, 1]
	planes[4], planes[5] = vp[:, 2], vp[:, 3] - vp[:, 2]
	for i in range(6):
		norm = np.linalg.norm(planes[i, :3])
		if norm > 0: planes[i] /= norm
	return planes


"""Math shared by the physics examples and benchmarks."""
import numpy as np

from graaal.Math import Quaternion


def advance_fixed_steps(accumulator: float, elapsed: float, tick_hz: float) -> tuple[int, float]:
	"""Return the number of fixed ticks and the unspent time in seconds."""
	accumulator += elapsed
	steps = int(accumulator * tick_hz)
	return steps, max(0.0, accumulator - steps / tick_hz)


def joint_error(parents: np.ndarray, orientations: np.ndarray, neutral: np.ndarray) -> np.ndarray:
	"""Shortest relative joint error in radians for one pose or a batch, excluding the root."""
	relative = Quaternion.multiply_batch(orientations[..., parents[1:], :] * [-1, -1, -1, 1], orientations[..., 1:, :])
	error = Quaternion.multiply_batch(relative, neutral[1:] * [-1, -1, -1, 1])
	return 2 * np.arctan2(np.linalg.norm(error[..., :3], axis=-1), np.abs(error[..., 3]))

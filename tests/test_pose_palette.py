import unittest
from unittest.mock import patch

import numpy as np

from graaal.EcsAnimation import PoseBatch
from graaal.PosePalette import PosePalette
from graaal.SkinMatrixBuffer import SkinMatrixBuffer


class FakeBuffer:
	def __init__(self, data, usage):
		self.content = data.copy()

	def write_range(self, data, offset):
		flat = self.content.reshape(-1)
		values = np.asarray(data).reshape(-1)
		flat[offset // 4:offset // 4 + len(values)] = values

	def close(self): pass


def batch(owners, bones, value):
	matrices = np.full((len(owners), bones, 3, 4), value, dtype=np.float32)
	return PoseBatch(0, 0, np.asarray(owners, dtype=np.uint64), matrices)


class PosePaletteTests(unittest.TestCase):
	def setUp(self):
		self.patch = patch("graaal.SkinMatrixBuffer.GpuBuffer", FakeBuffer)
		self.patch.start()
		self.addCleanup(self.patch.stop)

	def test_recycles_removed_ranges_and_uploads_all_poses(self):
		palette = SkinMatrixBuffer(4)
		poses = PosePalette(palette)
		poses.update([batch([1, 2], 2, 3)])
		self.assertEqual(palette.range(1), (0, 2))
		self.assertEqual(palette.range(2), (2, 2))
		new_handle = (1 << 32) + 1
		poses.update([batch([new_handle], 2, 5), batch([2], 2, 7)])
		self.assertEqual(palette.range(new_handle), (0, 2))
		with self.assertRaises(KeyError): palette.range(1)
		np.testing.assert_array_equal(palette.buffer.content[:, 0, 0], [5, 5, 7, 7])
		poses.update([])
		with self.assertRaises(KeyError): palette.range(2)
		self.assertEqual(palette.acquire(99, 4), 0)

	def test_validation_does_not_change_allocations_or_uploads(self):
		palette = SkinMatrixBuffer(4)
		poses = PosePalette(palette)
		poses.update([batch([1], 2, 3)])
		before = palette.buffer.content.copy()
		for batches in ([batch([1, 1], 2, 0)], [batch([1], 1, 0)], [batch([2], 2, np.nan)]):
			with self.assertRaises(ValueError): poses.update(batches)
			self.assertEqual(palette.range(1), (0, 2))
			np.testing.assert_array_equal(palette.buffer.content, before)
		palette.acquire(9, 1)
		with self.assertRaisesRegex(ValueError, "unmanaged"): poses.update([batch([9], 1, 0)])
		self.assertEqual(palette.range(1), (0, 2))

	def test_failed_capacity_rolls_back_new_allocations_and_can_retry(self):
		palette = SkinMatrixBuffer(3)
		poses = PosePalette(palette)
		poses.update([batch([1], 1, 1)])
		with self.assertRaisesRegex(ValueError, "insufficient"):
			poses.update([batch([2, 3], 2, 2)])
		for owner in (1, 2, 3):
			with self.assertRaises(KeyError): palette.range(owner)
		poses.update([batch([2], 2, 4)])
		self.assertEqual(palette.range(2), (0, 2))
		np.testing.assert_array_equal(palette.buffer.content[:2, 0, 0], [4, 4])

	def test_external_release_is_detected(self):
		palette = SkinMatrixBuffer(2)
		poses = PosePalette(palette)
		poses.update([batch([1], 1, 3)])
		palette.release(1)
		with self.assertRaisesRegex(ValueError, "released externally"): poses.update([])


if __name__ == "__main__":
	unittest.main()

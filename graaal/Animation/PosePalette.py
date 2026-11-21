"""Manage skin palette ranges from complete batches of animation poses."""

import numpy as np


class PosePalette:
	"""Own palette ranges for the owners present in the latest pose snapshot.

	The caller synchronizes draw membership and GPU work around updates. No other
	code may acquire or release the ranges owned by this adapter.
	"""
	def __init__(self, palette):
		self.palette = palette
		self._owners = {}

	def update(self, batches):
		"""Reconcile a complete pose snapshot and upload all supplied matrices."""
		prepared, desired = [], {}
		for batch in batches:
			owners = np.asarray(batch.entities)
			matrices = np.asarray(batch.matrices, dtype=np.float32)
			if owners.ndim != 1 or owners.dtype.kind not in "iu" or np.any(owners < 0):
				raise ValueError("owners must be a one-dimensional array of nonnegative integers")
			if matrices.ndim != 4 or matrices.shape[0] != len(owners) or matrices.shape[2:] != (3, 4) or matrices.shape[1] == 0 or not np.isfinite(matrices).all():
				raise ValueError("matrices must be finite with shape (owners, bones, 3, 4) and at least one bone")
			for owner in owners:
				owner = int(owner)
				if owner in desired: raise ValueError("owner IDs must be distinct across pose batches")
				desired[owner] = matrices.shape[1]
			prepared.append((owners, matrices))
		for owner, allocation in self._owners.items():
			try: current = self.palette.range(owner)
			except KeyError as exc: raise ValueError("managed palette range was released externally") from exc
			if current != allocation: raise ValueError("managed palette range was changed externally")
		for owner, bones in desired.items():
			if owner in self._owners:
				if self._owners[owner][1] != bones: raise ValueError("bone count changed for an existing owner")
			else:
				try: self.palette.range(owner)
				except KeyError: pass
				else: raise ValueError("owner already has an unmanaged palette range")
		for owner in self._owners.keys() - desired.keys():
			self.palette.release(owner)
			del self._owners[owner]
		acquired = []
		try:
			for owner, bones in desired.items():
				if owner in self._owners: continue
				base = self.palette.acquire(owner, bones)
				self._owners[owner] = (base, bones)
				acquired.append(owner)
		except Exception:
			for owner in acquired:
				self.palette.release(owner)
				del self._owners[owner]
			raise
		for owners, matrices in prepared:
			self.palette.write(owners, matrices)

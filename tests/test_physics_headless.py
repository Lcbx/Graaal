"""Native compatibility contracts for the optional, pinned physics backend."""

import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


def test_physics_is_optional():
	# Block the backend even on machines that have the physics extra installed.
	result = subprocess.run([sys.executable, "-c", '''
import sys
sys.modules["pyjoltphysics"] = None
import graaal
from graaal.ECS import ECS
from graaal.EcsPhysics import Collider, RigidBody, EcsPhysicsAdapter
from scenes.physics_headless_example import run_scene
assert ECS().create(1).size == 1
try:
	run_scene()
except ModuleNotFoundError as error:
	assert error.name == 'pyjoltphysics', str(error)
else:
	raise AssertionError("Missing backend should raise ModuleNotFoundError")
'''], capture_output=True, text=True, timeout=30)
	assert result.returncode == 0, result.stdout + result.stderr


def test_native_selftest_and_public_names(pj):
	assert pj.selftest() is True
	assert callable(pj.Shape.box)
	assert callable(pj.RagdollSettings.humanoid)
	assert {"STATIC", "KINEMATIC", "DYNAMIC"} <= pj.MotionType.__members__.keys()
	assert {"SENSOR", "CONTINUOUS", "REPORT"} <= pj.BodyFlag.__members__.keys()
	assert {"KINEMATIC", "POWERED", "LIMP"} <= pj.RagdollMode.__members__.keys()
	assert {"CONTACT", "TRIGGER", "STATE", "CHARACTER", "JOINT"} <= pj.EventKind.__members__.keys()
	assert {"BEGIN", "PERSIST", "END", "SLEEP", "WAKE"} <= pj.EventPhase.__members__.keys()


def test_headless_floor_contacts(pj):
	from scenes.physics_headless_example import run_scene

	result = run_scene()
	assert result["tick"] == 120
	assert result["time"] == pytest.approx(1.0)
	np.testing.assert_allclose(result["position"][:, 1], 0.5, atol=0.03)
	assert result["position"].flags.owndata
	events = result["events"]
	assert events.dtype == pj.EVENT_DTYPE
	contacts = events[(events["kind"] == pj.EventKind.CONTACT) & (events["phase"] == pj.EventPhase.BEGIN)]
	floor = result["floor_uids"][0]
	box_uids = np.where(contacts["a"] == floor, contacts["b"], contacts["a"])
	assert np.all((contacts["a"] == floor) | (contacts["b"] == floor))
	assert set(box_uids) == set(result["uids"])
	assert result["stats"]["events_dropped"] == 0


def test_lockstep_snapshots_and_uid_removal(pj):
	with pj.World(pj.WorldSettings(tick_hz=120, threads=1)) as world:
		uids, seq = world.add_bodies(pj.BodyBatch.boxes([[0, 5, 0], [2, 5, 0], [4, 5, 0]]))
		assert world.tick == 0
		world.run_ticks(1)
		first = world.latest()
		assert world.lockstep
		assert first.tick == 1
		assert first.applied_seq >= seq
		assert first.uid.dtype == np.uint32
		assert first.position.dtype == np.float64
		assert first.orientation.dtype == np.float32
		assert not first.position.flags.writeable
		assert not first.orientation.flags.writeable
		with pytest.raises(ValueError):
			first.position[0, 0] = 100
		np.testing.assert_allclose(first.orientation, [[0, 0, 0, 1]] * 3)
		saved = first.position.copy()
		seq = world.remove_bodies(uids[:1])
		world.run_ticks(1)
		second = world.latest()
		assert second.tick == 2
		assert second.applied_seq >= seq
		assert second.rows_of(uids[:1])[0] == -1
		rows = second.rows_of(uids[1:])
		assert np.all(rows >= 0)
		np.testing.assert_array_equal(second.uid[rows], uids[1:])
		assert np.all(second.position[rows, 1] < saved[1:, 1])
		np.testing.assert_array_equal(first.position, saved)
		assert world.drain_events().dtype == pj.EVENT_DTYPE
		assert len(world.drain_events()) == 0
	# stop joins the thread; repeated shutdown is supported.
	world.stop()
	assert world.tick == 2


def test_backend_validation_and_exception_cleanup(pj):
	with pytest.raises(ValueError, match="tick_hz"):
		pj.WorldSettings(tick_hz=0)
	world = pj.World(pj.WorldSettings(threads=1))
	with pytest.raises(ValueError):
		with world:
			world.add_bodies(pj.BodyBatch.boxes([[0, 2, 0]]))
			world.run_ticks(1)
			world.add_bodies(pj.BodyBatch.boxes([[0, np.nan, 0]]))
	world.stop()
	assert world.error is None


def test_worker_exception_reaches_caller(pj):
	failure = ValueError("physics callback failed")
	def fail(tick, time):
		raise failure

	with pj.World(pj.WorldSettings(threads=1)) as world:
		world.set_before_step(fail)
		with pytest.raises(pj.WorldError, match="world callback raised") as caught:
			world.run_ticks(1)
		assert caught.value.__cause__ is failure
		assert world.error is failure

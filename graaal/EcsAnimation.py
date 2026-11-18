"""Optional ECS clip playback adapter for the standalone skeletal evaluator."""
from dataclasses import dataclass

import numpy as np

from .ECS import component
from .RootMotion import _blend_deltas
from .SkeletalAnimation import model_matrices, sample_clip, sample_crossfade, skin_matrices


@component
class SkeletonRef:
	id: np.uint32


@component
class ClipPlayer:
	clip_id: np.uint32
	time: np.float32 = 0.0
	speed: np.float32 = 1.0
	loop: bool = True
	playing: bool = True
	transitioning: bool = False
	target_clip_id: np.uint32 = 0
	target_time: np.float32 = 0.0
	target_speed: np.float32 = 1.0
	target_loop: bool = True
	transition_time: np.float32 = 0.0
	transition_duration: np.float32 = 1.0
	extract_root_motion: bool = False


@dataclass(frozen=True)
class PoseBatch:
	"""Skin matrices in entity order for a skeleton and source clip at update start."""
	skeleton_id: int
	clip_id: int
	entities: np.ndarray
	matrices: np.ndarray


@dataclass(frozen=True)
class AnimationEvent:
	"""An event crossed by one player's source or target clip."""
	entity: int
	clip_id: int
	name: str
	role: str
	time: float


@dataclass(frozen=True)
class RootMotionBatch:
	"""Character-frame movement requests in full entity-handle order."""
	skeleton_id: int
	clip_id: int
	entities: np.ndarray
	matrices: np.ndarray


class EcsAnimationAdapter:
	"""Advance ECS players and produce batches keyed by full entity handles.

	`skeletons` and `clips` map component IDs to shared resources. Fetch a fresh
	query each update; pose batches are snapshots and never retain ECS selections.
	The caller owns render mapping and consumes the returned batches before the
	next update or structural change.
	"""
	def __init__(self, world, skeletons, clips, event_tracks=None, root_motions=None):
		self.world, self.skeletons, self.clips = world, skeletons, clips
		self.event_tracks = {} if event_tracks is None else event_tracks
		self.root_motions = {} if root_motions is None else root_motions
		self.query = world.query(SkeletonRef, ClipPlayer)
		self.batches = []
		self.events = []
		self.root_motion_batches = []
		self._pose_buffers = {}

	def clear_pose_buffers(self):
		"""Release cached pose storage; subsequent updates allocate it again."""
		self._pose_buffers.clear()

	def _pose_workspace(self, key, count, bones):
		buffers = self._pose_buffers.get(key)
		if buffers is None or buffers[0].shape[0] < count or buffers[0].shape[1] != bones:
			capacity = max(count, 1 if buffers is None else len(buffers[0]) * 2)
			buffers = tuple(np.empty((capacity, bones) + shape, dtype=np.float32) for shape in ((3,), (4,), (3,), (4, 4), (3, 4)))
			self._pose_buffers[key] = buffers
		return tuple(value[:count] for value in buffers)

	def _events(self, clip_id, clip, entities, times, speeds, loops, steps, role):
		track = self.event_tracks.get(clip_id)
		if track is None: return []
		if track.duration != clip.duration: raise ValueError("Event track duration must match its clip")
		events = []
		for entity, time, speed, loop, step in zip(entities, times, speeds, loops, steps):
			start = float(time) % clip.duration if loop else float(np.clip(time, 0, clip.duration))
			for occurrence in track.between(start, start + float(speed) * float(step), loop=bool(loop)):
				events.append(AnimationEvent(int(entity), clip_id, occurrence.name, role, occurrence.time))
		return events

	def _motion_requests(self, motion, start, end, loops):
		result = np.empty((len(start), 4, 4), dtype=np.float64)
		for repeats in (False, True):
			selected = loops == repeats
			if np.any(selected): result[selected] = motion.character_delta(start[selected], end[selected], loop=repeats)
		return result

	def update(self, dt):
		"""Advance by nonnegative seconds and return one PoseBatch per resource pair."""
		if not np.isscalar(dt) or not np.isfinite(dt) or dt < 0:
			raise ValueError("dt must be a finite nonnegative scalar")
		result = self.query.fetch()
		players = result[ClipPlayer].selection
		skeleton_ids = result[SkeletonRef].selection.id
		clip_ids, times, speeds = players.clip_id, players.time, players.speed
		loop, playing = players.loop, players.playing
		transitioning = players.transitioning
		target_ids, target_times, target_speeds = players.target_clip_id, players.target_time, players.target_speed
		target_loop, elapsed, duration = players.target_loop, players.transition_time, players.transition_duration
		if not np.isfinite(times).all() or not np.isfinite(speeds).all():
			raise ValueError("player times and speeds must be finite")
		active = transitioning.astype(bool)
		root_motion = players.extract_root_motion.astype(bool)
		if not all(np.isfinite(value[active]).all() for value in (target_times, target_speeds, elapsed, duration)) or np.any(elapsed[active] < 0) or np.any(duration[active] <= 0):
			raise ValueError("active transition times and speeds must be finite; elapsed must be nonnegative and duration positive")
		advanced = times.astype(np.float64) + np.where(playing, speeds.astype(np.float64) * dt, 0)
		advanced_target = target_times.astype(np.float64) + np.where(playing, target_speeds.astype(np.float64) * dt, 0)
		advanced_elapsed = elapsed.astype(np.float64) + np.where(playing, dt, 0)
		if not np.isfinite(advanced).all() or not np.isfinite(advanced_target[active]).all() or not np.isfinite(advanced_elapsed[active]).all():
			raise ValueError("advanced player times must be finite")
		next_times = np.empty(len(result), dtype=np.float32)
		next_ids, next_speeds, next_loop = clip_ids.copy(), speeds.copy(), loop.copy()
		next_target_times, next_elapsed = target_times.copy(), elapsed.copy()
		next_transitioning = transitioning.copy()
		batches = []
		events = []
		root_batches = []
		pairs = np.unique(np.stack((skeleton_ids, clip_ids), axis=1), axis=0)
		for skeleton_id, clip_id in pairs:
			skeleton_id, clip_id = int(skeleton_id), int(clip_id)
			if skeleton_id not in self.skeletons or clip_id not in self.clips:
				raise ValueError(f"Unknown animation resources: skeleton {skeleton_id}, clip {clip_id}")
			skeleton, clip = self.skeletons[skeleton_id], self.clips[clip_id]
			if clip.translation.shape[1] != len(skeleton.parents):
				raise ValueError("Clip and skeleton bone counts differ")
			rows = np.flatnonzero((skeleton_ids == skeleton_id) & (clip_ids == clip_id))
			source_steps = np.where(playing[rows], dt, 0.0)
			source_steps = np.where(active[rows], np.minimum(source_steps, np.maximum(duration[rows] - elapsed[rows], 0)), source_steps)
			events.extend(self._events(clip_id, clip, result.entities[rows], times[rows], speeds[rows], loop[rows], source_steps, "source"))
			if clip.duration:
				local_times = np.where(loop[rows], np.mod(advanced[rows], clip.duration), np.clip(advanced[rows], 0, clip.duration))
			else: local_times = np.zeros(len(rows), dtype=np.float64)
			next_times[rows] = local_times
			translation, rotation, scale, model, skin = self._pose_workspace((skeleton_id, clip_id), len(rows), len(skeleton.parents))
			pose = sample_clip(clip, local_times, loop=False, out=(translation, rotation, scale))
			motion_rows = np.flatnonzero(root_motion[rows])
			if len(motion_rows):
				motion = self.root_motions.get((skeleton_id, clip_id))
				if motion is None or motion.skeleton is not skeleton or motion.clip is not clip:
					raise ValueError(f"Missing or incompatible root motion for skeleton {skeleton_id}, clip {clip_id}")
				selected = rows[motion_rows[~active[rows[motion_rows]]]]
				if len(selected):
					requests = self._motion_requests(motion, times[selected], advanced[selected], loop[selected])
					root_batches.append(RootMotionBatch(skeleton_id, clip_id, result.entities[selected].copy(), requests))
			for target_id in np.unique(target_ids[rows[active[rows]]]):
				target_id = int(target_id)
				if target_id not in self.clips: raise ValueError(f"Unknown target clip {target_id}")
				target = self.clips[target_id]
				if target.translation.shape[1] != len(skeleton.parents): raise ValueError("Clip and skeleton bone counts differ")
				local_rows = np.flatnonzero(active[rows] & (target_ids[rows] == target_id))
				selected = rows[local_rows]
				target_steps = np.where(playing[selected], dt, 0.0)
				events.extend(self._events(target_id, target, result.entities[selected], target_times[selected], target_speeds[selected], target_loop[selected], target_steps, "target"))
				if target.duration:
					target_local = np.where(target_loop[selected], np.mod(advanced_target[selected], target.duration), np.clip(advanced_target[selected], 0, target.duration))
				else: target_local = np.zeros(len(selected), dtype=np.float64)
				progress = np.minimum(advanced_elapsed[selected], duration[selected])
				blended = sample_crossfade(clip, target, local_times[local_rows], target_local, progress / duration[selected], source_loop=False, target_loop=False)
				for output, values in zip(pose, blended): output[local_rows] = values
				motion_selected = selected[root_motion[selected]]
				if len(motion_selected):
					target_motion = self.root_motions.get((skeleton_id, target_id))
					if target_motion is None or target_motion.skeleton is not skeleton or target_motion.clip is not target or target_motion.root != motion.root:
						raise ValueError(f"Missing or incompatible root motion for skeleton {skeleton_id}, target clip {target_id}")
					step = np.where(playing[motion_selected], dt, 0.0)
					fade = np.minimum(step, np.maximum(duration[motion_selected].astype(np.float64) - elapsed[motion_selected], 0))
					source_end = times[motion_selected].astype(np.float64) + speeds[motion_selected].astype(np.float64) * fade
					target_mid = target_times[motion_selected].astype(np.float64) + target_speeds[motion_selected].astype(np.float64) * fade
					source_delta = self._motion_requests(motion, times[motion_selected], source_end, loop[motion_selected])
					target_delta = self._motion_requests(target_motion, target_times[motion_selected], target_mid, target_loop[motion_selected])
					weight = (np.minimum(elapsed[motion_selected], duration[motion_selected]) + fade * 0.5) / duration[motion_selected]
					remaining = self._motion_requests(target_motion, target_mid, advanced_target[motion_selected], target_loop[motion_selected])
					requests = _blend_deltas(source_delta, target_delta, weight) @ remaining
					root_batches.append(RootMotionBatch(skeleton_id, clip_id, result.entities[motion_selected].copy(), requests))
				next_target_times[selected], next_elapsed[selected] = target_local, progress
				finished = selected[progress >= duration[selected]]
				next_ids[finished] = target_ids[finished]
				next_times[finished] = next_target_times[finished]
				next_speeds[finished], next_loop[finished] = target_speeds[finished], target_loop[finished]
				next_transitioning[finished] = False
			if len(motion_rows):
				in_place = motion.in_place(*(value[motion_rows] for value in pose))
				for output, values in zip(pose, in_place): output[motion_rows] = values
			matrices = skin_matrices(skeleton, model_matrices(skeleton, *pose, out=model), out=skin)
			batches.append(PoseBatch(skeleton_id, clip_id, result.entities[rows].copy(), matrices))
		if len(result):
			players.assign(clip_id=next_ids, time=next_times, speed=next_speeds, loop=next_loop, transitioning=next_transitioning, target_time=next_target_times, transition_time=next_elapsed)
		self.events = events
		self.root_motion_batches = root_batches
		return batches

	def evaluate(self, world, commands, dt):
		"""Scheduler callback; the most recent result is available as `batches`."""
		if world is not self.world: raise ValueError("Animation adapter belongs to a different world")
		self.batches = self.update(dt)

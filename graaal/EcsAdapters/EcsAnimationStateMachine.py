"""Optional ECS adapter connecting conditional states to clip playback."""
from __future__ import annotations

from operator import index
from typing import cast, Mapping, TYPE_CHECKING

from numpy.typing import ArrayLike
import numpy as np

from ..Animation.AnimationSync import SyncMapping
from ..ECS import component
from .EcsAnimation import ClipPlayer


if TYPE_CHECKING:
	from ..Animation.AnimationStateMachine import AnimationStateMachine
	from ..ECS import ECS


@component
class AnimationState:
	state: np.uint32
	target_state: np.uint32 = cast(np.uint32, 0)
	pending: bool = False


class EcsAnimationStateAdapter:
	"""Start player crossfades and promote states after playback completes."""
	def __init__(self, world: ECS, machine: AnimationStateMachine, state_clips: Mapping[int, int], sync_mappings: Mapping[tuple[int, int], SyncMapping] | None=None) -> None:
		self.world, self.machine = world, machine
		self.sync_mappings = {} if sync_mappings is None else sync_mappings
		self.state_clips = {}
		for state, clip in state_clips.items():
			try:
				if isinstance(state, (bool, np.bool_)) or isinstance(clip, (bool, np.bool_)): raise ValueError
				state, clip = index(state), index(clip)
			except (TypeError, ValueError) as exc: raise ValueError("state and clip IDs must be uint32 integers") from exc
			if not (0 <= state <= 2**32 - 1 and 0 <= clip <= 2**32 - 1): raise ValueError("state and clip IDs must be uint32 integers")
			self.state_clips[state] = clip
		self.query = world.query(AnimationState, ClipPlayer)

	def update(self, parameters: ArrayLike) -> np.ndarray:
		"""Apply decisions for parameter rows in the current query entity order.

		Call before the playback adapter each tick. A completed fade is promoted
		at the next call, when another transition may also begin.
		"""
		result = self.query.fetch()
		states = result[AnimationState].selection
		players = result[ClipPlayer].selection
		current, target, pending = states.state, states.target_state, states.pending
		active = players.transitioning
		if np.any(active & ~pending): raise ValueError("player has a transition not owned by the state adapter")
		for state in np.unique(np.concatenate((current, target[pending]))):
			if int(state) not in self.state_clips: raise ValueError(f"No clip for state {int(state)}")
		source_clips = np.array([self.state_clips[int(state)] for state in current], dtype=np.uint32)
		target_clips = np.array([self.state_clips[int(state)] for state in target[pending]], dtype=np.uint32)
		if np.any(players.clip_id[~pending | active] != source_clips[~pending | active]): raise ValueError("player clip does not match current state")
		if np.any(players.target_clip_id[pending & active] != np.array([self.state_clips[int(state)] for state in target[pending & active]], dtype=np.uint32)):
			raise ValueError("player target clip does not match target state")
		if np.any(players.clip_id[pending & ~active] != target_clips[~active[pending]]): raise ValueError("completed player clip does not match target state")
		completed = pending & ~active
		effective = np.where(completed, target, current)
		selected, next_target, durations = self.machine.select(effective, parameters, active)
		if np.any(next_target[selected] > np.iinfo(np.uint32).max): raise ValueError("target state exceeds uint32")
		missing = [int(state) for state in np.unique(next_target[selected]) if int(state) not in self.state_clips]
		if missing: raise ValueError(f"No clip for state {missing[0]}")
		with np.errstate(over="ignore", under="ignore"):
			fade = durations[selected].astype(np.float32)
		if not np.isfinite(fade).all() or np.any(fade <= 0): raise ValueError("transition durations must fit positive float32")
		new_clips = np.array([self.state_clips[int(state)] for state in next_target[selected]], dtype=np.uint32)
		new_times = np.zeros(len(new_clips), dtype=np.float64)
		new_speeds = players.speed[selected].astype(np.float64)
		selected_sources = players.clip_id[selected]
		for source_id, target_id in np.unique(np.column_stack((selected_sources, new_clips)), axis=0):
			mapping = self.sync_mappings.get((int(source_id), int(target_id)))
			if mapping is None: continue
			if not isinstance(mapping, SyncMapping): raise ValueError("invalid sync mapping")
			rows = (selected_sources == source_id) & (new_clips == target_id)
			if not np.all(players.loop[selected][rows]): raise ValueError("synchronized transitions require looping clips")
			times = players.time[selected][rows]
			new_times[rows] = mapping.map(times)[0]
			for reverse in (False, True):
				choose = rows & ((players.speed[selected] < 0) == reverse)
				if np.any(choose): new_speeds[choose] *= mapping.map(players.time[selected][choose], reverse=reverse)[1]
		with np.errstate(over="ignore", invalid="ignore"):
			new_times, new_speeds = new_times.astype(np.float32), new_speeds.astype(np.float32)
		if not np.isfinite(new_times).all() or not np.isfinite(new_speeds).all(): raise ValueError("transition times and speeds must fit float32")
		if np.any(completed): states.take(completed).assign(state=target[completed], pending=False)
		if np.any(selected):
			players.take(selected).assign(transitioning=True, target_clip_id=new_clips, target_time=new_times, target_speed=new_speeds, target_loop=players.loop[selected], transition_time=0, transition_duration=fade)
			states.take(selected).assign(target_state=next_target[selected].astype(np.uint32), pending=True)
		return result.entities.copy()

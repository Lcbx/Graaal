"""Track airborne, relocated landings and post-landing COM braking."""


def rescale_base_actions(base_action, non_ankle, *, base_scale, step_scale):
	"""Express standing targets in the step environment's action scale."""
	base = base_action.clamp(-1, 1).clone()
	base[:, non_ankle] *= base_scale / step_scale
	return base.clamp_(-1, 1)


def motion_gated_actions(features, base_action, step_action, gate_ticks, non_ankle, *, trigger_speed, hold_ticks, base_scale, step_scale):
	"""Blend bounded actions using measured backward COM speed and a hold timer."""
	import torch

	base = rescale_base_actions(base_action, non_ankle, base_scale=base_scale, step_scale=step_scale)
	trigger = features[:, -2] < -trigger_speed
	ticks = torch.where(trigger, hold_ticks, (gate_ticks - 1).clamp_min(0))
	return torch.where((ticks > 0)[:, None], step_action.clamp(-1, 1), base), ticks


class StepTracker:
	"""Measure the first qualifying post-push landings of each foot.

	Contact is the simulator's per-foot net upward force threshold, so these are
	diagnostics, not a substitute for collision-pair or slip verification.
	"""

	def __init__(self, num_envs, device):
		import torch

		self.torch = torch
		self.vectors = torch.tensor(((1, 0), (-1, 0), (0, 1), (0, -1)), device=device, dtype=torch.float32)
		self.push_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
		self.origin = torch.zeros((num_envs, 2, 3), device=device)
		self.direction = torch.zeros((num_envs, 2), device=device)
		self.air_ticks = torch.zeros((num_envs, 2), dtype=torch.int32, device=device)
		self.contact_ticks = torch.zeros_like(self.air_ticks)
		self.max_lift = torch.zeros((num_envs, 2), device=device)
		self.pending = torch.full((num_envs, 2), -1, dtype=torch.int32, device=device)
		self.speed_at_landing = torch.zeros((num_envs, 2), device=device)
		self.pending_held = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
		self.landed = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
		self.braked = torch.zeros_like(self.landed)

	def reset(self, mask):
		"""Clear finished environments without changing other trials."""
		self.push_seen.masked_fill_(mask, False)
		self.origin.masked_fill_(mask[:, None, None], 0)
		self.direction.masked_fill_(mask[:, None], 0)
		for value in (self.air_ticks, self.contact_ticks, self.max_lift, self.speed_at_landing, self.pending_held, self.landed, self.braked):
			value.masked_fill_(mask[:, None], 0)
		self.pending.masked_fill_(mask[:, None], -1)

	def update(self, positions, contact, com_speed, active, push, direction):
		"""Update one 60 Hz sample, using pre-reset positions and contact."""
		torch = self.torch
		new_push = push & active & ~self.push_seen
		self.origin = torch.where(new_push[:, None, None], positions, self.origin)
		self.direction = torch.where(new_push[:, None], self.vectors[direction], self.direction)
		self.push_seen |= new_push
		valid = (active & self.push_seen)[:, None]
		lift = positions[:, :, 1] - self.origin[:, :, 1]
		self.max_lift = torch.where(valid & ~contact, torch.maximum(self.max_lift, lift), self.max_lift)
		displacement = positions[:, :, [0, 2]] - self.origin[:, :, [0, 2]]
		along = (displacement * self.direction[:, None]).sum(dim=-1)
		touchdown = valid & contact & (self.air_ticks >= 3) & (self.max_lift >= 0.03) & (along >= 0.05) & (self.pending < 0)
		self.pending = torch.where(touchdown, 0, self.pending)
		self.pending_held &= ~touchdown
		self.speed_at_landing = torch.where(touchdown, com_speed[:, None], self.speed_at_landing)
		self.contact_ticks = torch.where(valid & contact, self.contact_ticks + 1, 0)
		self.air_ticks = torch.where(valid & ~contact, self.air_ticks + 1, 0)
		self.max_lift = torch.where(contact, 0, self.max_lift)
		self.pending = torch.where(valid & ~contact & (self.pending >= 0) & ~self.pending_held, -1, self.pending)
		self.pending = torch.where(valid & (self.pending >= 0), self.pending + 1, self.pending)
		self.pending_held |= valid & (self.pending >= 0) & (self.contact_ticks >= 3)
		self.landed |= self.pending_held
		at_window = valid & (self.pending == 13)
		self.braked |= at_window & self.pending_held & (self.speed_at_landing - com_speed[:, None] >= 0.02)
		self.pending = torch.where(at_window, -1, self.pending)

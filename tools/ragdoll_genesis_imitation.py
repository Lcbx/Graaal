"""Initialize a Genesis actor from oracle-timed recovery-step trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.ragdoll_genesis_train import StandingEnv, warm_start_foot_features
from tools.ragdoll_project import default_project


def load_dataset(path):
	"""Load a step-reference dataset, retaining whole-episode split labels."""
	with np.load(path, allow_pickle=False) as data:
		observations = np.asarray(data["observations"], dtype=np.float32)
		actions = np.asarray(data["actions"], dtype=np.float32)
		swing = np.asarray(data["swing"], dtype=bool)
		metadata = json.loads(str(data["metadata"]))
		episodes = int(metadata["episodes"] if "episodes" in metadata else metadata["successful_episodes"])
		if "episode" in data:
			episode = np.asarray(data["episode"], dtype=np.int32)
		elif episodes > 0 and len(observations) % episodes == 0:
			episode = np.tile(np.arange(episodes, dtype=np.int32), len(observations) // episodes)
		else:
			raise ValueError("dataset does not identify complete episodes")
	if episodes < 2 or observations.ndim != 2 or observations.shape[1] != 147 or actions.shape != (len(observations), 30) or swing.shape != (len(observations),) or episode.shape != (len(observations),) or not np.isfinite(observations).all() or not np.isfinite(actions).all() or not swing.any() or not metadata.get("oracle_timing"):
		raise ValueError("expected finite, oracle-labeled step trajectories from at least two episodes")
	if not np.array_equal(np.unique(episode), np.arange(episodes)):
		raise ValueError("dataset episode identifiers are incomplete")
	return observations, actions, swing, episode, metadata


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("run", type=Path, help="original 133-observation standing run")
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("dataset", type=Path, nargs="+")
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--learning-rate", type=float, default=0.0001)
	parser.add_argument("--swing-weight", type=float, default=20)
	parser.add_argument("--seed", type=int, default=11)
	parser.add_argument("--save-every", type=int, default=0, help="save intermediate actor checkpoints every N epochs")
	parser.add_argument("--student-checkpoint", type=Path, help="continue fitting a compatible foot-observing actor")
	args = parser.parse_args()
	if args.epochs < 1 or args.batch_size < 1 or args.save_every < 0 or not np.isfinite((args.learning_rate, args.swing_weight)).all() or min(args.learning_rate, args.swing_weight) <= 0:
		parser.error("epochs, batch size, learning rate and swing weight must be positive")
	parts = [load_dataset(path) for path in args.dataset]
	observations = np.concatenate([part[0] for part in parts])
	actions = np.concatenate([part[1] for part in parts])
	swing = np.concatenate([part[2] for part in parts])
	offset = 0
	episode_parts = []
	for _, _, _, labels, _ in parts:
		episode_parts.append(labels + offset)
		offset += int(labels.max()) + 1
	episode = np.concatenate(episode_parts)
	metadata = [part[4] for part in parts]
	config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
	if not config.get("balance_features") or config.get("foot_features", False):
		parser.error("the source must be a 133-observation balance policy")
	project = default_project()
	args.output.mkdir(parents=True, exist_ok=False)
	(args.output / "rig.xml").write_text((args.run / "rig.xml").read_text(encoding="utf-8"), encoding="utf-8")
	config.update(dict(foot_features=True, limb_scale=0.5, imitation_dataset=[str(path) for path in args.dataset], imitation_source=str(args.checkpoint), imitation_student=str(args.student_checkpoint) if args.student_checkpoint else None, imitation_seed=args.seed, imitation_epochs=args.epochs, imitation_learning_rate=args.learning_rate, imitation_swing_weight=args.swing_weight))
	(args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
	import genesis as gs
	import torch
	from rsl_rl.runners import OnPolicyRunner
	from tensordict import TensorDict

	gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=args.seed)
	env = StandingEnv(args.output / "rig.xml", config["actions"], project.settings, num_envs=4, episode_seconds=4, push_impulse=0, balance_features=True, foot_features=True, motor_kp=config.get("motor_kp", 100), motor_kd=config.get("motor_kd", 5), limb_scale=0.5)
	runner = OnPolicyRunner(env, config["ppo"], log_dir=None, device=gs.device)
	warm_start_foot_features(runner, args.checkpoint)
	if args.student_checkpoint is not None:
		student_config = json.loads((args.student_checkpoint.parent / "config.json").read_text(encoding="utf-8"))
		if not student_config.get("foot_features", False) or student_config["actions"] != config["actions"] or student_config.get("limb_scale", 0.22) != 0.5 or (args.student_checkpoint.parent / "rig.xml").read_text(encoding="utf-8") != (args.run / "rig.xml").read_text(encoding="utf-8"):
			raise ValueError("student checkpoint must use the same rig and foot-observation contract")
		student = torch.load(args.student_checkpoint, map_location=gs.device, weights_only=False)
		runner.alg.actor.load_state_dict(student["actor_state_dict"])
		runner.alg.critic.load_state_dict(student["critic_state_dict"])
	actor = runner.alg.actor
	optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=args.learning_rate)
	x = torch.from_numpy(observations).to(gs.device)
	y = torch.from_numpy(actions).to(gs.device)
	weight = torch.from_numpy(np.where(swing, args.swing_weight, 1).astype(np.float32)).to(gs.device)
	train = torch.from_numpy(np.flatnonzero(episode % 5 != 0)).to(gs.device)
	validation = torch.from_numpy(np.flatnonzero(episode % 5 == 0)).to(gs.device)
	if not len(train) or not len(validation):
		raise ValueError("dataset needs both training and held-out episodes")
	for epoch in range(args.epochs):
		order = train[torch.randperm(len(train), device=gs.device)]
		for batch in order.split(args.batch_size):
			prediction = actor(TensorDict({"policy": x[batch]}, batch_size=[len(batch)])).clamp(-1, 1)
			error = (prediction - y[batch]).square().mean(dim=1)
			loss = (error * weight[batch]).sum() / weight[batch].sum()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		with torch.no_grad():
			prediction = actor(TensorDict({"policy": x[validation]}, batch_size=[len(validation)])).clamp(-1, 1)
			error = (prediction - y[validation]).square().mean(dim=1)
			validation_loss = float(((error * weight[validation]).sum() / weight[validation].sum()).item())
		print(f"epoch {epoch + 1}: held-out weighted action MSE {validation_loss:.6f}", flush=True)
		if args.save_every and (epoch + 1) % args.save_every == 0:
			runner.save(str(args.output / f"model_epoch_{epoch + 1}.pt"), infos=dict(imitation_dataset=[str(path) for path in args.dataset], validation_loss=validation_loss))
	runner.save(str(args.output / "model_0.pt"), infos=dict(imitation_dataset=[str(path) for path in args.dataset], validation_loss=validation_loss))
	(args.output / "imitation.json").write_text(json.dumps(dict(dataset=metadata, training_rows=len(train), validation_rows=len(validation), validation_loss=validation_loss), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()

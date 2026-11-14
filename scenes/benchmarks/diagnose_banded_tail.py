"""Measure banded-tail sampling bias and noise with the packaged noise tile."""

import argparse
import json
from pathlib import Path

import numpy as np

from graaal.Assets import load_blue_noise


def uniform_tile(tile):
	"""Map distinct tile values to uniform rank midpoints without moving texels."""
	tile = np.asarray(tile)
	if tile.ndim != 2 or not np.isfinite(tile).all(): raise ValueError("Expected a finite 2D noise tile")
	_, ranks, counts = np.unique(tile, return_inverse=True, return_counts=True)
	if np.any(counts != 1): raise ValueError("Expected distinct noise ranks")
	return (ranks.reshape(tile.shape)+0.5)/tile.size


def select(noise, probabilities):
	"""Return one fragment index per threshold; -1 represents an empty selection."""
	p = np.asarray(probabilities, dtype=np.float64)
	if p.ndim != 1 or not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
		raise ValueError("Expected nonempty probabilities in [0, 1]")
	edges = np.concatenate(([0.0], np.cumsum(p)))
	winner = np.searchsorted(edges, noise*max(1.0, edges[-1]), side="right")-1
	return np.where(winner < len(p), winner, -1)


def filter3(image):
	"""Apply a wrapped 3x3 box to a constant-color image."""
	weights = np.ones(3)
	weights /= weights.sum()
	return sum(weights[i+1]*weights[j+1]*np.roll(image, (i, j), axis=(0, 1)) for i in (-1, 0, 1) for j in (-1, 0, 1))


def estimate(noise, probabilities, colors):
	"""Return filtered complementary-pair tail radiance using deterministic coverage."""
	p = np.asarray(probabilities, dtype=np.float64)
	colors = np.asarray(colors, dtype=np.float64)
	numerator = np.zeros((*noise.shape, 3))
	for threshold in (noise, 1-noise):
		winner = select(threshold, p)
		numerator += 0.5*np.where((winner >= 0)[..., None], colors[np.maximum(winner, 0)], 0)
	coverage = 1-np.prod(1-p)
	denominator = min(1.0, p.sum())
	return filter3(numerator)*coverage/max(denominator, 1e-12)


def error_stats(image, reference):
	"""Measure maximum-channel bias, spatial noise, RMSE and low-frequency power."""
	error = image-reference
	spectrum = np.abs(np.fft.fft2(error, axes=(0, 1)))**2/np.prod(error.shape[:2])**2
	fy, fx = np.meshgrid(np.fft.fftfreq(error.shape[0]), np.fft.fftfreq(error.shape[1]), indexing="ij")
	band = (fx*fx+fy*fy > 0) & (fx*fx+fy*fy <= 0.125**2)
	return dict(bias=float(np.abs(error.mean(axis=(0, 1))).max()), noise=float(error.std(axis=(0, 1)).max()),
		rmse=float(np.sqrt(np.mean(error**2))), low_frequency_power=float(spectrum[band].sum(axis=0).max()))


def sweep(trials=64, seed=2026):
	"""Measure 2-6 equal-opacity layers with random colors against sorted stacks."""
	if type(trials) is not int or trials < 1: raise ValueError("trials must be positive")
	noise = uniform_tile(load_blue_noise())
	rng = np.random.default_rng(seed)
	rows = []
	for layers in range(2, 7):
		for alpha in (0.03, 0.05, 0.1, 0.15, 0.2):
			p = np.full(layers, alpha)
			weights = p*np.cumprod(np.concatenate(([1.0], 1-p[:-1])))
			measurements = [error_stats(estimate(noise, p, colors), weights@colors) for colors in rng.random((trials, layers, 3))]
			rows.append(dict(layers=layers, alpha=alpha, **{key: float(np.mean([m[key] for m in measurements])) for key in measurements[0]}))
	return dict(seed=seed, trials=trials, tile_shape=list(noise.shape), rows=rows,
		limitations=["Constant colors and a screen-aligned tile; excludes GPU scheduling, motion and MSAA.", "Thick-tail color is opacity-weighted and can differ from depth-sorted composition."])


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--trials", type=int, default=64)
	parser.add_argument("--seed", type=int, default=2026)
	parser.add_argument("--output", type=Path, default=Path("scenes/benchmarks/results/banded_tail_gate.json"))
	args = parser.parse_args()
	result = sweep(args.trials, args.seed)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
	print(f"Saved {args.output}")


if __name__ == "__main__":
	main()

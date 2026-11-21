"""Standalone graph authoring and pose inspection tool."""
import json
from pathlib import Path
import time

import numpy as np

from graaal.Animation.SkeletalAnimation import model_matrices, sample_clip_pairs
from .animation_preview import animation_bounds
from .animation_viewport import draw_animation_viewport
from .animation_asset import clip_resources, load_animations
from .animation_graph_playback import GraphPlayback, transition_pose
from .animation_project import compile_graph, read_project, write_project


def default_definition(animations):
	names = tuple(clip_resources(animations))
	return {
		"version": 1,
		"parameters": {"advance": 0},
		"states": {name: {"clip": name} for name in names},
		"initial_state": names[0],
		"transitions": [
			{"source": name, "target": names[(i + 1) % len(names)], "conditions": [{"parameter": "advance", "comparison": ">", "threshold": 0.5}], "duration": 0.25}
			for i, name in enumerate(names) if len(names) > 1
		],
		"sample_slots": ["base"],
		"sync_mappings": [],
		"tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}


class GraphWindow:
	"""Compile a draft and inspect poses with manual or automatic transitions."""
	STATE_SAMPLE = "<state / transition>"
	BLEND_PREFIX = "<blend: "
	GRID_PREFIX = "<grid: "

	def __init__(self, root, animations, *, closed=None):
		import tkinter as tk
		from tkinter import ttk

		self.tk, self.ttk = tk, ttk
		self.animations, self.asset = animations, animations[0].preview.asset
		self.closed = closed
		self.animation_id, self.position, self.playing = 0, 0.0, False
		self.last_tick = time.monotonic()
		self.graph, self.parameters, self.target_state = None, None, None
		self.parameter_entries, self.bindings = {}, {}
		self.transition_window = None
		self.blend_space_window = None
		self.blend_grid_window = None
		self.sync_marker_window = None
		self.project_path = None
		self.window = tk.Toplevel(root)
		self.window.title("Animation graph configuration")
		self.window.geometry("1500x850")
		self.window.minsize(850, 650)
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Edit the definition, then compile to preview it.")
		self.state = tk.StringVar()
		self.decision = tk.StringVar(value="No transition selected")
		self.fade = tk.DoubleVar(value=0)
		self.automatic = tk.BooleanVar(value=False)
		self.preview_enabled = tk.BooleanVar(value=True)
		self.clip_bounds = [animation_bounds(animation) for animation in animations]
		self.bounds = np.min([bound[0] for bound in self.clip_bounds], axis=0), np.max([bound[1] for bound in self.clip_bounds], axis=0)

		toolbar = ttk.Frame(self.window, padding=8)
		toolbar.pack(fill="x")
		ttk.Button(toolbar, text="Load project", command=self.load_project).pack(side="left", padx=(0, 5))
		ttk.Button(toolbar, text="Save project", command=self.save_project).pack(side="left", padx=(0, 5))
		ttk.Button(toolbar, text="Compile draft", command=self.compile_draft).pack(side="left")
		ttk.Button(toolbar, text="Edit transitions", command=self.open_transitions).pack(side="left", padx=5)
		ttk.Button(toolbar, text="Edit sync markers", command=self.open_sync_markers).pack(side="left", padx=5)
		ttk.Button(toolbar, text="Edit 1D blends", command=self.open_blend_spaces).pack(side="left", padx=5)
		ttk.Button(toolbar, text="Edit 2D grids", command=self.open_blend_grids).pack(side="left", padx=5)
		ttk.Checkbutton(toolbar, text="Preview graph in viewport", variable=self.preview_enabled, command=self.change_preview).pack(side="left", padx=10)
		clock_row = ttk.Frame(self.window, padding=(8, 0, 8, 8))
		clock_row.pack(fill="x")
		ttk.Label(clock_row, text="Clip").pack(side="left")
		self.clip_box = ttk.Combobox(clock_row, state="readonly", values=tuple(clip_resources(animations)), width=35)
		self.clip_box.pack(side="left", padx=5)
		self.clip_box.current(0)
		self.clip_box.bind("<<ComboboxSelected>>", self.change_clip)
		self.play_button = ttk.Button(clock_row, text="Play", command=self.toggle_play)
		self.play_button.pack(side="left", padx=5)
		self.clock = tk.DoubleVar(value=0)
		self.clock_scale = ttk.Scale(clock_row, from_=0, to=max(animation.clip.duration for animation in animations), variable=self.clock, command=self.change_time)
		self.clock_scale.pack(side="left", fill="x", expand=True, padx=5)
		self.clock_label = ttk.Label(clock_row, text="0.000s")
		self.clock_label.pack(side="left")
		panes = ttk.Panedwindow(self.window, orient="horizontal")
		panes.pack(fill="both", expand=True, padx=8)
		draft_frame, controls, view_frame = ttk.Frame(panes), ttk.Frame(panes), ttk.Frame(panes)
		panes.add(draft_frame, weight=3)
		panes.add(controls, weight=2)
		panes.add(view_frame, weight=2)
		view_controls = ttk.Frame(view_frame)
		view_controls.pack(fill="x")
		ttk.Label(view_controls, text="Animation viewport").pack(side="left", padx=5)
		self.show_mesh, self.show_bones = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
		ttk.Checkbutton(view_controls, text="Mesh", variable=self.show_mesh, command=self.draw_viewport).pack(side="left")
		ttk.Checkbutton(view_controls, text="Bones", variable=self.show_bones, command=self.draw_viewport).pack(side="left")
		self.viewport = tk.Canvas(view_frame, background="#171c24", highlightthickness=0)
		self.viewport.pack(fill="both", expand=True)
		self.viewport.bind("<Configure>", lambda _: self.draw_viewport())
		ttk.Label(draft_frame, text="Graph definition (JSON)").pack(anchor="w")
		self.editor = tk.Text(draft_frame, wrap="none", width=62, undo=True)
		self.editor.pack(side="left", fill="both", expand=True)
		scroll = ttk.Scrollbar(draft_frame, orient="vertical", command=self.editor.yview)
		scroll.pack(side="right", fill="y")
		self.editor.configure(yscrollcommand=scroll.set)
		ttk.Label(controls, text="States and transitions — click a state to inspect").pack(anchor="w")
		diagram_frame = ttk.Frame(controls)
		diagram_frame.pack(fill="x")
		diagram_frame.columnconfigure(0, weight=1)
		self.diagram = tk.Canvas(diagram_frame, height=230, background="#20242b", highlightthickness=0)
		self.diagram.grid(row=0, column=0, sticky="ew")
		diagram_scroll = ttk.Scrollbar(diagram_frame, orient="vertical", command=self.diagram.yview)
		diagram_scroll.grid(row=0, column=1, sticky="ns")
		self.diagram.configure(yscrollcommand=diagram_scroll.set)
		self.diagram.bind("<Configure>", lambda _: self.draw_diagram())
		self.state_box = ttk.Combobox(controls, textvariable=self.state, state="readonly")
		self.state_box.pack(fill="x", pady=5)
		self.state_box.bind("<<ComboboxSelected>>", self.change_state)
		ttk.Label(controls, text="Preview parameters").pack(anchor="w")
		self.parameter_frame = ttk.Frame(controls)
		self.parameter_frame.pack(fill="x")
		ttk.Button(controls, text="Apply parameters / evaluate conditions", command=self.apply_parameters).pack(fill="x", pady=5)
		ttk.Label(controls, textvariable=self.decision, wraplength=370).pack(anchor="w")
		ttk.Checkbutton(controls, text="Automatic state playback", variable=self.automatic, command=self.change_playback).pack(anchor="w", pady=(8, 0))
		ttk.Label(controls, text="Manual transition weight").pack(anchor="w", pady=(8, 0))
		self.fade_scale = ttk.Scale(controls, from_=0, to=1, variable=self.fade, command=self.change_fade)
		self.fade_scale.pack(fill="x")
		ttk.Label(controls, text="Sample bindings").pack(anchor="w", pady=(8, 0))
		self.binding_frame = ttk.Frame(controls)
		self.binding_frame.pack(fill="x")
		ttk.Label(controls, text="Play advances automatic states and fades. Other slot clips loop at the displayed clock. Root-motion extraction is disabled.", wraplength=370).pack(anchor="w", pady=10)
		ttk.Label(self.window, textvariable=self.status, wraplength=1050, padding=8).pack(fill="x")
		self.editor.insert("1.0", json.dumps(default_definition(animations), indent=2))
		self.compile_draft()
		self.window.after(33, self.tick)

	def compile_definition(self, data):
		return compile_graph(data, self.animations)

	def save_project(self, path=None):
		if path is None:
			from tkinter import filedialog
			path = filedialog.asksaveasfilename(title="Save animation project", parent=self.window, defaultextension=".animation.json", initialfile=self.project_path.name if self.project_path is not None else "animation.animation.json", filetypes=[("Animation project", "*.json")])
		if not path: return
		try:
			data = json.loads(self.editor.get("1.0", "end-1c"))
			if not isinstance(data, dict): raise ValueError("graph draft must be a JSON object")
			write_project(path, self.animations, data)
		except (OSError, ValueError, TypeError) as exc:
			self.status.set(f"Cannot save project: {exc}")
			return
		self.project_path = Path(path)
		self.status.set(f"Saved graph draft and markers to {path}. Preview unchanged.")

	def load_project(self, path=None):
		if path is None:
			from tkinter import filedialog
			path = filedialog.askopenfilename(title="Load animation project", parent=self.window, filetypes=[("Animation project", "*.json")])
		if not path: return
		try:
			definition, graph, timelines = read_project(path, self.animations)
			if definition is None: definition, graph = compile_graph(json.loads(self.editor.get("1.0", "end-1c")), self.animations, timelines)
		except (OSError, ValueError, TypeError) as exc:
			self.status.set(f"Cannot load project: {exc}")
			return
		for form in (self.transition_window, self.blend_space_window, self.blend_grid_window, self.sync_marker_window):
			if form is not None: form.close()
		for animation, (events, sync) in zip(self.animations, timelines): animation.events, animation.sync_markers = events, sync
		self.editor.delete("1.0", "end")
		self.editor.insert("1.0", json.dumps(definition.data, indent=2, ensure_ascii=False))
		self.editor.edit_reset()
		self.preview_enabled.set(True)
		self.apply_definition(definition, graph)
		self.project_path = Path(path)
		self.status.set(f"Loaded graph and markers from {path}.")

	def open_transitions(self):
		if self.transition_window is not None:
			self.transition_window.window.lift()
			return
		try:
			text = self.editor.get("1.0", "end-1c")
			definition, _ = self.compile_definition(json.loads(text))
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot edit transitions: {exc}")
			return
		from .animation_graph_forms import TransitionWindow
		self.transition_window = TransitionWindow(self, definition.data, text)

	def open_sync_markers(self):
		if self.sync_marker_window is not None:
			self.sync_marker_window.window.lift()
			return
		from .animation_graph_forms import SyncMarkerWindow
		self.sync_marker_window = SyncMarkerWindow(self)

	def open_blend_spaces(self):
		if self.blend_space_window is not None:
			self.blend_space_window.window.lift()
			return
		try:
			text = self.editor.get("1.0", "end-1c")
			definition, _ = self.compile_definition(json.loads(text))
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot edit 1D blends: {exc}")
			return
		from .animation_blend_space_forms import BlendSpaceWindow
		self.blend_space_window = BlendSpaceWindow(self, definition.data, text)

	def open_blend_grids(self):
		if self.blend_grid_window is not None:
			self.blend_grid_window.window.lift()
			return
		try:
			text = self.editor.get("1.0", "end-1c")
			definition, _ = self.compile_definition(json.loads(text))
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot edit 2D grids: {exc}")
			return
		from .animation_blend_space_forms import BlendGridWindow
		self.blend_grid_window = BlendGridWindow(self, definition.data, text)

	def compile_draft(self):
		try:
			definition, graph = self.compile_definition(json.loads(self.editor.get("1.0", "end-1c")))
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot compile draft: {exc}")
			return
		self.apply_definition(definition, graph)
		self.status.set("Compiled draft applied to this preview. No asset was written.")

	def apply_definition(self, definition, graph):
		self.definition, self.graph = definition, graph
		self.playback = GraphPlayback(graph)
		self.playing = False
		self.play_button.configure(text="Play")
		self.parameters = graph.parameter_defaults.copy()
		self.state_box.configure(values=tuple(graph.state_ids))
		self.state_box.current(graph.initial_state)
		for child in self.parameter_frame.winfo_children(): child.destroy()
		self.parameter_entries = {}
		for name, column in graph.parameter_columns.items():
			row = self.ttk.Frame(self.parameter_frame)
			row.pack(fill="x")
			self.ttk.Label(row, text=name, width=22).pack(side="left")
			value = self.tk.StringVar(value=f"{self.parameters[column]:.17g}")
			self.ttk.Entry(row, textvariable=value).pack(side="left", fill="x", expand=True)
			self.parameter_entries[name] = value
		for child in self.binding_frame.winfo_children(): child.destroy()
		self.bindings = {}
		for name in graph.sample_slots:
			self.ttk.Label(self.binding_frame, text=name).pack(anchor="w")
			value = self.tk.StringVar(value=self.STATE_SAMPLE)
			grids = (f"{self.GRID_PREFIX}{space} / {role}>" for space, grid in graph.blend_spaces_2d.items() for role, slot in zip(("lower", "upper"), grid.sample_slots) if slot == graph.sample_slots[name])
			box = self.ttk.Combobox(self.binding_frame, textvariable=value, state="readonly", values=(self.STATE_SAMPLE, *graph.clip_ids, *(f"{self.BLEND_PREFIX}{space}>" for space in graph.blend_spaces), *grids))
			box.pack(fill="x")
			box.bind("<<ComboboxSelected>>", lambda _, slot=name: self.change_binding(slot))
			self.bindings[name] = value
		self.change_state()

	def change_state(self, event=None):
		if self.graph is None: return
		self.playback.reset(self.graph.state_ids[self.state.get()])
		self.last_tick = time.monotonic()
		self.fade.set(0)
		self.evaluate_conditions()
		if self.preview_enabled.get(): self.select_clip(self.state_clip(self.graph.state_ids[self.state.get()]))
		self.draw_diagram()
		self.draw_viewport()

	def state_clip(self, state):
		"""Choose a representative imported clip for the shared asset viewport."""
		if state in self.graph.state_clips: return self.graph.state_clips[state]
		return int(self.graph.blend_spaces[self.graph.state_blend_spaces[state]].clip_ids[0])

	def apply_parameters(self):
		if self.graph is None: return
		try:
			parameters = np.array([float(self.parameter_entries[name].get()) for name in self.graph.parameter_columns])
			if not np.isfinite(parameters).all(): raise ValueError("parameters must be finite")
			weights = parameters[list(self.graph.weight_columns)]
			if np.any((weights < 0) | (weights > 1)): raise ValueError("layer weights must be in [0, 1]")
		except ValueError as exc:
			self.status.set(f"Cannot apply parameters: {exc}")
			return
		self.parameters = parameters
		if not self.automatic.get(): self.fade.set(0)
		self.evaluate_conditions()
		self.draw_diagram()
		self.draw_viewport()
		self.status.set("Preview parameters applied.")

	def evaluate_conditions(self):
		state = self.graph.state_ids[self.state.get()]
		if self.automatic.get() and self.playback.target_state is not None:
			selected, targets, durations = [True], [self.playback.target_state], [self.playback.duration]
		else: selected, targets, durations = self.graph.machine.select(np.array([state]), self.parameters[None], np.array([False]))
		self.target_state = int(targets[0]) if selected[0] else None
		if selected[0]:
			target = next(name for name, value in self.graph.state_ids.items() if value == self.target_state)
			pair = self.graph.state_clips.get(state), self.graph.state_clips.get(self.target_state)
			sync = ", marker sync" if pair in self.graph.sync_mappings else ""
			self.decision.set(f"{self.state.get()} → {target}, fade {durations[0]:g}s{sync}")
		else: self.decision.set("No transition selected")

	def change_playback(self):
		self.playback.reset(self.graph.state_ids[self.state.get()], self.position)
		self.preview_enabled.set(True)
		self.animation_id = self.state_clip(self.playback.state)
		self.clip_box.current(self.animation_id)
		self.position = self.playback.time
		self.clock.set(self.position)
		self.clock_label.configure(text=f"{self.position:.3f}s")
		self.fade_scale.configure(state="disabled" if self.automatic.get() else "normal")
		self.fade.set(0)
		self.last_tick = time.monotonic()
		self.evaluate_conditions()
		self.draw_diagram()
		self.draw_viewport()

	def change_fade(self, value):
		self.draw_viewport()

	def change_binding(self, slot):
		binding = self.bindings[slot].get()
		for name, grid in self.graph.blend_spaces_2d.items():
			lower, upper = (next(label for label, index in self.graph.sample_slots.items() if index == position) for position in grid.sample_slots)
			labels = (f"{self.GRID_PREFIX}{name} / lower>", f"{self.GRID_PREFIX}{name} / upper>")
			if binding in labels:
				self.bindings[lower].set(labels[0])
				self.bindings[upper].set(labels[1])
		for name, grid in self.graph.blend_spaces_2d.items():
			lower, upper = (next(label for label, index in self.graph.sample_slots.items() if index == position) for position in grid.sample_slots)
			labels = (f"{self.GRID_PREFIX}{name} / lower>", f"{self.GRID_PREFIX}{name} / upper>")
			if (self.bindings[lower].get() == labels[0]) != (self.bindings[upper].get() == labels[1]):
				for item, label in zip((lower, upper), labels):
					if self.bindings[item].get() == label: self.bindings[item].set(self.STATE_SAMPLE)
		self.draw_viewport()

	def change_preview(self):
		if self.graph is not None and self.preview_enabled.get(): self.change_state()
		else: self.draw_viewport()

	def select_clip(self, clip):
		self.animation_id = clip
		self.clip_box.current(clip)
		self.position = 0.0
		self.clock.set(0)
		self.clock_label.configure(text="0.000s")
		self.draw_viewport()

	def change_clip(self, event=None):
		self.preview_enabled.set(False)
		self.animation_id = self.clip_box.current()
		self.position = 0.0
		self.clock.set(0)
		self.clock_label.configure(text="0.000s")
		self.draw_viewport()

	def change_time(self, value):
		self.position = float(value)
		if self.automatic.get() and self.preview_enabled.get():
			self.playing = False
			self.play_button.configure(text="Play")
			self.playback.reset(self.graph.state_ids[self.state.get()], self.position)
			self.position = self.playback.time
			self.clock.set(self.position)
			self.fade.set(0)
			self.evaluate_conditions()
			self.draw_diagram()
		self.clock_label.configure(text=f"{self.position:.3f}s")
		self.draw_viewport()

	def toggle_play(self):
		self.playing = not self.playing
		self.last_tick = time.monotonic()
		self.play_button.configure(text="Pause" if self.playing else "Play")

	def tick(self):
		now = time.monotonic()
		if self.playing:
			try: self.advance_playback(now - self.last_tick)
			except (ValueError, FloatingPointError) as exc:
				self.playing = False
				self.play_button.configure(text="Play")
				self.status.set(f"Cannot advance preview: {exc}")
		self.last_tick = now
		self.window.after(33, self.tick)

	def advance_playback(self, dt):
		if self.automatic.get() and self.preview_enabled.get():
			self.playback.advance(dt, self.parameters)
			self.state.set(next(name for name, state in self.graph.state_ids.items() if state == self.playback.state))
			self.animation_id = self.state_clip(self.playback.state)
			self.clip_box.current(self.animation_id)
			self.position = self.playback.time
			self.fade.set(self.playback.elapsed / self.playback.duration if self.playback.target_state is not None else 0)
			self.evaluate_conditions()
			self.draw_diagram()
		else:
			state = self.graph.state_ids[self.state.get()] if self.preview_enabled.get() else None
			if state in self.graph.state_blend_spaces: self.position += dt
			else:
				duration = self.animations[self.animation_id].clip.duration
				self.position = (self.position + dt) % duration if duration > 0 else 0.0
		self.clock.set(self.position)
		self.clock_label.configure(text=f"{self.position:.3f}s")
		self.draw_viewport()

	def draw_viewport(self):
		animation = self.animations[self.animation_id]
		graph = self.graph if self.preview_enabled.get() else None
		try: model = self.joint_model(self.position) if graph is not None else None
		except (ValueError, IndexError, FloatingPointError) as exc:
			self.viewport.delete("all")
			self.viewport.create_text(8, 8, anchor="nw", fill="#ff9aa5", text=f"Preview unavailable: {exc}")
			return
		camera = draw_animation_viewport(self.viewport, animation, self.position, self.bounds if graph is not None else self.clip_bounds[self.animation_id], joint_model=model, show_mesh=self.show_mesh.get(), show_bones=self.show_bones.get())
		if camera is not None: self.camera_position, self.camera_target = camera

	def joint_model(self, time):
		graph = self.graph
		state = graph.state_ids[self.state.get()]
		fade = self.fade.get() if self.target_state is not None else 0
		poses, state_pose = [], None
		weights_for_plan = self.parameters[None, list(graph.weight_columns)].copy()
		grid_pairs = {}
		for name, grid in graph.blend_spaces_2d.items():
			lower, upper = (next(label for label, index in graph.sample_slots.items() if index == position) for position in grid.sample_slots)
			if self.bindings[lower].get() == f"{self.GRID_PREFIX}{name} / lower>" and self.bindings[upper].get() == f"{self.GRID_PREFIX}{name} / upper>":
				ids, horizontal, vertical = grid.select(self.parameters[None])
				grid_pairs[grid.sample_slots[0]] = (ids[0, 0], horizontal[0])
				grid_pairs[grid.sample_slots[1]] = (ids[0, 1], horizontal[0])
				weights_for_plan[0, grid.weight_slot] = vertical[0]
		for slot, value in enumerate(self.bindings.values()):
			binding = value.get()
			if slot not in grid_pairs and (binding == self.STATE_SAMPLE or binding.startswith(self.GRID_PREFIX)):
				if state_pose is None:
					if self.automatic.get(): state_pose = self.playback.pose(self.parameters)
					else:
						pair = graph.state_clips.get(state), graph.state_clips.get(self.target_state)
						mapping = graph.sync_mappings.get(pair)
						target_time = float(mapping.map([time])[0][0]) if mapping is not None else time
						state_pose = transition_pose(graph, state, self.target_state, time, target_time, fade, self.parameters)
				poses.append(state_pose)
				continue
			if slot in grid_pairs:
				ids, weights = grid_pairs[slot]
			elif binding.startswith(self.BLEND_PREFIX) and binding.endswith(">"):
				ids_array, weights_array = graph.blend_spaces[binding[len(self.BLEND_PREFIX):-1]].select(self.parameters[None])
				ids, weights = tuple(int(clip) for clip in ids_array[0]), tuple(float(weight) for weight in weights_array[0])
			else:
				ids, weights = (graph.clip_ids[binding],) * 2, (1, 0)
			times = [time % graph.clips[clip].duration if graph.clips[clip].duration > 0 else 0 for clip in ids]
			poses.append(sample_clip_pairs(self.asset.skeleton, graph.clips, np.array([ids]), np.array([times]), np.array([weights])))
		motions = [(None, np.array([False])) for _ in graph.root_motion_slots]
		pose = graph.input_plan.evaluate(graph.clips, (), weights_for_plan, poses=poses, root_motions=motions)
		return model_matrices(self.asset.skeleton, *pose)[0]

	def draw_diagram(self):
		canvas, graph = self.diagram, self.graph
		canvas.delete("all")
		if graph is None: return
		width = max(canvas.winfo_width(), 360)
		positions = {name: (width * (0.25 + (i % 2) * 0.5), 35 + (i // 2) * 65) for i, name in enumerate(graph.state_ids)}
		canvas.configure(scrollregion=(0, 0, width, max(230, (len(positions)+1)//2*65+15)))
		for edge in self.definition.data.get("transitions", []):
			a, b = positions[edge["source"]], positions[edge["target"]]
			canvas.create_line(*a, *b, fill="#758499", arrow="last")
		for name, (x, y) in positions.items():
			active = name == self.state.get()
			target = graph.state_ids[name] == self.target_state
			tag = f"state:{graph.state_ids[name]}"
			canvas.create_rectangle(x-75, y-17, x+75, y+17, fill="#34694b" if active else "#70552c" if target else "#354251", outline="#d2dbe6", tags=tag)
			canvas.create_text(x, y, text=name, width=145, fill="#ffffff", tags=tag)
			canvas.tag_bind(tag, "<Button-1>", lambda _, state=name: self.pick_state(state))

	def pick_state(self, name):
		self.state.set(name)
		self.change_state()

	def close(self):
		self.preview_enabled.set(False)
		self.window.destroy()
		if self.closed is not None: self.closed()


def main(argv=None):
	"""Open a supported skinned glTF/GLB in the standalone graph editor."""
	import argparse
	import tkinter as tk
	from tkinter import filedialog, messagebox

	parser = argparse.ArgumentParser(description="Inspect and configure an in-memory animation graph")
	parser.add_argument("file", nargs="?", help="Animated glTF or GLB file")
	args = parser.parse_args(argv)
	root = tk.Tk()
	root.withdraw()
	path = args.file or filedialog.askopenfilename(title="Open animated glTF", filetypes=[("glTF files", "*.gltf *.glb"), ("All files", "*")])
	if not path:
		root.destroy()
		return 0
	try:
		animations = load_animations(path)
		if not animations or animations[0].preview.asset is None or any(animation.clip is None for animation in animations): raise ValueError("Graph configuration requires a skinned asset supported by the runtime importer")
	except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
		messagebox.showerror("Cannot open animation graph", str(exc), parent=root)
		root.destroy()
		return 1
	GraphWindow(root, animations, closed=root.quit)
	root.mainloop()
	root.destroy()
	return 0


if __name__ == "__main__": raise SystemExit(main())

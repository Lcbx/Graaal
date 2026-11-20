"""In-memory graph configuration and pose inspection for the timeline tool."""
import json

import numpy as np

from graaal.AnimationGraph import AnimationGraphDefinition
from graaal.SkeletalAnimation import model_matrices
from .animation_preview import animation_bounds


def clip_resources(animations):
	return {f"{i}: {animation.name}": animation.clip for i, animation in enumerate(animations)}


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
	"""Compile a draft and preview explicit samples without owning playback."""
	STATE_SAMPLE = "<state / transition>"

	def __init__(self, root, animations, *, select_clip, redraw, closed):
		import tkinter as tk
		from tkinter import ttk

		self.tk, self.ttk = tk, ttk
		self.animations, self.asset = animations, animations[0].preview.asset
		self.select_clip, self.redraw, self.closed = select_clip, redraw, closed
		self.graph, self.parameters, self.target_state = None, None, None
		self.parameter_entries, self.bindings = {}, {}
		self.window = tk.Toplevel(root)
		self.window.title("Animation graph configuration")
		self.window.geometry("1120x820")
		self.window.minsize(850, 650)
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Edit the definition, then compile to preview it.")
		self.state = tk.StringVar()
		self.decision = tk.StringVar(value="No transition selected")
		self.fade = tk.DoubleVar(value=0)
		self.preview_enabled = tk.BooleanVar(value=True)
		bounds = [animation_bounds(animation) for animation in animations]
		self.bounds = np.min([bound[0] for bound in bounds], axis=0), np.max([bound[1] for bound in bounds], axis=0)

		toolbar = ttk.Frame(self.window, padding=8)
		toolbar.pack(fill="x")
		ttk.Button(toolbar, text="Compile draft", command=self.compile_draft).pack(side="left")
		ttk.Checkbutton(toolbar, text="Preview graph in viewport", variable=self.preview_enabled, command=self.change_preview).pack(side="left", padx=10)
		ttk.Label(toolbar, text="Draft edits stay in memory. Playback uses the timeline.").pack(side="left")
		panes = ttk.Panedwindow(self.window, orient="horizontal")
		panes.pack(fill="both", expand=True, padx=8)
		draft_frame, controls = ttk.Frame(panes), ttk.Frame(panes)
		panes.add(draft_frame, weight=3)
		panes.add(controls, weight=2)
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
		ttk.Label(controls, text="Manual transition weight").pack(anchor="w", pady=(8, 0))
		ttk.Scale(controls, from_=0, to=1, variable=self.fade, command=self.change_fade).pack(fill="x")
		ttk.Label(controls, text="Sample bindings").pack(anchor="w", pady=(8, 0))
		self.binding_frame = ttk.Frame(controls)
		self.binding_frame.pack(fill="x")
		ttk.Label(controls, text="Slot clips loop at timeline time. State samples use the manual fade. Root-motion extraction is disabled.", wraplength=370).pack(anchor="w", pady=10)
		ttk.Label(self.window, textvariable=self.status, wraplength=1050, padding=8).pack(fill="x")
		self.editor.insert("1.0", json.dumps(default_definition(animations), indent=2))
		self.compile_draft()

	def compile_draft(self):
		try:
			definition = AnimationGraphDefinition(json.loads(self.editor.get("1.0", "end-1c")))
			tracks = {name: animation.sync_markers.snapshot()[1] for name, animation in zip(clip_resources(self.animations), self.animations) if animation.sync_markers is not None}
			graph = definition.compile(self.asset.skeleton, clip_resources(self.animations), sync_tracks=tracks)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot compile draft: {exc}")
			return
		self.definition, self.graph = definition, graph
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
			box = self.ttk.Combobox(self.binding_frame, textvariable=value, state="readonly", values=(self.STATE_SAMPLE, *graph.clip_ids))
			box.pack(fill="x")
			box.bind("<<ComboboxSelected>>", lambda _: self.redraw())
			self.bindings[name] = value
		self.change_state()
		self.status.set("Compiled draft applied to this preview. No asset was written.")

	def change_state(self, event=None):
		if self.graph is None: return
		self.fade.set(0)
		self.evaluate_conditions()
		if self.preview_enabled.get(): self.select_clip(self.graph.state_clips[self.graph.state_ids[self.state.get()]])
		self.draw_diagram()
		self.redraw()

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
		self.fade.set(0)
		self.evaluate_conditions()
		self.draw_diagram()
		self.redraw()
		self.status.set("Preview parameters applied.")

	def evaluate_conditions(self):
		state = self.graph.state_ids[self.state.get()]
		selected, targets, durations = self.graph.machine.select(np.array([state]), self.parameters[None], np.array([False]))
		self.target_state = int(targets[0]) if selected[0] else None
		if selected[0]:
			target = next(name for name, value in self.graph.state_ids.items() if value == self.target_state)
			pair = self.graph.state_clips[state], self.graph.state_clips[self.target_state]
			sync = ", marker sync" if pair in self.graph.sync_mappings else ""
			self.decision.set(f"{self.state.get()} → {target}, fade {durations[0]:g}s{sync}")
		else: self.decision.set("No transition selected")

	def change_fade(self, value):
		self.redraw()

	def change_preview(self):
		if self.graph is not None and self.preview_enabled.get(): self.change_state()
		else: self.redraw()

	def joint_model(self, time):
		graph = self.graph
		state = graph.state_ids[self.state.get()]
		source = graph.state_clips[state]
		target = source if self.target_state is None else graph.state_clips[self.target_state]
		fade = self.fade.get() if self.target_state is not None else 0
		samples = []
		for value in self.bindings.values():
			ids = (source, target) if value.get() == self.STATE_SAMPLE else (graph.clip_ids[value.get()],) * 2
			times = [time % graph.clips[clip].duration if graph.clips[clip].duration > 0 else 0 for clip in ids]
			mapping = graph.sync_mappings.get(ids) if value.get() == self.STATE_SAMPLE else None
			if mapping is not None: times[1] = mapping.map([times[0]])[0][0]
			weights = (1-fade, fade) if value.get() == self.STATE_SAMPLE else (1, 0)
			samples.append((np.array([ids]), np.array([times]), np.array([weights])))
		motions = [(None, np.array([False])) for _ in graph.root_motion_slots]
		pose = graph.plan.evaluate(graph.clips, samples, self.parameters[None, list(graph.weight_columns)], root_motions=motions)
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
		self.closed()

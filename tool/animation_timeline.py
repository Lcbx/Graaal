"""Inspect glTF animation tracks and edit temporary event and sync markers.

Run from the repository root: python -m tool.animation_timeline [file.gltf|file.glb]
"""
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import argparse
import time

import numpy as np

from graaal.AnimationEvents import EventTimeline, EventTrack
from graaal.Assets import read_gltf_accessor
from .animation_preview import animation_bounds, camera_from_bounds, load_preview, mesh_triangles, pose_matrices, project


@dataclass(frozen=True)
class Track:
	label: str
	interpolation: str
	times: np.ndarray
	values: np.ndarray
	node: int | None = None
	path: str = ""
	tangents: np.ndarray | None = None


@dataclass
class Animation:
	name: str
	duration: float
	tracks: tuple[Track, ...]
	events: EventTimeline | None
	preview: object = None
	clip: object = None
	sync_markers: EventTimeline | None = None

	def __post_init__(self):
		duration = self.clip.duration if self.clip is not None else self.duration
		if self.sync_markers is None and duration > 0: self.sync_markers = EventTimeline(EventTrack(duration, [], []))


class MarkerKind(StrEnum):
	EVENTS = "Events"
	SYNC = "Sync markers"


def load_animations(path):
	"""Read source channels and load supported runtime clips for the viewport."""
	from pygltflib import GLTF2

	path = Path(path)
	gltf = GLTF2().load(str(path))
	preview = load_preview(gltf, path)
	animations = []
	for animation_id, animation in enumerate(gltf.animations or ()):
		tracks = []
		for channel in animation.channels or ():
			if channel.sampler is None or not 0 <= channel.sampler < len(animation.samplers): raise ValueError("Animation references an invalid sampler")
			sampler = animation.samplers[channel.sampler]
			if sampler.input is None or sampler.output is None or not 0 <= sampler.input < len(gltf.accessors) or not 0 <= sampler.output < len(gltf.accessors):
				raise ValueError("Animation sampler references an invalid accessor")
			times = read_gltf_accessor(gltf, sampler.input, path.parent)
			values = read_gltf_accessor(gltf, sampler.output, path.parent)
			if times.ndim != 2 or times.shape[1] != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times[:, 0]) <= 0):
				raise ValueError("Animation key times must be finite, nonnegative and strictly increasing")
			interpolation = sampler.interpolation or "LINEAR"
			factor = 3 if interpolation == "CUBICSPLINE" else 1
			if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE") or values.size == 0 or values.size % (len(times) * factor) or not np.isfinite(values).all():
				raise ValueError("Invalid animation sampler values or interpolation")
			samples = values.reshape(len(times), factor, -1)
			tangents = samples[:, (0, 2)].copy() if factor == 3 else None
			values = samples[:, 1 if factor == 3 else 0].copy()
			times = times[:, 0].astype(np.float64, copy=True)
			times.flags.writeable = False
			values.flags.writeable = False
			target = channel.target
			if target is None: raise ValueError("Animation channel has no target")
			node = target.node
			if node is None: node_name = "Unbound target"
			elif not 0 <= node < len(gltf.nodes): raise ValueError("Animation references an invalid node")
			else: node_name = f"{gltf.nodes[node].name or 'Node'} [{node}]"
			label = f"{node_name} / {target.path or 'unknown'}"
			if target.path in ("translation", "scale", "rotation") and values.shape[1] != (4 if target.path == "rotation" else 3): raise ValueError("Invalid transform channel width")
			tracks.append(Track(label, interpolation, times, values, node, target.path or "", tangents))
		duration = max((float(track.times[-1]) for track in tracks), default=0.0)
		events = EventTimeline(EventTrack(duration, [], [])) if duration > 0 else None
		clip = preview.asset.clips[animation_id][1] if preview.asset is not None else None
		animations.append(Animation(animation.name or f"Animation {animation_id}", duration, tuple(tracks), events, preview, clip))
	return animations


class TimelineWindow:
	"""Tk interface for track keyframes and unsaved per-animation events."""
	LABEL_WIDTH = 250
	HEADER_HEIGHT = 42
	ROW_HEIGHT = 27

	def __init__(self, root, path=None):
		import tkinter as tk
		from tkinter import ttk

		self.tk, self.ttk, self.root = tk, ttk, root
		self.animations, self.animation_id = [], -1
		self.position, self.playing, self.last_tick = 0.0, False, time.monotonic()
		self.path = tk.StringVar(value="No file loaded")
		self.status = tk.StringVar(value="Open a glTF or GLB file with animation.")
		self.clip = tk.StringVar()
		self.event_name = tk.StringVar(value="event")
		self.event_time = tk.StringVar(value="0.000")
		self.marker_kind = tk.StringVar(value=MarkerKind.EVENTS)
		self.event_preview = tk.StringVar(value="Preview events: none")
		self.drag_event_id = None
		self.graph_window = None
		self.zoom = tk.DoubleVar(value=100)
		self.auto_fit, self.bounds = True, None
		self.show_mesh, self.show_bones = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
		root.title("Animation timeline")
		root.geometry("1400x780")
		root.minsize(680, 400)

		file_row = ttk.Frame(root, padding=8)
		file_row.pack(fill="x")
		ttk.Button(file_row, text="Open glTF / GLB", command=self.open_file).pack(side="left")
		ttk.Button(file_row, text="Graph configuration", command=self.open_graph).pack(side="left", padx=5)
		ttk.Label(file_row, textvariable=self.path).pack(side="left", padx=10, fill="x", expand=True)
		clip_row = ttk.Frame(root, padding=(8, 0, 8, 8))
		clip_row.pack(fill="x")
		ttk.Label(clip_row, text="Animation").pack(side="left")
		self.clip_box = ttk.Combobox(clip_row, textvariable=self.clip, state="readonly", width=38)
		self.clip_box.pack(side="left", padx=8)
		self.clip_box.bind("<<ComboboxSelected>>", self.select_animation)
		self.play_button = ttk.Button(clip_row, text="Play", command=self.toggle_play)
		self.play_button.pack(side="left", padx=4)
		ttk.Label(clip_row, text="Zoom").pack(side="left", padx=(15, 0))
		self.zoom_control = ttk.Scale(clip_row, from_=25, to=400, variable=self.zoom, command=self.change_zoom)
		self.zoom_control.pack(side="left", fill="x", expand=True, padx=6)
		ttk.Button(clip_row, text="Fit timeline", command=self.fit_timeline).pack(side="left")

		panes = ttk.Panedwindow(root, orient="horizontal")
		panes.pack(fill="both", expand=True)
		timeline_frame = ttk.Frame(panes, padding=(8, 0), width=850)
		panes.add(timeline_frame, weight=3)
		self.canvas = tk.Canvas(timeline_frame, background="#20242b", highlightthickness=0)
		self.canvas.grid(row=0, column=0, sticky="nsew")
		vscroll = ttk.Scrollbar(timeline_frame, orient="vertical", command=self.canvas.yview)
		vscroll.grid(row=0, column=1, sticky="ns")
		hscroll = ttk.Scrollbar(timeline_frame, orient="horizontal", command=self.canvas.xview)
		hscroll.grid(row=1, column=0, sticky="ew")
		self.canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
		timeline_frame.columnconfigure(0, weight=1)
		timeline_frame.rowconfigure(0, weight=1)
		self.canvas.bind("<Button-1>", self.click_timeline)
		self.canvas.bind("<B1-Motion>", self.drag_timeline)
		self.canvas.bind("<ButtonRelease-1>", self.end_drag)
		self.canvas.bind("<Configure>", self.resize_timeline)
		view_frame = ttk.Frame(panes, width=450)
		panes.add(view_frame, weight=2)
		view_controls = ttk.Frame(view_frame)
		view_controls.pack(fill="x")
		ttk.Label(view_controls, text="Animation viewport").pack(side="left", padx=5)
		ttk.Checkbutton(view_controls, text="Mesh", variable=self.show_mesh, command=self.draw_viewport).pack(side="left")
		ttk.Checkbutton(view_controls, text="Bones", variable=self.show_bones, command=self.draw_viewport).pack(side="left")
		self.viewport = tk.Canvas(view_frame, background="#171c24", highlightthickness=0)
		self.viewport.pack(fill="both", expand=True)
		self.viewport.bind("<Configure>", lambda _: self.draw_viewport())

		event_row = ttk.Frame(root, padding=8)
		event_row.pack(fill="x")
		marker_box = ttk.Combobox(event_row, textvariable=self.marker_kind, state="readonly", values=tuple(MarkerKind), width=13)
		marker_box.pack(side="left", padx=(0, 8))
		marker_box.bind("<<ComboboxSelected>>", self.change_marker_kind)
		self.marker_name_label = ttk.Label(event_row, text="Event name")
		self.marker_name_label.pack(side="left")
		ttk.Entry(event_row, textvariable=self.event_name, width=20).pack(side="left", padx=5)
		ttk.Label(event_row, text="Time (s)").pack(side="left")
		ttk.Entry(event_row, textvariable=self.event_time, width=10).pack(side="left", padx=5)
		ttk.Button(event_row, text="Use playhead", command=self.use_playhead).pack(side="left", padx=3)
		ttk.Button(event_row, text="Add marker", command=self.add_event).pack(side="left", padx=3)
		ttk.Button(event_row, text="Update selected", command=self.update_event).pack(side="left", padx=3)
		ttk.Button(event_row, text="Remove selected", command=self.remove_event).pack(side="left", padx=3)
		self.event_list = ttk.Treeview(root, columns=("time", "name"), show="headings", height=4, selectmode="browse")
		self.event_list.heading("time", text="Time (s)")
		self.event_list.heading("name", text="Event")
		self.event_list.column("time", width=120, stretch=False)
		self.event_list.column("name", width=300)
		self.event_list.pack(fill="x", padx=8)
		self.event_list.bind("<<TreeviewSelect>>", self.select_event)
		ttk.Label(root, textvariable=self.event_preview, anchor="w", padding=(8, 4)).pack(fill="x")
		ttk.Label(root, textvariable=self.status, anchor="w", padding=8).pack(fill="x")
		if path is not None: self.load(path)
		self.draw()
		root.after(33, self.tick)

	@property
	def animation(self):
		return self.animations[self.animation_id] if 0 <= self.animation_id < len(self.animations) else None

	@property
	def marker_timeline(self):
		if self.animation is None: return None
		return self.animation.sync_markers if self.marker_kind.get() == MarkerKind.SYNC else self.animation.events

	def change_marker_kind(self, event=None):
		self.event_list.selection_remove(*self.event_list.selection())
		self.drag_event_id = None
		sync = self.marker_kind.get() == MarkerKind.SYNC
		self.marker_name_label.configure(text="Sync marker name" if sync else "Event name")
		self.event_list.heading("name", text="Sync marker" if sync else "Event")
		self.refresh_events()
		self.draw()

	def open_file(self):
		from tkinter import filedialog, messagebox
		path = filedialog.askopenfilename(title="Open animated glTF", filetypes=[("glTF files", "*.gltf *.glb"), ("All files", "*")])
		if path:
			if any(track and track.snapshot()[0] for animation in self.animations for track in (animation.events, animation.sync_markers)):
				if not messagebox.askyesno("Discard markers?", "Opening another file will discard the current in-memory events and sync markers. Continue?", parent=self.root): return
			self.load(path)

	def load(self, path):
		try: animations = load_animations(path)
		except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
			self.status.set(f"Could not load file: {exc}")
			return
		if self.graph_window is not None: self.graph_window.close()
		self.animations, self.animation_id, self.position, self.playing = animations, -1, 0.0, False
		self.drag_event_id = None
		self.event_preview.set("Preview events: none")
		self.play_button.configure(text="Play")
		self.path.set(str(path))
		self.clip_box.configure(values=[f"{i}: {clip.name}" for i, clip in enumerate(animations)])
		if animations:
			self.clip_box.current(0)
			self.select_animation()
		else:
			self.clip.set("")
			self.status.set("This file contains no animations.")
			self.refresh_events()
			self.draw()

	def select_animation(self, event=None, *, graph_preview=False):
		if self.graph_window is not None and not graph_preview: self.graph_window.preview_enabled.set(False)
		self.event_list.selection_remove(*self.event_list.selection())
		self.animation_id = self.clip_box.current()
		self.position, self.playing = 0.0, False
		self.drag_event_id = None
		self.event_preview.set("Preview events: none")
		self.play_button.configure(text="Play")
		self.event_time.set("0.000")
		animation = self.animation
		if animation:
			self.status.set(f"{len(animation.tracks)} tracks, {animation.duration:.3f} s. Events are temporary.")
			self.bounds = animation_bounds(animation)
		self.auto_fit = True
		self.refresh_events()
		self.fit_timeline()

	def open_graph(self):
		if self.graph_window is not None:
			self.graph_window.window.lift()
			return
		if not self.animations or self.animations[0].preview.asset is None:
			self.status.set("Graph configuration requires a skinned asset supported by the runtime importer.")
			return
		from .animation_graph import GraphWindow
		self.graph_window = GraphWindow(self.root, self.animations, select_clip=self.select_graph_clip, redraw=self.draw_viewport, closed=self.close_graph)
		self.draw_viewport()

	def select_graph_clip(self, clip):
		self.clip_box.current(clip)
		self.select_animation(graph_preview=True)

	def close_graph(self):
		self.graph_window = None
		self.draw_viewport()

	def fit_timeline(self):
		self.auto_fit = True
		if self.animation:
			fit = max(self.canvas.winfo_width() - self.LABEL_WIDTH - 30, 50) / max(self.animation.duration, 1e-6)
			self.zoom_control.configure(from_=fit / 10, to=fit * 10)
			self.zoom.set(fit)
			self.canvas.xview_moveto(0)
		self.draw()

	def change_zoom(self, value):
		self.auto_fit = False
		self.draw()

	def resize_timeline(self, event=None):
		if self.auto_fit: self.fit_timeline()
		else: self.draw()

	def toggle_play(self):
		if not self.animation or self.animation.duration <= 0: return
		self.playing = not self.playing
		self.last_tick = time.monotonic()
		self.play_button.configure(text="Pause" if self.playing else "Play")

	def tick(self):
		now = time.monotonic()
		self.advance_playback(now - self.last_tick)
		self.last_tick = now
		self.root.after(33, self.tick)

	def advance_playback(self, dt):
		animation = self.animation
		if not self.playing or not animation or animation.duration <= 0: return
		end = self.position + dt
		if animation.events:
			_, track = animation.events.snapshot()
			occurrences = track.between(self.position, end)
			if occurrences:
				self.event_preview.set("Preview events: " + "; ".join(f"{event.name} @ {event.time % animation.duration:.3f}s" for event in occurrences[-8:]))
		self.position = end % animation.duration
		self.move_playhead()

	def use_playhead(self):
		self.event_time.set(f"{self.position:.17g}")

	def add_event(self):
		animation = self.animation
		if not animation or self.marker_timeline is None:
			self.status.set("Select an animation with a positive duration to add markers.")
			return
		try: event_id = self.marker_timeline.add(float(self.event_time.get()), self.event_name.get())
		except ValueError as exc:
			self.status.set(f"Cannot add marker: {exc}. Time must be in [0, {self.marker_timeline.duration:g}).")
			return
		self.refresh_events(event_id)
		self.draw()
		self.status.set("Marker added in memory.")

	def update_event(self):
		selection = self.event_list.selection()
		if not selection or self.marker_timeline is None: return
		try: self.marker_timeline.update(int(selection[0]), time=float(self.event_time.get()), name=self.event_name.get())
		except ValueError as exc:
			self.status.set(f"Cannot update marker: {exc}. Time must be in [0, {self.marker_timeline.duration:g}).")
			return
		self.refresh_events(int(selection[0]))
		self.select_event()
		self.status.set("Marker updated in memory.")

	def remove_event(self):
		selection = self.event_list.selection()
		if not selection or self.marker_timeline is None: return
		self.marker_timeline.remove(int(selection[0]))
		self.drag_event_id = None
		self.refresh_events()
		self.draw()

	def refresh_events(self, selected=None):
		selection = self.event_list.selection()
		if selected is None and selection: selected = int(selection[0])
		self.event_list.delete(*self.event_list.get_children())
		if self.marker_timeline is None: return
		ids, track = self.marker_timeline.snapshot()
		for event_id, event_time, name in zip(ids, track.times, track.names):
			self.event_list.insert("", "end", iid=str(event_id), values=(f"{event_time:.9f}", name))
		if selected in ids:
			self.event_list.selection_set(str(selected))
			self.event_list.see(str(selected))

	def select_event(self, event=None):
		selection = self.event_list.selection()
		if not selection or self.marker_timeline is None: return
		ids, track = self.marker_timeline.snapshot()
		selected = ids.index(int(selection[0]))
		self.position, self.playing = float(track.times[selected]), False
		self.play_button.configure(text="Play")
		self.event_name.set(track.names[selected])
		self.use_playhead()
		self.draw()

	def click_timeline(self, event):
		self.drag_event_id = None
		x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
		markers = [tag for item in self.canvas.find_overlapping(x-6, y-6, x+6, y+6) for tag in self.canvas.gettags(item) if tag.startswith(("event:", "sync:"))]
		if markers:
			selection = self.event_list.selection()
			prefix = "sync" if self.marker_kind.get() == MarkerKind.SYNC else "event"
			current = f"{prefix}:{selection[0]}" if selection else None
			kind, selected = (current if current in markers else markers[-1]).split(":")
			wanted = MarkerKind.SYNC if kind == "sync" else MarkerKind.EVENTS
			if self.marker_kind.get() != wanted:
				self.marker_kind.set(wanted)
				self.change_marker_kind()
			self.event_list.selection_set(selected)
			self.select_event()
			self.drag_event_id = int(selected)
			return
		self.scrub_timeline(event)

	def drag_timeline(self, event):
		if self.drag_event_id is None:
			self.scrub_timeline(event)
			return
		animation = self.animation
		if not animation or self.marker_timeline is None: return
		x = self.canvas.canvasx(event.x)
		self.position = float(np.clip((x - self.LABEL_WIDTH) / self.zoom.get(), 0, np.nextafter(self.marker_timeline.duration, 0)))
		self.marker_timeline.update(self.drag_event_id, time=self.position)
		self.refresh_events(self.drag_event_id)
		self.use_playhead()
		self.draw()

	def end_drag(self, event=None):
		self.drag_event_id = None

	def scrub_timeline(self, event):
		animation = self.animation
		if not animation: return
		self.playing = False
		self.play_button.configure(text="Play")
		x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
		self.position = min(max((x - self.LABEL_WIDTH) / self.zoom.get(), 0), animation.duration)
		self.use_playhead()
		row = int((y - self.HEADER_HEIGHT) // self.ROW_HEIGHT)
		if 0 <= row < len(animation.tracks):
			track = animation.tracks[row]
			key = int(np.argmin(np.abs(track.times - self.position)))
			if abs(track.times[key] - self.position) * self.zoom.get() <= 7:
				self.position = float(track.times[key])
				self.use_playhead()
				self.status.set(f"{track.label}: {track.interpolation}, {self.position:.6f} s, value {np.array2string(track.values[key], precision=4)}")
		self.move_playhead()

	def move_playhead(self):
		if not self.animation: return
		x = self.LABEL_WIDTH + self.position * self.zoom.get()
		self.canvas.coords("playhead_line", x, self.HEADER_HEIGHT - 5, x, self.draw_height)
		self.canvas.coords("playhead_label", x + 4, 5)
		self.canvas.itemconfigure("playhead_label", text=f"{self.position:.3f}s")
		self.draw_viewport()

	def draw_viewport(self):
		canvas, animation = self.viewport, self.animation
		canvas.delete("all")
		if not animation or not len(animation.preview.nodes): return
		width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
		try:
			graph = self.graph_window
			graph = graph if graph is not None and graph.preview_enabled.get() and graph.graph is not None else None
			model = pose_matrices(animation, self.position, joint_model=graph.joint_model(self.position) if graph is not None else None)
			self.camera_position, self.camera_target = camera_from_bounds(graph.bounds if graph is not None else self.bounds, width / height)
			points = model[:, :3, 3]
			screen, depth = project(points, self.camera_position, self.camera_target, width, height)
			if self.show_mesh.get():
				triangles = mesh_triangles(animation.preview, model)
				if len(triangles):
					vertices, distances = project(triangles.reshape(-1, 3), self.camera_position, self.camera_target, width, height)
					vertices, distances = vertices.reshape(-1, 6), distances.reshape(-1, 3)
					normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
					normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
					brightness = 0.35 + 0.65 * np.abs(normal @ (np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])))
					for face in np.argsort(distances.mean(axis=1))[::-1]:
						if np.any(distances[face] <= 0): continue
						color = tuple((np.array([100, 185, 125]) * brightness[face]).astype(int))
						canvas.create_polygon(*vertices[face], fill="#%02x%02x%02x" % color, outline="")
			if self.show_bones.get():
				bones = set(map(int, animation.preview.bones))
				for bone in bones:
					parent = int(animation.preview.parents[bone])
					while parent != -1 and parent not in bones: parent = int(animation.preview.parents[parent])
					if parent != -1 and min(depth[bone], depth[parent]) > 0: canvas.create_line(*screen[parent], *screen[bone], fill="#f4c96a", width=2)
					if depth[bone] > 0:
						x, y = screen[bone]
						canvas.create_oval(x-3, y-3, x+3, y+3, fill="#ffe3a3", outline="")
			canvas.create_text(8, 8, anchor="nw", fill="#bbc6d5", width=max(width-16, 1), text=animation.preview.message)
		except (ValueError, IndexError, FloatingPointError) as exc:
			canvas.create_text(8, 8, anchor="nw", fill="#ff9aa5", text=f"Preview unavailable: {exc}", width=max(width-16, 1))

	def draw(self):
		canvas, animation = self.canvas, self.animation
		canvas.delete("all")
		if not animation:
			canvas.create_text(20, 20, anchor="nw", fill="#d7dde8", text="Open a glTF or GLB file to inspect its animations.")
			canvas.configure(scrollregion=(0, 0, 1, 1))
			self.draw_viewport()
			return
		width = max(canvas.winfo_width(), self.LABEL_WIDTH + animation.duration * self.zoom.get() + 30)
		height = max(canvas.winfo_height(), self.HEADER_HEIGHT + (len(animation.tracks) + 2) * self.ROW_HEIGHT + 10)
		self.draw_height = height
		canvas.configure(scrollregion=(0, 0, width, height))
		canvas.create_text(10, 20, anchor="w", fill="#e8eef9", text=f"{animation.name}  ({animation.duration:.3f} s)")
		step = max(0.001, 10 ** np.floor(np.log10(80 / self.zoom.get())))
		while step * self.zoom.get() < 60: step *= 2
		for tick in np.arange(0, animation.duration + step * 0.5, step):
			x = self.LABEL_WIDTH + tick * self.zoom.get()
			canvas.create_line(x, self.HEADER_HEIGHT - 5, x, height, fill="#383e48")
			canvas.create_text(x + 3, self.HEADER_HEIGHT - 10, anchor="sw", fill="#aab4c4", text=f"{tick:g}s")
		for row, track in enumerate(animation.tracks):
			y = self.HEADER_HEIGHT + row * self.ROW_HEIGHT + self.ROW_HEIGHT / 2
			canvas.create_text(10, y, anchor="w", fill="#d7dde8", text=f"{track.label}  ({track.interpolation})", width=self.LABEL_WIDTH - 18)
			canvas.create_line(self.LABEL_WIDTH, y, width, y, fill="#45505f")
			for key_time in track.times:
				x = self.LABEL_WIDTH + key_time * self.zoom.get()
				canvas.create_polygon(x, y - 5, x + 4, y, x, y + 5, x - 4, y, fill="#75c7ff", outline="")
		for row, (kind, timeline, prefix, color) in enumerate(((MarkerKind.EVENTS, animation.events, "event", "#ffc56c"), (MarkerKind.SYNC, animation.sync_markers, "sync", "#ab9bff"))):
			event_y = self.HEADER_HEIGHT + (len(animation.tracks) + row + 0.5) * self.ROW_HEIGHT
			canvas.create_text(10, event_y, anchor="w", fill=color, text=kind)
			canvas.create_line(self.LABEL_WIDTH, event_y, width, event_y, fill="#45505f")
			if timeline is None: continue
			ids, track = timeline.snapshot()
			selection = self.event_list.selection() if self.marker_kind.get() == kind else ()
			for event_id, event_time in zip(ids, track.times):
				x = self.LABEL_WIDTH + event_time * self.zoom.get()
				selected = str(event_id) in selection
				canvas.create_oval(x - 5, event_y - 5, x + 5, event_y + 5, fill="#fff1bb" if selected else color, outline="#ffffff" if selected else "", tags=f"{prefix}:{event_id}")
		playhead = self.LABEL_WIDTH + self.position * self.zoom.get()
		canvas.create_line(playhead, self.HEADER_HEIGHT - 5, playhead, height, fill="#ff7684", width=2, tags="playhead_line")
		canvas.create_text(playhead + 4, 5, anchor="nw", fill="#ff9aa5", text=f"{self.position:.3f}s", tags="playhead_label")
		self.draw_viewport()


def main():
	parser = argparse.ArgumentParser(description="Inspect glTF animation tracks and edit temporary event/sync markers")
	parser.add_argument("path", nargs="?", help="glTF or GLB file to open")
	args = parser.parse_args()
	import tkinter as tk
	root = tk.Tk()
	TimelineWindow(root, args.path)
	root.mainloop()


if __name__ == "__main__":
	main()

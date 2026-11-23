"""Edit ragdoll colliders and passive joint settings with a live physics preview."""

import argparse
from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np

from graaal.Math import Quaternion
from .ragdoll_preview import ColliderViewport, RagdollPreview, SCENARIOS, TICK_HZ
from .ragdoll_project import GAINS, SHAPE_FIELDS, default_project, open_project, write_project


PROJECT_FILES = (("Ragdoll JSON", "*.json"), ("All files", "*"))


def _text(values):
	return " ".join(f"{value:.15g}" for value in np.atleast_1d(values))


def _numbers(text, count, label):
	try: values = tuple(float(value) for value in text.replace(",", " ").split())
	except ValueError: values = ()
	if len(values) != count or not np.isfinite(values).all():
		raise ValueError(f"{label} needs {count} finite {'value' if count == 1 else 'values'}")
	return values


def _euler_degrees(turn):
	matrix = Quaternion(turn).matrix33
	cos_pitch = np.hypot(*matrix[0, :2])
	pitch = np.arctan2(matrix[0, 2], cos_pitch)
	if cos_pitch > 1e-10:
		roll = np.arctan2(-matrix[1, 2], matrix[2, 2])
		yaw = np.arctan2(-matrix[0, 1], matrix[0, 0])
	else:
		roll, yaw = np.arctan2(matrix[2, 1], matrix[1, 1]), 0
	return np.rad2deg([roll, pitch, yaw])


def _bone_fields(project, index):
	bone = project.settings.bones[index]
	shape = bone.shape.kind.name.capitalize()
	return {
		"shape": shape,
		"dimensions": _text(bone.shape.params[:SHAPE_FIELDS[shape][1]]),
		"mass": _text(bone.mass),
		"shape_offset": _text(bone.shape_offset),
		"shape_rotation": _text(_euler_degrees(bone.shape_rotation)),
		"twist": _text(np.rad2deg(bone.twist)),
		"swing": _text(np.rad2deg(bone.swing)),
		"bend": _text(np.rad2deg(bone.bend)),
		"twist_axis": "" if bone.twist_axis is None else _text(bone.twist_axis),
		"plane_axis": "" if bone.plane_axis is None else _text(bone.plane_axis),
		"friction": _text(bone.friction),
		"neutral": _text(_euler_degrees(project.neutral[index])),
		**{f"spring_{key}": _text(project.spring[key][index]) for key in GAINS},
		"margin": _text(np.rad2deg(project.soft_limit["margin"][index])),
		**{f"limit_{key}": _text(project.soft_limit[key][index]) for key in GAINS},
	}


def _edited_project(project, index, fields, initial):
	import pyjoltphysics as pj

	change = {key for key, value in fields.items() if value != initial[key]}
	if not change: return project
	neutral, spring, soft = project.neutral.copy(), {key: row.copy() for key, row in project.spring.items()}, {key: row.copy() for key, row in project.soft_limit.items()}
	bone = project.settings.bones[index]
	values = {}
	if change & {"shape", "dimensions"}:
		shape = fields["shape"]
		label, count = SHAPE_FIELDS[shape]
		params = _numbers(fields["dimensions"], count, label)
		values["shape"] = pj.Shape(pj.ShapeKind[shape.upper()], params + (0,) * (4 - count))
	for key, label, count in (("mass", "Mass in kg", 1), ("shape_offset", "Offset XYZ", 3), ("friction", "Joint friction", 1)):
		if key in change:
			numbers = _numbers(fields[key], count, label)
			values[key] = numbers[0] if count == 1 else numbers
	if "shape_rotation" in change:
		values["shape_rotation"] = tuple(Quaternion.from_eulers(np.deg2rad(_numbers(fields["shape_rotation"], 3, "Collider rotation XYZ"))))
	if bone.parent >= 0:
		for key, label, count in (("twist", "Twist limits", 2), ("swing", "Swing half angles", 2), ("bend", "Bend", 1)):
			if key in change:
				radians = np.deg2rad(_numbers(fields[key], count, label))
				values[key] = float(radians[0]) if count == 1 else tuple(radians)
		for key, label in (("twist_axis", "Twist axis XYZ"), ("plane_axis", "Plane axis XYZ")):
			if key in change: values[key] = _numbers(fields[key], 3, label) if fields[key].strip() else None
		if "neutral" in change:
			neutral[index] = Quaternion.from_eulers(np.deg2rad(_numbers(fields["neutral"], 3, "Neutral rotation XYZ")))
		if "margin" in change:
			soft["margin"][index] = np.deg2rad(_numbers(fields["margin"], 1, "Soft-limit margin")[0])
		for gains, prefix, label in ((spring, "spring", "Neutral spring"), (soft, "limit", "Soft limit")):
			for key in GAINS:
				field = f"{prefix}_{key}"
				if field in change: gains[key][index] = _numbers(fields[field], 1, f"{label} {key.replace('_', ' ')}")[0]
	bones = project.settings.bones.copy()
	bones[index] = replace(bone, **values)
	return replace(project, settings=replace(project.settings, bones=bones), neutral=neutral, spring=spring, soft_limit=soft)


class RagdollTuner:
	def __init__(self, root, project, path=None, *, native_import=False):
		import tkinter as tk
		from tkinter import filedialog, messagebox, ttk

		self.root, self.ttk, self.filedialog, self.messagebox = root, ttk, filedialog, messagebox
		self.path, self.saved, self.native_import = path, project, native_import
		self.playing, self.timer, self.bone, self.drag = False, None, 0, None
		self.preview = None
		self.scenario = tk.StringVar(value="Drop")
		self.passive = tk.BooleanVar(value=True)
		self.impulse = tk.StringVar(value="3 0 0")
		self.status = tk.StringVar(value="Select a bone to edit, then apply your changes.")
		self.time = tk.StringVar()
		self.fields = {key: tk.StringVar() for key in _bone_fields(project, 0)}
		self.shape_label = tk.StringVar()
		root.geometry("1250x740")
		root.minsize(1000, 645)
		root.protocol("WM_DELETE_WINDOW", self.close)
		root.columnconfigure(0, weight=1)
		root.rowconfigure(1, weight=1)

		toolbar = ttk.Frame(root, padding=(12, 10))
		toolbar.grid(row=0, column=0, sticky="ew")
		ttk.Button(toolbar, text="Open rig…", command=self.open).pack(side="left")
		ttk.Button(toolbar, text="Save", command=self.save).pack(side="left", padx=(6, 0))
		ttk.Button(toolbar, text="Save as…", command=lambda: self.save(as_new=True)).pack(side="left", padx=6)
		ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
		scenario = ttk.Combobox(toolbar, textvariable=self.scenario, values=tuple(SCENARIOS), state="readonly", width=13)
		scenario.pack(side="left")
		scenario.bind("<<ComboboxSelected>>", self.reset)
		self.play_button = ttk.Button(toolbar, text="Play", command=self.toggle_play)
		self.play_button.pack(side="left", padx=(6, 0))
		ttk.Button(toolbar, text="Step", command=self.step).pack(side="left", padx=6)
		ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left")
		ttk.Checkbutton(toolbar, text="Passive torques", variable=self.passive).pack(side="left", padx=12)
		ttk.Label(toolbar, textvariable=self.time).pack(side="right")

		content = ttk.Frame(root, padding=(12, 0, 12, 0))
		content.grid(row=1, column=0, sticky="nsew")
		content.columnconfigure(1, weight=1)
		content.rowconfigure(0, weight=1)
		bones = ttk.Frame(content)
		bones.grid(row=0, column=0, sticky="ns", padx=(0, 10))
		bones.rowconfigure(0, weight=1)
		self.tree = ttk.Treeview(bones, columns=("mass",), selectmode="browse")
		self.tree.heading("#0", text="Bone", anchor="w")
		self.tree.heading("mass", text="kg", anchor="e")
		self.tree.column("#0", width=144, minwidth=90)
		self.tree.column("mass", width=46, minwidth=46, anchor="e")
		self.tree.grid(row=0, column=0, sticky="ns")
		scroll = ttk.Scrollbar(bones, orient="vertical", command=self.tree.yview)
		scroll.grid(row=0, column=1, sticky="ns")
		self.tree.configure(yscrollcommand=scroll.set)
		self.tree.bind("<<TreeviewSelect>>", self.select_tree)

		center = ttk.Frame(content)
		center.grid(row=0, column=1, sticky="nsew")
		self.canvas = tk.Canvas(center, background="#131e25", highlightthickness=0)
		self.canvas.pack(fill="both", expand=True)
		self.preview_image = tk.PhotoImage(master=self.canvas)
		self.canvas.create_image(0, 0, anchor="nw", image=self.preview_image)
		self.canvas.bind("<Configure>", lambda _: self.draw())
		for button in (1, 3):
			self.canvas.bind(f"<ButtonPress-{button}>", self.orbit_start)
			self.canvas.bind(f"<B{button}-Motion>", self.orbit)
			self.canvas.bind(f"<ButtonRelease-{button}>", self.orbit_end)
		self.canvas.bind("<MouseWheel>", lambda event: self.zoom(1 if event.delta > 0 else -1))
		self.canvas.bind("<Button-4>", lambda _: self.zoom(1))
		self.canvas.bind("<Button-5>", lambda _: self.zoom(-1))
		view_controls = ttk.Frame(center, padding=(0, 6, 0, 0))
		view_controls.pack(fill="x")
		ttk.Button(view_controls, text="Frame rig", command=self.frame).pack(side="left")
		ttk.Label(view_controls, text="Drag to orbit · Scroll to zoom").pack(side="right")

		properties = ttk.Frame(content, padding=(12, 0, 0, 0), width=330)
		properties.grid(row=0, column=2, sticky="ns")
		self.title = ttk.Label(properties, font=("TkDefaultFont", 12, "bold"))
		self.title.pack(fill="x", pady=(0, 8))
		self.tabs = ttk.Notebook(properties)
		self.tabs.pack(fill="x")
		collider, joint, passive = (ttk.Frame(self.tabs, padding=9) for _ in range(3))
		for panel, name in ((collider, "Collider"), (joint, "Joint"), (passive, "Passive")):
			self.tabs.add(panel, text=name)
			panel.columnconfigure(0, weight=1)
		ttk.Label(collider, text="Shape").grid(row=0, column=0, sticky="w")
		shape = ttk.Combobox(collider, textvariable=self.fields["shape"], values=tuple(SHAPE_FIELDS), state="readonly")
		shape.grid(row=1, column=0, sticky="ew")
		shape.bind("<<ComboboxSelected>>", self.change_shape)
		self.entries(collider, (("dimensions", self.shape_label), ("mass", "Mass (kg)"), ("shape_offset", "Bone-local offset XYZ (m)"), ("shape_rotation", "Bone-local rotation XYZ (°)")), start=2)
		self.entries(joint, (("twist", "Hard twist min, max (°)"), ("swing", "Hard swing half angles (°)"), ("bend", "Swing centre bend (°)"), ("twist_axis", "Twist axis XYZ (blank = auto)"),
			("plane_axis", "Plane axis XYZ (blank = auto)"), ("friction", "Joint friction (N·m)")))
		self.entries(passive, (("neutral", "Parent-relative neutral rotation XYZ (°)"),))
		self.gain_fields(passive, "spring", "Neutral spring", 2)
		self.entries(passive, (("margin", "Soft-limit margin inside hard limits (°)"),), start=6)
		self.gain_fields(passive, "limit", "Soft limits", 8)
		actions = ttk.Frame(properties, padding=(0, 8, 0, 0))
		actions.pack(fill="x")
		ttk.Button(actions, text="Apply to bone", command=self.apply).pack(side="left")
		ttk.Button(actions, text="Revert fields", command=self.show_fields).pack(side="left", padx=6)
		ttk.Separator(properties).pack(fill="x", pady=10)
		ttk.Label(properties, text="World impulse XYZ (N·s)").pack(anchor="w")
		push = ttk.Frame(properties)
		push.pack(fill="x", pady=(3, 0))
		ttk.Entry(push, textvariable=self.impulse).pack(side="left", fill="x", expand=True)
		ttk.Button(push, text="Push bone", command=self.push).pack(side="right", padx=(6, 0))
		ttk.Label(root, textvariable=self.status, padding=(12, 8), wraplength=960).grid(row=2, column=0, sticky="ew")
		self.install(project, redraw_tree=True)

	def entries(self, parent, fields, start=0):
		for row, (field, label) in enumerate(fields, start // 2):
			label_args = {"text": label} if isinstance(label, str) else {"textvariable": label}
			self.ttk.Label(parent, **label_args).grid(row=row * 2, column=0, sticky="w", pady=(5, 0))
			self.ttk.Entry(parent, textvariable=self.fields[field]).grid(row=row * 2 + 1, column=0, sticky="ew")

	def gain_fields(self, parent, prefix, label, row):
		self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(7, 0))
		frame = self.ttk.Frame(parent)
		frame.grid(row=row + 1, rowspan=3, column=0, sticky="ew")
		for column, (key, unit) in enumerate(zip(GAINS, ("N·m/rad", "N·m·s/rad", "N·m"))):
			frame.columnconfigure(column, weight=1, uniform="gains")
			self.ttk.Label(frame, text=key.replace("_", " ").capitalize()).grid(row=0, column=column, sticky="w")
			self.ttk.Entry(frame, width=11, textvariable=self.fields[f"{prefix}_{key}"]).grid(row=1, column=column, sticky="ew", padx=(0, 4))
			self.ttk.Label(frame, text=unit).grid(row=2, column=column, sticky="w")

	def install(self, project, *, redraw_tree=False):
		candidate = RagdollPreview(project, self.scenario.get())
		try: viewport = ColliderViewport(candidate)
		except BaseException:
			candidate.close()
			raise
		if self.preview is not None:
			if not redraw_tree and candidate.scenario == self.preview.scenario:
				viewport.yaw, viewport.pitch, viewport.zoom = self.viewport.yaw, self.viewport.pitch, self.viewport.zoom
			self.preview.close()
		self.project, self.preview, self.viewport = project, candidate, viewport
		self.active_scenario = self.scenario.get()
		self.pause()
		if redraw_tree:
			self.tree.delete(*self.tree.get_children())
			for i, bone in enumerate(project.settings.bones):
				self.tree.insert("" if bone.parent == -1 else str(bone.parent), "end", iid=str(i), text=bone.name, values=(f"{bone.mass:.4g}",), open=True)
			self.select(0, force=True)
		self.update_title()
		self.draw()

	def update_title(self):
		title = self.path.name if self.path is not None else "Humanoid template"
		dirty = " *" if self.project is not self.saved else ""
		self.root.title(f"{title}{dirty} — Ragdoll tuner")

	def draw(self):
		if self.preview is None: return
		self.time.set(f"{self.preview.world.tick / TICK_HZ:5.2f} s")
		width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
		if min(width, height) < 2: return
		rgb = self.viewport.render(self.preview, self.bone, width, height)
		self.preview_image.configure(data=f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes(), format="ppm")

	def show_fields(self):
		self.initial = _bone_fields(self.project, self.bone)
		for key, value in self.initial.items(): self.fields[key].set(value)
		self.active_shape = self.initial["shape"]
		self.shape_label.set(SHAPE_FIELDS[self.active_shape][0])
		bone = self.project.settings.bones[self.bone]
		self.title.configure(text=bone.name)
		if bone.parent == -1: self.tabs.select(0)
		for tab in (1, 2): self.tabs.tab(tab, state="disabled" if bone.parent == -1 else "normal")

	def field_values(self): return {key: var.get() for key, var in self.fields.items()}

	def select(self, index, *, force=False):
		if not force and index == self.bone: return
		if not force and self.field_values() != self.initial:
			self.status.set("Apply or revert the current fields before choosing another bone.")
			self.tree.selection_set(str(self.bone))
			self.tree.focus(str(self.bone))
			return
		self.bone = index
		self.tree.selection_set(str(index))
		self.tree.focus(str(index))
		self.tree.see(str(index))
		self.show_fields()
		self.status.set("Root body · Joint settings belong to child bones." if index == 0 else "Edit the selected bone, then apply changes to restart the preview.")
		self.draw()

	def select_tree(self, _):
		selection = self.tree.selection()
		if selection: self.select(int(selection[0]))

	def pick(self, event):
		bone = self.viewport.bone_at(event.x, event.y)
		if bone is not None: self.select(bone)

	def change_shape(self, _=None):
		old, new = self.active_shape, self.fields["shape"].get()
		if old == new: return
		bone = self.project.settings.bones[self.bone]
		try: params = _numbers(self.fields["dimensions"].get(), SHAPE_FIELDS[old][1], "Previous shape dimensions")
		except ValueError: old, params = bone.shape.kind.name.capitalize(), bone.shape.params
		if old == "Box": width, height, depth = params[:3]
		else:
			width = depth = params[0] * 2
			height = width if old == "Sphere" else width + params[1] if old == "Capsule" else params[1]
		width, height, depth = (max(0.001, size) for size in (width, height, depth))
		radius = min(width, depth) / 2
		dims = {"Box": (width, height, depth, 0), "Sphere": (min(radius, height / 2),),
			"Capsule": (radius, max(radius * 0.1, height - 2 * radius)), "Cylinder": (radius, height)}
		self.fields["dimensions"].set(_text(dims[new]))
		self.active_shape = new
		self.shape_label.set(SHAPE_FIELDS[new][0])

	def apply(self):
		try:
			project = _edited_project(self.project, self.bone, self.field_values(), self.initial)
			if project is self.project: return
			self.install(project)
		except (ValueError, RuntimeError, OverflowError) as error:
			self.status.set(str(error))
			return
		bone = self.project.settings.bones[self.bone]
		self.tree.set(str(self.bone), "mass", f"{bone.mass:.4g}")
		self.show_fields()
		self.status.set(f"Applied {bone.name}. Preview reset to tick zero.")

	def reset(self, _=None):
		try:
			self.install(self.project)
		except (ValueError, RuntimeError, OverflowError) as error:
			self.scenario.set(self.active_scenario)
			self.status.set(str(error))

	def frame(self):
		self.viewport.frame(self.preview)
		self.draw()

	def orbit_start(self, event):
		self.drag = event.num, event.x, event.y
		self.orbiting = False

	def orbit(self, event):
		if self.drag is None: return
		button, x, y = self.drag
		dx, dy = event.x - x, event.y - y
		if not self.orbiting and dx*dx + dy*dy < 16: return
		self.orbiting = True
		self.drag = button, event.x, event.y
		self.viewport.yaw -= dx * 0.01
		self.viewport.pitch = float(np.clip(self.viewport.pitch + dy * 0.01, -1.45, 1.45))
		self.draw()

	def orbit_end(self, event):
		if self.drag is None or event.num != self.drag[0]: return
		self.drag = None
		if event.num == 1 and not self.orbiting: self.pick(event)

	def zoom(self, direction):
		self.viewport.zoom = float(np.clip(self.viewport.zoom * 0.9**direction, 0.45, 4))
		self.draw()

	def pause(self):
		if self.timer is not None:
			self.root.after_cancel(self.timer)
			self.timer = None
		self.playing = False
		self.play_button.configure(text="Play")

	def toggle_play(self):
		if self.playing:
			self.pause()
			return
		self.playing = True
		self.play_button.configure(text="Pause")
		self.last_tick = monotonic()
		self.timer = self.root.after(20, self.advance)

	def advance(self):
		self.timer = None
		now = monotonic()
		self.last_tick = max(self.last_tick, now - 0.1)
		ticks = int((now - self.last_tick) * TICK_HZ)
		self.last_tick += ticks / TICK_HZ
		if ticks: self.simulate(ticks)
		if self.playing: self.timer = self.root.after(20, self.advance)

	def simulate(self, ticks):
		try:
			self.preview.step(ticks, passive=self.passive.get())
			self.draw()
			return True
		except (ValueError, RuntimeError) as error:
			self.pause()
			self.status.set(str(error))
			return False

	def step(self):
		self.pause()
		self.simulate(1)

	def push(self):
		try:
			self.preview.push(self.bone, _numbers(self.impulse.get(), 3, "World impulse XYZ"))
		except (ValueError, RuntimeError, OverflowError) as error:
			self.status.set(str(error))
			return
		if not self.playing and not self.simulate(1): return
		self.status.set(f"Pushed {self.project.settings.bones[self.bone].name}.")

	def may_discard(self):
		return (self.project is self.saved and self.field_values() == self.initial) or self.messagebox.askyesno(
			"Discard changes?", "Discard unsaved ragdoll edits?", parent=self.root)

	def open(self):
		path = self.filedialog.askopenfilename(parent=self.root, filetypes=PROJECT_FILES)
		if not path or not self.may_discard(): return
		try:
			project, native_import = open_project(path)
			self.install(project, redraw_tree=True)
		except (OSError, ValueError, RuntimeError) as error:
			self.status.set(f"Could not open rig. {error}")
			return
		self.saved, self.path, self.native_import = self.project, Path(path), native_import
		self.update_title()
		self.status.set(f"Opened {self.path.name}.")

	def save(self, *, as_new=False):
		if self.field_values() != self.initial:
			self.status.set("Apply or revert the current fields before saving.")
			return
		path = self.path
		if as_new or path is None or self.native_import:
			name = "ragdoll.json" if path is None else f"{path.stem}.tuning.json" if self.native_import else path.name
			path = self.filedialog.asksaveasfilename(parent=self.root, initialfile=name, defaultextension=".json", filetypes=PROJECT_FILES)
		if not path: return
		path = Path(path)
		if self.native_import and path.resolve() == self.path.resolve():
			self.status.set("Choose a separate .json file for this project to preserve the imported native settings.")
			return
		try:
			write_project(path, self.project)
		except (OSError, ValueError) as error:
			self.status.set(f"Could not save rig. {error}")
			return
		self.saved, self.path, self.native_import = self.project, path, False
		self.update_title()
		self.status.set(f"Saved {path.name}.")

	def close(self):
		if not self.may_discard(): return
		self.pause()
		self.dispose()
		self.root.destroy()

	def dispose(self):
		if self.preview is not None:
			self.preview.close()
			self.preview = None


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("project", type=Path, nargs="?", help="tuner project or native RagdollSettings JSON")
	args = parser.parse_args(argv)
	try:
		import tkinter as tk
	except ImportError as error:
		parser.error(f"{error}. Use a Python installation with Tk support.")
	try:
		project, native_import = open_project(args.project) if args.project is not None else (default_project(), False)
		root = tk.Tk()
		tuner = None
		try:
			tuner = RagdollTuner(root, project, args.project, native_import=native_import)
			root.mainloop()
		finally:
			if tuner is not None: tuner.dispose()
			try: root.destroy()
			except tk.TclError: pass
	except ModuleNotFoundError as error:
		if error.name != "pyjoltphysics": raise
		parser.error(f'{error}. Install the physics extra with python -m pip install -e ".[physics]".')
	except (OSError, ValueError, RuntimeError, tk.TclError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()

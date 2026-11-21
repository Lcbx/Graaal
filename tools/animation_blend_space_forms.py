"""Structured editing of named blend spaces in an in-memory graph draft."""
from copy import deepcopy
import json
import math


class BlendSpaceWindow:
	def __init__(self, host, data, original_text):
		self.host, self.data, self.original_text = host, data, original_text
		self.selected_name, self.points = None, []
		tk, ttk = host.tk, host.ttk
		self.window = tk.Toplevel(host.window)
		self.window.title("Edit 1D blend spaces")
		self.window.geometry("720x600")
		self.window.minsize(580, 480)
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Stage blend spaces here, then write them to the draft.")
		body = ttk.Frame(self.window, padding=10)
		body.pack(fill="both", expand=True)
		ttk.Label(body, text="Named 1D blend spaces").pack(anchor="w")
		self.spaces = ttk.Treeview(body, columns=("name", "parameter", "points"), show="headings", height=5, selectmode="browse")
		for column in ("name", "parameter", "points"):
			self.spaces.heading(column, text=column.capitalize())
			self.spaces.column(column, width=220 if column != "points" else 80)
		self.spaces.pack(fill="both", expand=True)
		self.spaces.bind("<<TreeviewSelect>>", self.select_space)
		ttk.Button(body, text="New blend space", command=self.new_space).pack(anchor="w", pady=5)
		row = ttk.Frame(body)
		row.pack(fill="x")
		self.name, self.parameter = tk.StringVar(), tk.StringVar()
		ttk.Label(row, text="Name").pack(side="left")
		ttk.Entry(row, textvariable=self.name, width=25).pack(side="left", padx=5)
		ttk.Label(row, text="Parameter").pack(side="left")
		ttk.Combobox(row, textvariable=self.parameter, values=tuple(data.get("parameters", {})), state="readonly", width=22).pack(side="left", padx=5)
		ttk.Label(body, text="Points in increasing position order").pack(anchor="w", pady=(10, 0))
		self.point_list = ttk.Treeview(body, columns=("position", "clip"), show="headings", height=7, selectmode="browse")
		for column in ("position", "clip"):
			self.point_list.heading(column, text=column.capitalize())
			self.point_list.column(column, width=150 if column == "position" else 390)
		self.point_list.pack(fill="both", expand=True)
		self.point_list.bind("<<TreeviewSelect>>", self.select_point)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		self.position, self.clip = tk.StringVar(value="0"), tk.StringVar()
		ttk.Label(row, text="Position").pack(side="left")
		ttk.Entry(row, textvariable=self.position, width=14).pack(side="left", padx=5)
		ttk.Label(row, text="Clip").pack(side="left")
		ttk.Combobox(row, textvariable=self.clip, values=tuple(host.graph.clip_ids), state="readonly", width=27).pack(side="left", padx=5)
		row = ttk.Frame(body)
		row.pack(fill="x")
		for label, command in (("Add point", lambda: self.store_point(False)), ("Update point", lambda: self.store_point(True)), ("Remove point", self.remove_point)):
			ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=7)
		for label, command in (("Add blend space", lambda: self.store_space(False)), ("Update selected", lambda: self.store_space(True)), ("Remove selected", self.remove_space)):
			ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
		ttk.Button(body, text="Write 1D blends to draft", command=self.apply).pack(fill="x", pady=5)
		ttk.Label(body, textvariable=self.status, wraplength=680).pack(anchor="w")
		self.refresh_spaces()
		self.new_space()

	def refresh_spaces(self, name=None):
		self.spaces.delete(*self.spaces.get_children())
		for key, space in self.data.get("blend_spaces", {}).items():
			self.spaces.insert("", "end", iid=key, values=(key, space["parameter"], len(space["points"])))
		if name is not None: self.spaces.selection_set(name)

	def refresh_points(self, index=None):
		self.point_list.delete(*self.point_list.get_children())
		for i, point in enumerate(self.points):
			self.point_list.insert("", "end", iid=str(i), values=(f"{float(point['position']):.17g}", point["clip"]))
		if index is not None: self.point_list.selection_set(str(index))

	def new_space(self):
		self.spaces.selection_remove(*self.spaces.selection())
		self.selected_name = None
		self.name.set("")
		self.parameter.set(next(iter(self.data.get("parameters", {})), ""))
		self.points = []
		self.refresh_points()
		self.position.set("0")
		self.clip.set(next(iter(self.host.graph.clip_ids), ""))

	def select_space(self, event=None):
		selection = self.spaces.selection()
		if not selection: return
		self.selected_name = selection[0]
		space = self.data["blend_spaces"][self.selected_name]
		self.name.set(self.selected_name)
		self.parameter.set(space["parameter"])
		self.points = deepcopy(space["points"])
		self.refresh_points()

	def select_point(self, event=None):
		selection = self.point_list.selection()
		if not selection: return
		point = self.points[int(selection[0])]
		self.position.set(f"{float(point['position']):.17g}")
		self.clip.set(point["clip"])

	def store_point(self, update):
		selection = self.point_list.selection()
		if update and not selection:
			self.status.set("Select a point to update.")
			return
		try:
			position = float(self.position.get())
			if not math.isfinite(position): raise ValueError("position must be finite")
			if self.clip.get() not in self.host.graph.clip_ids: raise ValueError("choose an imported clip")
		except ValueError as exc:
			self.status.set(f"Cannot stage point: {exc}")
			return
		point = {"position": position, "clip": self.clip.get()}
		if update: self.points[int(selection[0])] = point
		else: self.points.append(point)
		self.points.sort(key=lambda item: item["position"])
		self.refresh_points(self.points.index(point))
		self.status.set("Point staged. Add or update the blend space to retain it.")

	def remove_point(self):
		selection = self.point_list.selection()
		if not selection: return
		del self.points[int(selection[0])]
		self.refresh_points()

	def store_space(self, update):
		if update and self.selected_name is None:
			self.status.set("Select a blend space to update.")
			return
		name = self.name.get()
		spaces = self.data.get("blend_spaces", {})
		if not name or (name in spaces and (not update or name != self.selected_name)):
			self.status.set("Choose a unique nonempty blend-space name.")
			return
		candidate = deepcopy(self.data)
		candidate.setdefault("blend_spaces", {})
		if update: del candidate["blend_spaces"][self.selected_name]
		candidate["blend_spaces"][name] = {"parameter": self.parameter.get(), "points": deepcopy(self.points)}
		try: self.host.compile_definition(candidate)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot stage blend space: {exc}")
			return
		self.data, self.selected_name = candidate, name
		self.refresh_spaces(name)
		self.status.set("Blend space staged. Write 1D blends to draft when ready.")

	def remove_space(self):
		selection = self.spaces.selection()
		if not selection: return
		del self.data["blend_spaces"][selection[0]]
		self.refresh_spaces()
		self.new_space()

	def apply(self):
		if self.host.editor.get("1.0", "end-1c") != self.original_text:
			self.status.set("The JSON draft changed. Reopen this editor to load the latest draft before writing.")
			return
		try: self.host.compile_definition(self.data)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot write 1D blends: {exc}")
			return
		self.original_text = json.dumps(self.data, indent=2)
		self.host.editor.edit_separator()
		self.host.editor.configure(autoseparators=False)
		self.host.editor.delete("1.0", "end")
		self.host.editor.insert("1.0", self.original_text)
		self.host.editor.configure(autoseparators=True)
		self.host.editor.edit_separator()
		self.status.set("1D blends written to draft. Compile draft to apply the preview.")
		self.host.status.set(self.status.get())

	def close(self):
		self.window.destroy()
		self.host.blend_space_window = None


def _axis_positions(text, axis):
	try: positions = [float(value.strip()) for value in text.split(",")]
	except ValueError as exc: raise ValueError(f"{axis}: enter comma-separated numeric positions") from exc
	if not positions or not all(math.isfinite(value) for value in positions): raise ValueError(f"{axis}: positions must be finite")
	if any(right <= left for left, right in zip(positions, positions[1:])): raise ValueError(f"{axis}: positions must increase strictly")
	return positions


class BlendGridWindow:
	"""Stage rectangular 2D grids and write only the grid field to a draft."""
	def __init__(self, host, data, original_text):
		self.host, self.data, self.original_text = host, data, original_text
		self.selected_name = None
		tk, ttk = host.tk, host.ttk
		self.window = tk.Toplevel(host.window)
		self.window.title("Edit 2D blend grids")
		self.window.geometry("850x710")
		self.window.minsize(700, 600)
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Stage grids here, then write them to the draft.")
		body = ttk.Frame(self.window, padding=10)
		body.pack(fill="both", expand=True)
		ttk.Label(body, text="Named 2D blend grids").pack(anchor="w")
		self.grids = ttk.Treeview(body, columns=("name", "x", "y", "size"), show="headings", height=4, selectmode="browse")
		for column in ("name", "x", "y", "size"):
			self.grids.heading(column, text=column.capitalize())
			self.grids.column(column, width=220 if column == "name" else 170)
		self.grids.pack(fill="both", expand=True)
		self.grids.bind("<<TreeviewSelect>>", self.select_grid)
		ttk.Button(body, text="New grid", command=self.new_grid).pack(anchor="w", pady=5)
		self.name = tk.StringVar()
		self.x_parameter, self.y_parameter = tk.StringVar(), tk.StringVar()
		self.x_positions, self.y_positions = tk.StringVar(value="0"), tk.StringVar(value="0")
		self.lower_slot, self.upper_slot, self.weight_slot = tk.StringVar(), tk.StringVar(), tk.StringVar()
		row = ttk.Frame(body)
		row.pack(fill="x")
		ttk.Label(row, text="Name").pack(side="left")
		ttk.Entry(row, textvariable=self.name, width=28).pack(side="left", padx=5)
		parameters = tuple(data.get("parameters", {}))
		for label, variable in (("X parameter", self.x_parameter), ("Y parameter", self.y_parameter)):
			ttk.Label(row, text=label).pack(side="left")
			ttk.Combobox(row, textvariable=variable, values=parameters, state="readonly", width=15).pack(side="left", padx=5)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		for label, variable in (("X positions", self.x_positions), ("Y positions", self.y_positions)):
			ttk.Label(row, text=label).pack(side="left")
			ttk.Entry(row, textvariable=variable, width=30).pack(side="left", padx=5)
		ttk.Button(row, text="Apply axes", command=self.set_axes).pack(side="left", padx=5)
		row = ttk.Frame(body)
		row.pack(fill="x")
		for label, variable, values in (("Lower slot", self.lower_slot, data["sample_slots"]), ("Upper slot", self.upper_slot, data["sample_slots"]), ("Weight slot", self.weight_slot, data.get("weight_slots", []))):
			ttk.Label(row, text=label).pack(side="left")
			ttk.Combobox(row, textvariable=variable, values=tuple(values), state="readonly", width=15).pack(side="left", padx=5)
		ttk.Label(body, text="Cells: Y rows, X columns. Select a cell to assign an imported clip.").pack(anchor="w", pady=(10, 0))
		self.cells = ttk.Treeview(body, columns=("y", "x", "clip"), show="headings", height=8, selectmode="browse")
		for column in ("y", "x", "clip"):
			self.cells.heading(column, text=column.capitalize())
			self.cells.column(column, width=140 if column != "clip" else 420)
		self.cells.pack(fill="both", expand=True)
		self.cells.bind("<<TreeviewSelect>>", self.select_cell)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		self.clip = tk.StringVar()
		ttk.Label(row, text="Clip").pack(side="left")
		ttk.Combobox(row, textvariable=self.clip, values=tuple(host.graph.clip_ids), state="readonly", width=38).pack(side="left", padx=5)
		ttk.Button(row, text="Set selected cell", command=self.store_cell).pack(side="left")
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		for label, command in (("Add grid", lambda: self.store_grid(False)), ("Update selected", lambda: self.store_grid(True)), ("Remove selected", self.remove_grid)):
			ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
		ttk.Button(body, text="Write 2D grids to draft", command=self.apply).pack(fill="x", pady=5)
		ttk.Label(body, textvariable=self.status, wraplength=800).pack(anchor="w")
		self.refresh_grids()
		self.new_grid()

	def refresh_grids(self, name=None):
		self.grids.delete(*self.grids.get_children())
		for key, grid in self.data.get("blend_spaces_2d", {}).items():
			self.grids.insert("", "end", iid=key, values=(key, grid["x_parameter"], grid["y_parameter"], f"{len(grid['x_positions'])} × {len(grid['y_positions'])}"))
		if name is not None: self.grids.selection_set(name)

	def refresh_cells(self, selected=None):
		self.cells.delete(*self.cells.get_children())
		for y, row in enumerate(self.clip_grid):
			for x, clip in enumerate(row):
				self.cells.insert("", "end", iid=f"{y},{x}", values=(f"{self.axis_y[y]:.17g}", f"{self.axis_x[x]:.17g}", clip))
		if selected is not None: self.cells.selection_set(selected)

	def new_grid(self):
		self.grids.selection_remove(*self.grids.selection())
		self.selected_name = None
		self.name.set("")
		parameters = tuple(self.data.get("parameters", {}))
		self.x_parameter.set(parameters[0] if parameters else "")
		self.y_parameter.set(parameters[1] if len(parameters) > 1 else "")
		self.axis_x, self.axis_y = [0.0], [0.0]
		self.x_positions.set("0")
		self.y_positions.set("0")
		self.applied_axes = ("0", "0")
		slots = self.data["sample_slots"]
		self.lower_slot.set(slots[0] if slots else "")
		self.upper_slot.set(slots[1] if len(slots) > 1 else "")
		self.weight_slot.set(next(iter(self.data.get("weight_slots", [])), ""))
		self.clip.set(next(iter(self.host.graph.clip_ids), ""))
		self.clip_grid = [[self.clip.get()]]
		self.refresh_cells()

	def select_grid(self, event=None):
		selection = self.grids.selection()
		if not selection: return
		self.selected_name = selection[0]
		grid = self.data["blend_spaces_2d"][self.selected_name]
		self.name.set(self.selected_name)
		for key in ("x_parameter", "y_parameter", "weight_slot"):
			getattr(self, key).set(grid[key])
		self.lower_slot.set(grid["sample_slots"][0])
		self.upper_slot.set(grid["sample_slots"][1])
		self.axis_x, self.axis_y = list(grid["x_positions"]), list(grid["y_positions"])
		self.x_positions.set(", ".join(f"{value:.17g}" for value in self.axis_x))
		self.y_positions.set(", ".join(f"{value:.17g}" for value in self.axis_y))
		self.applied_axes = (self.x_positions.get(), self.y_positions.get())
		self.clip_grid = deepcopy(grid["clips"])
		self.refresh_cells()

	def set_axes(self):
		try:
			x_axis = _axis_positions(self.x_positions.get(), "X axis")
			y_axis = _axis_positions(self.y_positions.get(), "Y axis")
		except ValueError as exc:
			self.status.set(f"Cannot apply axes: {exc}")
			return
		old = {(y, x): clip for y, row in zip(self.axis_y, self.clip_grid) for x, clip in zip(self.axis_x, row)}
		default = next(iter(self.host.graph.clip_ids), "")
		self.axis_x, self.axis_y = x_axis, y_axis
		self.clip_grid = [[old.get((y, x), default) for x in x_axis] for y in y_axis]
		self.x_positions.set(", ".join(f"{value:.17g}" for value in x_axis))
		self.y_positions.set(", ".join(f"{value:.17g}" for value in y_axis))
		self.applied_axes = (self.x_positions.get(), self.y_positions.get())
		self.refresh_cells()
		self.status.set("Axes staged. Assign clips to new cells, then add or update the grid.")

	def select_cell(self, event=None):
		selection = self.cells.selection()
		if not selection: return
		y, x = (int(value) for value in selection[0].split(","))
		self.clip.set(self.clip_grid[y][x])

	def store_cell(self):
		selection = self.cells.selection()
		if not selection:
			self.status.set("Select a cell to assign a clip.")
			return
		if self.clip.get() not in self.host.graph.clip_ids:
			self.status.set("Choose an imported clip.")
			return
		y, x = (int(value) for value in selection[0].split(","))
		self.clip_grid[y][x] = self.clip.get()
		self.refresh_cells(selection[0])
		self.status.set("Cell staged. Add or update the grid to retain it.")

	def store_grid(self, update):
		if update and self.selected_name is None:
			self.status.set("Select a grid to update.")
			return
		name = self.name.get()
		grids = self.data.get("blend_spaces_2d", {})
		if not name or (name in grids and (not update or name != self.selected_name)):
			self.status.set("Choose a unique nonempty grid name.")
			return
		if (self.x_positions.get(), self.y_positions.get()) != self.applied_axes:
			self.status.set("Apply X/Y axes before storing the grid.")
			return
		candidate = deepcopy(self.data)
		candidate.setdefault("blend_spaces_2d", {})
		if update: del candidate["blend_spaces_2d"][self.selected_name]
		candidate["blend_spaces_2d"][name] = {
			"x_parameter": self.x_parameter.get(), "y_parameter": self.y_parameter.get(),
			"x_positions": self.axis_x.copy(), "y_positions": self.axis_y.copy(), "clips": deepcopy(self.clip_grid),
			"sample_slots": [self.lower_slot.get(), self.upper_slot.get()], "weight_slot": self.weight_slot.get(),
		}
		try: self.host.compile_definition(candidate)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot stage grid: {exc}")
			return
		self.data, self.selected_name = candidate, name
		self.refresh_grids(name)
		self.status.set("Grid staged. Write 2D grids to draft when ready.")

	def remove_grid(self):
		selection = self.grids.selection()
		if not selection: return
		del self.data["blend_spaces_2d"][selection[0]]
		self.refresh_grids()
		self.new_grid()

	def apply(self):
		if self.host.editor.get("1.0", "end-1c") != self.original_text:
			self.status.set("The JSON draft changed. Reopen this editor to load the latest draft before writing.")
			return
		try: self.host.compile_definition(self.data)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot write 2D grids: {exc}")
			return
		self.original_text = json.dumps(self.data, indent=2)
		self.host.editor.edit_separator()
		self.host.editor.configure(autoseparators=False)
		self.host.editor.delete("1.0", "end")
		self.host.editor.insert("1.0", self.original_text)
		self.host.editor.configure(autoseparators=True)
		self.host.editor.edit_separator()
		self.status.set("2D grids written to draft. Compile draft to apply the preview.")
		self.host.status.set(self.status.get())

	def close(self):
		self.window.destroy()
		self.host.blend_grid_window = None

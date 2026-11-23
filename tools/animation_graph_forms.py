"""Structured editing for an in-memory graph draft."""
from copy import deepcopy
import math


class StateParameterWindow:
	def __init__(self, host, data, original_text):
		self.host, self.data, self.original_text = host, data, original_text
		tk, ttk = host.tk, host.ttk
		self.window = tk.Toplevel(host.window)
		self.window.title("Edit graph states and parameters")
		self.window.geometry("740x510")
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Stage edits here, then write them to the draft.")
		self.sources = {"Clip": ("clip", tuple(host.clip_box["values"])),
			"1D blend": ("blend_space", tuple(data.get("blend_spaces", {}))),
			"2D blend": ("blend_space_2d", tuple(data.get("blend_spaces_2d", {})))}
		self.source_type, self.source = tk.StringVar(value="Clip"), tk.StringVar()
		self.names = {kind: tk.StringVar() for kind in ("states", "parameters")}
		self.default = tk.StringVar(value="0")
		body = ttk.Frame(self.window, padding=10)
		body.pack(fill="both", expand=True)
		tabs = ttk.Notebook(body)
		tabs.pack(fill="both", expand=True)
		self.trees = {}
		for kind, columns in (("states", ("Name", "Source", "Resource", "Initial")), ("parameters", ("Name", "Default"))):
			page = ttk.Frame(tabs, padding=8)
			tabs.add(page, text=kind.capitalize())
			tree = self.trees[kind] = ttk.Treeview(page, columns=columns, show="headings", selectmode="browse")
			for column in columns:
				tree.heading(column, text=column)
				tree.column(column, width=74 if column == "Initial" else 170)
			tree.pack(fill="both", expand=True)
			tree.bind("<<TreeviewSelect>>", lambda _, kind=kind: self.select(kind))
			row = ttk.Frame(page)
			row.pack(fill="x", pady=6)
			ttk.Label(row, text="Name").pack(side="left")
			ttk.Entry(row, textvariable=self.names[kind], width=20).pack(side="left", padx=5)
			if kind == "states":
				types = ttk.Combobox(row, textvariable=self.source_type, values=tuple(self.sources), state="readonly", width=12)
				types.pack(side="left", padx=5)
				types.bind("<<ComboboxSelected>>", lambda _: self.select_source())
				self.resource = ttk.Combobox(row, textvariable=self.source, state="readonly", width=26)
				self.resource.pack(side="left")
				self.select_source()
			else:
				ttk.Label(row, text="Default").pack(side="left")
				ttk.Entry(row, textvariable=self.default, width=16).pack(side="left", padx=5)
			buttons = ttk.Frame(page)
			buttons.pack(fill="x")
			for action, label in (("add", "Add"), ("update", "Update selected"), ("remove", "Remove selected")):
				ttk.Button(buttons, text=label, command=lambda kind=kind, action=action: self.edit(kind, action)).pack(side="left", padx=3)
			if kind == "states":
				ttk.Button(buttons, text="Set initial state", command=lambda: self.edit("states", "initial")).pack(side="left", padx=3)
			self.refresh(kind)
		ttk.Button(body, text="Write states and parameters to draft", command=self.apply).pack(fill="x", pady=7)
		ttk.Label(body, textvariable=self.status, wraplength=680).pack(anchor="w")

	def refresh(self, kind, selected=None):
		tree = self.trees[kind]
		tree.delete(*tree.get_children())
		for name, value in self.data.get(kind, {}).items():
			if kind == "states":
				label = next(label for label, (key, _) in self.sources.items() if key in value)
				values = (name, label, value[self.sources[label][0]], "Yes" if name == self.data["initial_state"] else "")
			else:
				values = (name, value)
			tree.insert("", "end", iid=name, values=values)
		if selected is not None: tree.selection_set(selected)

	def select_source(self, resource=None):
		_, choices = self.sources[self.source_type.get()]
		self.resource.configure(values=choices)
		self.source.set(resource if resource is not None else (choices[0] if choices else ""))

	def select(self, kind):
		selection = self.trees[kind].selection()
		if not selection: return
		name = selection[0]
		self.names[kind].set(name)
		value = self.data[kind][name]
		if kind == "states":
			label = next(label for label, (key, _) in self.sources.items() if key in value)
			self.source_type.set(label)
			self.select_source(value[self.sources[label][0]])
		else:
			self.default.set(f"{float(value):.17g}")

	def edit(self, kind, action):
		selection = self.trees[kind].selection()
		if action != "add" and not selection: return
		selected = selection[0] if action != "add" else None
		candidate = deepcopy(self.data)
		values = candidate.setdefault(kind, {})
		name = self.names[kind].get().strip()
		try:
			if action in ("add", "update"):
				if not name or (name != selected and name in values): raise ValueError("Choose a unique, nonempty name")
				value = {self.sources[self.source_type.get()][0]: self.source.get()} if kind == "states" else float(self.default.get())
				if selected is not None and name != selected:
					candidate[kind] = {name if key == selected else key: item for key, item in values.items()}
					_rename_graph_binding(candidate, kind, selected, name)
				candidate[kind][name] = value
				selected = name
			elif action == "initial":
				candidate["initial_state"] = selected
			else:
				del values[selected]
				selected = None
			self.host.compile_definition(candidate)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot stage edit: {exc}")
			return
		self.data = candidate
		self.refresh(kind, selected)
		self.status.set("Edit staged. Write states and parameters to draft when ready.")

	def apply(self):
		self.original_text = self.host.write_form_draft(self.data, self.original_text, "states and parameters", self.status)

	def close(self):
		self.window.destroy()
		self.host.state_parameter_window = None


def _rename_graph_binding(data, kind, old, name):
	if kind == "states":
		if data["initial_state"] == old: data["initial_state"] = name
		for rule in data.get("transitions", ()):
			for key in ("source", "target"):
				if rule[key] == old: rule[key] = name
		return
	for rule in data.get("transitions", ()):
		for condition in rule["conditions"]:
			if condition["parameter"] == old: condition["parameter"] = name
	if old in data.get("weight_slots", ()):
		data["weight_slots"] = [name if slot == old else slot for slot in data["weight_slots"]]
		for task in data["tasks"]:
			if task.get("weight") == old: task["weight"] = name
		for grid in data.get("blend_spaces_2d", {}).values():
			if grid.get("weight_slot") == old: grid["weight_slot"] = name
	for field, columns in (("blend_spaces", ("parameter",)), ("blend_spaces_2d", ("x_parameter", "y_parameter"))):
		for space in data.get(field, {}).values():
			for column in columns:
				if space[column] == old: space[column] = name


class TransitionWindow:
	def __init__(self, host, data, original_text):
		self.host, self.data, self.original_text = host, data, original_text
		tk, ttk = host.tk, host.ttk
		self.conditions = []
		self.window = tk.Toplevel(host.window)
		self.window.title("Edit graph transitions")
		self.window.geometry("840x680")
		self.window.minsize(720, 600)
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Stage rules here, then write them to the draft. Compile draft applies the preview.")
		body = ttk.Frame(self.window, padding=10)
		body.pack(fill="both", expand=True)
		ttk.Label(body, text="Transition priority: first matching rule wins. All conditions on a rule must match.").pack(anchor="w")
		self.rules = ttk.Treeview(body, columns=("source", "target", "duration"), show="headings", height=7, selectmode="browse")
		for name in ("source", "target", "duration"):
			self.rules.heading(name, text=name.capitalize())
			self.rules.column(name, width=180 if name != "duration" else 90)
		self.rules.pack(fill="both", expand=True)
		self.rules.bind("<<TreeviewSelect>>", self.select_rule)
		buttons = ttk.Frame(body)
		buttons.pack(fill="x", pady=5)
		for label, command in (("New rule", self.new_rule), ("Remove selected", self.remove_rule), ("Higher priority", lambda: self.move_rule(-1)), ("Lower priority", lambda: self.move_rule(1))):
			ttk.Button(buttons, text=label, command=command).pack(side="left", padx=2)
		row = ttk.Frame(body)
		row.pack(fill="x")
		self.source, self.target = tk.StringVar(), tk.StringVar()
		self.duration = tk.StringVar(value="0.25")
		for name, value in (("Source", self.source), ("Target", self.target)):
			ttk.Label(row, text=name).pack(side="left", padx=3)
			ttk.Combobox(row, textvariable=value, values=tuple(data["states"]), state="readonly", width=22).pack(side="left")
		ttk.Label(row, text="Fade (s)").pack(side="left", padx=3)
		ttk.Entry(row, textvariable=self.duration, width=12).pack(side="left")
		ttk.Label(body, text="Conditions for the rule being edited").pack(anchor="w", pady=(10, 0))
		self.condition_list = ttk.Treeview(body, columns=("parameter", "comparison", "threshold"), show="headings", height=5, selectmode="browse")
		for name in ("parameter", "comparison", "threshold"):
			self.condition_list.heading(name, text=name.capitalize())
			self.condition_list.column(name, width=160)
		self.condition_list.pack(fill="both", expand=True)
		self.condition_list.bind("<<TreeviewSelect>>", self.select_condition)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		self.parameter, self.comparison, self.threshold = tk.StringVar(), tk.StringVar(value=">"), tk.StringVar(value="0.5")
		ttk.Combobox(row, textvariable=self.parameter, values=tuple(data.get("parameters", {})), state="readonly", width=22).pack(side="left")
		ttk.Combobox(row, textvariable=self.comparison, values=("<", "<=", ">", ">=", "==", "!="), state="readonly", width=5).pack(side="left", padx=5)
		ttk.Entry(row, textvariable=self.threshold, width=16).pack(side="left")
		for label, command in (("Add", lambda: self.store_condition(False)), ("Update", lambda: self.store_condition(True)), ("Remove", self.remove_condition)):
			ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=5)
		ttk.Button(row, text="Add rule", command=lambda: self.store_rule(False)).pack(side="left")
		ttk.Button(row, text="Update selected rule", command=lambda: self.store_rule(True)).pack(side="left", padx=5)
		ttk.Button(body, text="Write transitions to draft", command=self.apply).pack(fill="x", pady=5)
		ttk.Label(body, textvariable=self.status, wraplength=780).pack(anchor="w")
		self.refresh_rules()
		self.new_rule()

	def selected(self, tree):
		selection = tree.selection()
		return int(selection[0]) if selection else None

	def refresh_rules(self, selected=None):
		self.rules.delete(*self.rules.get_children())
		for i, rule in enumerate(self.data.get("transitions", [])):
			self.rules.insert("", "end", iid=str(i), values=(rule["source"], rule["target"], f"{float(rule['duration']):.17g}"))
		if selected is not None: self.rules.selection_set(str(selected))

	def refresh_conditions(self, selected=None):
		self.condition_list.delete(*self.condition_list.get_children())
		for i, condition in enumerate(self.conditions):
			self.condition_list.insert("", "end", iid=str(i), values=(condition["parameter"], condition["comparison"], f"{float(condition['threshold']):.17g}"))
		if selected is not None: self.condition_list.selection_set(str(selected))

	def new_rule(self):
		self.rules.selection_remove(*self.rules.selection())
		states = tuple(self.data["states"])
		self.source.set(states[0])
		self.target.set(states[min(1, len(states)-1)])
		self.duration.set("0.25")
		self.conditions = []
		self.refresh_conditions()
		self.parameter.set(next(iter(self.data.get("parameters", {})), ""))
		self.comparison.set(">")
		self.threshold.set("0.5")

	def select_rule(self, event=None):
		index = self.selected(self.rules)
		if index is None: return
		rule = self.data["transitions"][index]
		self.source.set(rule["source"])
		self.target.set(rule["target"])
		self.duration.set(f"{float(rule['duration']):.17g}")
		self.conditions = deepcopy(rule["conditions"])
		self.refresh_conditions()

	def select_condition(self, event=None):
		index = self.selected(self.condition_list)
		if index is None: return
		condition = self.conditions[index]
		self.parameter.set(condition["parameter"])
		self.comparison.set(condition["comparison"])
		self.threshold.set(f"{float(condition['threshold']):.17g}")

	def store_condition(self, update):
		index = self.selected(self.condition_list) if update else len(self.conditions)
		if index is None:
			self.status.set("Select a condition to update.")
			return
		try:
			threshold = float(self.threshold.get())
			if not math.isfinite(threshold): raise ValueError("threshold must be finite")
			if self.parameter.get() not in self.data.get("parameters", {}): raise ValueError("choose a draft parameter")
			if self.comparison.get() not in ("<", "<=", ">", ">=", "==", "!="): raise ValueError("choose a comparison")
		except ValueError as exc:
			self.status.set(f"Cannot stage condition: {exc}")
			return
		condition = {"parameter": self.parameter.get(), "comparison": self.comparison.get(), "threshold": threshold}
		if update: self.conditions[index] = condition
		else: self.conditions.append(condition)
		self.refresh_conditions(index)
		self.status.set("Condition staged. Add or update the rule to retain it.")

	def remove_condition(self):
		index = self.selected(self.condition_list)
		if index is None: return
		del self.conditions[index]
		self.refresh_conditions()
		self.status.set("Condition removed from the form. Update the rule to retain the change.")

	def store_rule(self, update):
		rules = self.data.get("transitions", [])
		index = self.selected(self.rules) if update else len(rules)
		if index is None:
			self.status.set("Select a rule to update.")
			return
		try:
			rule = {"source": self.source.get(), "target": self.target.get(), "duration": float(self.duration.get()), "conditions": deepcopy(self.conditions)}
			candidate = deepcopy(self.data)
			candidate.setdefault("transitions", [])
			if update: candidate["transitions"][index] = rule
			else: candidate["transitions"].append(rule)
			self.host.compile_definition(candidate)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot stage rule: {exc}")
			return
		self.data = candidate
		self.refresh_rules(index)
		self.status.set("Rule staged. Write transitions to draft when ready.")

	def remove_rule(self):
		index = self.selected(self.rules)
		if index is None: return
		del self.data["transitions"][index]
		self.refresh_rules()
		self.new_rule()
		self.status.set("Rule removed. Write transitions to draft when ready.")

	def move_rule(self, direction):
		index = self.selected(self.rules)
		if index is None: return
		target = index + direction
		rules = self.data["transitions"]
		if not 0 <= target < len(rules): return
		rules[index], rules[target] = rules[target], rules[index]
		self.refresh_rules(target)
		self.status.set("Priority changed. Write transitions to draft when ready.")

	def apply(self):
		self.original_text = self.host.write_form_draft(self.data, self.original_text, "transitions", self.status)

	def close(self):
		self.window.destroy()
		self.host.transition_window = None


class SyncMarkerWindow:
	"""Edit in-memory sync tracks consumed on the next graph compilation."""
	def __init__(self, host):
		self.host = host
		tk, ttk = host.tk, host.ttk
		self.window = tk.Toplevel(host.window)
		self.window.title("Edit graph sync markers")
		self.window.geometry("520x430")
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.status = tk.StringVar(value="Markers are in memory. Compile draft to apply them to the preview.")
		body = ttk.Frame(self.window, padding=10)
		body.pack(fill="both", expand=True)
		ttk.Label(body, text="Clip").pack(anchor="w")
		self.clip = tk.StringVar()
		self.clip_box = ttk.Combobox(body, textvariable=self.clip, values=tuple(host.graph.clip_ids), state="readonly")
		self.clip_box.pack(fill="x")
		self.clip_box.current(0)
		self.clip_box.bind("<<ComboboxSelected>>", self.refresh)
		self.markers = ttk.Treeview(body, columns=("time", "name"), show="headings", selectmode="browse")
		for column in ("time", "name"):
			self.markers.heading(column, text=column.capitalize())
		self.markers.pack(fill="both", expand=True, pady=7)
		self.markers.bind("<<TreeviewSelect>>", self.select_marker)
		row = ttk.Frame(body)
		row.pack(fill="x")
		self.marker_time, self.marker_name = tk.StringVar(value="0"), tk.StringVar()
		ttk.Label(row, text="Time (s)").pack(side="left")
		ttk.Entry(row, textvariable=self.marker_time, width=13).pack(side="left", padx=5)
		ttk.Label(row, text="Name").pack(side="left")
		ttk.Entry(row, textvariable=self.marker_name, width=22).pack(side="left", padx=5)
		row = ttk.Frame(body)
		row.pack(fill="x", pady=6)
		for label, command in (("Add", self.add), ("Update selected", self.update), ("Remove selected", self.remove)):
			ttk.Button(row, text=label, command=command).pack(side="left", padx=2)
		ttk.Label(body, textvariable=self.status, wraplength=490).pack(anchor="w")
		self.refresh()

	@property
	def track(self):
		return self.host.animations[self.clip_box.current()].sync_markers

	def refresh(self, event=None, selected=None):
		self.markers.delete(*self.markers.get_children())
		if self.track is None: return
		ids, track = self.track.snapshot()
		for marker_id, marker_time, name in zip(ids, track.times, track.names):
			self.markers.insert("", "end", iid=str(marker_id), values=(f"{marker_time:.17g}", name))
		if selected is not None: self.markers.selection_set(str(selected))

	def select_marker(self, event=None):
		selection = self.markers.selection()
		if not selection: return
		self.marker_time.set(self.markers.set(selection[0], "time"))
		self.marker_name.set(self.markers.set(selection[0], "name"))

	def add(self):
		if self.track is None:
			self.status.set("This clip has no duration for sync markers.")
			return
		try: marker_id = self.track.add(float(self.marker_time.get()), self.marker_name.get())
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot add marker: {exc}")
			return
		self.refresh(selected=marker_id)
		self.status.set("Marker added. Compile draft to apply its sync mappings.")

	def update(self):
		selection = self.markers.selection()
		if not selection: return
		if self.track is None: return
		try: self.track.update(int(selection[0]), time=float(self.marker_time.get()), name=self.marker_name.get())
		except (ValueError, TypeError, KeyError) as exc:
			self.status.set(f"Cannot update marker: {exc}")
			return
		self.refresh(selected=selection[0])
		self.status.set("Marker updated. Compile draft to apply its sync mappings.")

	def remove(self):
		selection = self.markers.selection()
		if not selection: return
		if self.track is None: return
		self.track.remove(int(selection[0]))
		self.refresh()
		self.status.set("Marker removed. Compile draft to apply its sync mappings.")

	def close(self):
		self.window.destroy()
		self.host.sync_marker_window = None

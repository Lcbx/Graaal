"""Structured transition editing for an in-memory graph draft."""
from copy import deepcopy
import json
import math


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
		if self.host.editor.get("1.0", "end-1c") != self.original_text:
			self.status.set("The JSON draft changed. Reopen this editor to load the latest draft before writing.")
			return
		try: self.host.compile_definition(self.data)
		except (ValueError, TypeError) as exc:
			self.status.set(f"Cannot write transitions: {exc}")
			return
		self.original_text = json.dumps(self.data, indent=2)
		self.host.editor.edit_separator()
		self.host.editor.configure(autoseparators=False)
		self.host.editor.delete("1.0", "end")
		self.host.editor.insert("1.0", self.original_text)
		self.host.editor.configure(autoseparators=True)
		self.host.editor.edit_separator()
		self.status.set("Transitions written to draft. Compile draft to apply the preview.")
		self.host.status.set(self.status.get())

	def close(self):
		self.window.destroy()
		self.host.transition_window = None

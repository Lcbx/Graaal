"""Reusable checks for settings dataclasses."""

from dataclasses import dataclass
from typing import Annotated, get_args, get_origin, get_type_hints

import numpy as np


def float_range(lower = 0.0001, upper=None):
	"""Check finite values from lower through upper, or from lower upward.
	default start at a small non-zero value because a lot of algorithms can get close but break at actual 0.0 value

	Example: threshold: float_range(0, 1) = 0.5
	"""
	def check(value, name):
		try: valid = np.isfinite(value) and value >= lower and (upper is None or value <= upper)
		except (TypeError, ValueError): valid = False
		if not valid:
			bounds = f"at least {lower}" if upper is None else f"between {lower} and {upper}"
			raise ValueError(f"{name} must be finite and {bounds}")
	return Annotated[float, check]


def int_range(lower, upper):
	"""Check integers from lower through upper, inclusive.

	Example: count: int_range(1, 8) = 4
	"""
	def check(value, name):
		if type(value) is not int or not lower <= value <= upper:
			raise ValueError(f"{name} must be an integer between {lower} and {upper}")
	return Annotated[int, check]


def _check_bool(value, name):
	if type(value) is not bool: raise ValueError(f"{name} must be boolean")


def checked_dataclass(**options):
	"""Create a dataclass that checks bool fields and marked fields using dataclass options.

	Example: @checked_dataclass(frozen=True)
	"""
	def decorate(cls):
		checks = []
		for name, hint in get_type_hints(cls, include_extras=True).items():
			if get_origin(hint) is Annotated:
				hint, *validators = get_args(hint)
			else: validators = ()
			if hint is bool: checks.append((name, _check_bool))
			checks.extend((name, check) for check in validators)
		post_init = getattr(cls, "__post_init__", None)

		def __post_init__(self):
			for name, check in checks: check(getattr(self, name), name)
			if post_init is not None: post_init(self)

		cls.__post_init__ = __post_init__
		return dataclass(cls, **options)
	return decorate

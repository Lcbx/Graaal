"""Public ECS API. Implementation and optional backends load on first access."""
from importlib import import_module as _import_module


__all__ = (
	'BoolArray',
	'CommandBuffer',
	'CommandError',
	'ComponentAccessor',
	'ComponentColumns',
	'ComponentPerEntityCount',
	'ComponentRows',
	'ComponentSchema',
	'ComponentSelection',
	'ComponentStorage',
	'CountArray',
	'DeferredEntities',
	'ECS',
	'Entity',
	'EntityArray',
	'EntityLike',
	'FieldArray',
	'GENERATION_INCREMENT',
	'Generation',
	'GenerationArray',
	'INDEX_BITS',
	'INDEX_MASK',
	'INDEX_MAX',
	'INITIAL_CAPACITY',
	'Index',
	'IndexArray',
	'IndexLike',
	'Mask',
	'MaskArray',
	'MultiComponentStorage',
	'NONE',
	'NO_ENTITY',
	'OBJECT_DTYPE',
	'Query',
	'QueryComponent',
	'QueryResult',
	'RowLike',
	'Scheduler',
	'arg_names',
	'array_field',
	'as_entities',
	'as_rows',
	'check_world',
	'columns',
	'component',
	'entity_generation',
	'entity_index',
	'higher_pow2',
	'storage_info',
	'world_info',
)


def __getattr__(name):
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	value = getattr(_import_module('.core', __name__), name)
	globals()[name] = value
	return value

import ast
import os
import subprocess
import sys
from pathlib import Path


def test_inline_imports_are_confined_to_cold_paths():
	# Frame recording, simulation updates and array operations keep imports out.
	cold_paths = {
		"Assets": {"load_gltf_scene", "load_gltf_first_mesh", "load_blue_noise"},
		"Diagnostics": {"__getattr__", "setup_memory_monitor"},
		"EcsPhysics": {"EcsPhysicsAdapter.__init__"},
		"PhysicsRagdoll": {"RagdollAsset.__init__"},
		"Geometry": {"make_cube_mesh"},
		"ImGui": {"ImGui.__init__"},
		"MaterialRenderer": {"__getattr__"},
		"MeshArena": {"_Pool.__init__", "MeshArena.__init__"},
		"SkeletalGltf": {"load_skinned_gltf"},
		"ecs_diagnostics": {"check_world"},
		"__main__": {"render_example"},
	}
	root = Path(__file__).resolve().parents[1] / "graaal"
	inline = []
	def inspect(path, node, scope=()):
		if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
			scope = (*scope, node.name)
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ".".join(scope) not in cold_paths.get(path.stem, ()):
			inline.extend(f"{path.name}:{item.lineno}" for item in ast.walk(node) if isinstance(item, (ast.Import, ast.ImportFrom)))
		for child in ast.iter_child_nodes(node): inspect(path, child, scope)
	for path in sorted(root.rglob("*.py")):
		inspect(path, ast.parse(path.read_text(encoding="utf-8-sig")))
	assert not inline, "Inline imports outside cold paths: " + ", ".join(inline)


def test_module_imports_work_in_either_order():
	for modules in (("ECS", "ecs_diagnostics"), ("ecs_diagnostics", "ECS"), ("RenderContext", "MeshArena"), ("MeshArena", "RenderContext")):
		result = subprocess.run([sys.executable, "-c", f"import graaal.{modules[0]}; import graaal.{modules[1]}"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
		assert result.returncode == 0, result.stderr


def test_cpu_library_imports_need_no_optional_dependencies_or_path_mutation(tmp_path):
	env = os.environ.copy()
	env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
	result = subprocess.run([sys.executable, "-c", '''
import sys
import importlib.abc
import numpy
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'wgpu', 'glfw', 'jinja2', 'pyrr', 'pygltflib', 'PIL', 'imgui_bundle', 'mypy_extensions', 'pyjoltphysics', 'psutil'}:
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, BlockOptional())
before = list(sys.path)
import graaal
from graaal import ECS, Math, Assets, TransparencyData, DrawList, InstanceData
from graaal import draw_compiler, draw_layout, utils
from graaal import SkeletalAnimation, SkeletalGltf, RootMotion, EcsAnimation, AnimationEvents, AnimationGraph, Geometry
from graaal.diagnostics_visualizer import DiagnosticsVisualizer
from graaal import EcsPhysics
from graaal import PhysicsRagdoll
from graaal.MaterialTransform import MaterialTransform
world = ECS.ECS()
world.register(MaterialTransform)
entity = world.create(1)
world.add(entity, MaterialTransform())
assert world.get(MaterialTransform)[entity].rotation.tolist() == [[0, 0, 0, 1]]
assert 'graaal.EcsMaterials' not in sys.modules
assert Math.Quaternion.from_y_rotation(0.5).shape == (4,)
from types import SimpleNamespace as NS
from graaal.__main__ import simulate
from graaal.Diagnostics import WatchTimer, setup_memory_monitor
with WatchTimer("cpu"): pass
assert Assets.load_blue_noise().shape == (64, 64)
widget = DiagnosticsVisualizer(visible=False)
widget.update({'commands': []})
widget.draw()
def missing_dependency(callback, names):
    names = (names,) if isinstance(names, str) else names
    try:
        callback()
    except ModuleNotFoundError as error:
        assert error.name in names, str(error)
    else:
        raise AssertionError('Expected optional dependency error')
missing_dependency(lambda: Assets.load_blue_noise('missing.png'), 'PIL')
missing_dependency(lambda: Assets.load_gltf_first_mesh('missing.glb'), 'pygltflib')
missing_dependency(lambda: Assets.load_gltf_scene('missing.glb'), 'PIL')
missing_dependency(lambda: SkeletalGltf.load_skinned_gltf('missing.glb'), 'pygltflib')
missing_dependency(Geometry.make_cube_mesh, ('wgpu', 'glfw', 'jinja2'))
missing_dependency(lambda: EcsPhysics.EcsPhysicsAdapter(world, {}), 'pyjoltphysics')
missing_dependency(setup_memory_monitor, 'psutil')
widget.visible = True
missing_dependency(widget.draw, 'imgui_bundle')
assert simulate().shape == (2, 3)
assert sys.path == before
assert Math.Camera((0,0,3), (0,0,0), (0,1,0), 60).view().shape == (4,4)
grouping = DrawList.DrawList.build([10, 10], [3, 3])
geometry = NS(index_count=3, index_range=(4,3), vertex_range=(8,3))
meshes = {1: NS(mesh=geometry, box_center=numpy.zeros(3), box_extents=numpy.ones(3))}
layout = draw_compiler.compile_destinations(2, grouping.batches,
    {10: draw_layout.LodGroup((1,), ())}, meshes, {3: True})
assert layout.arrays['main_draw_cmd']['first_index'].tolist() == [4]
assert layout.main_count == 2 and layout.prepass_count == 2
'''], cwd=tmp_path, text=True, capture_output=True, env=env)
	assert result.returncode == 0, result.stderr


def test_scene_imports_do_not_initialize_glfw_or_gpu():
	result = subprocess.run([sys.executable, "-c", '''
import glfw, wgpu
def forbidden(*args, **kwargs): raise AssertionError('graphics initialization during import')
glfw.init = forbidden
wgpu.gpu.request_adapter_sync = forbidden
from scenes import transparency_example, draw_batches_example, direct_draw_example, prototype, msaa_example, material_example
assert all(callable(scene.run_scene) for scene in (transparency_example, draw_batches_example, direct_draw_example, prototype, msaa_example, material_example))
'''], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_cpu_imports_never_attempt_to_load_optional_backends():
	result = subprocess.run([sys.executable, "-c", """
import sys
import importlib.abc
import numpy
class ForbidBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'wgpu', 'glfw', 'jinja2', 'PIL', 'pygltflib', 'pyjoltphysics', 'imgui_bundle', 'psutil'}:
            raise AssertionError('Unexpected optional import: ' + name)
sys.meta_path.insert(0, ForbidBackends())
from graaal import Assets, SkeletalGltf, Geometry, EcsPhysics, PhysicsRagdoll, Diagnostics
from graaal.__main__ import simulate
from graaal.Diagnostics import WatchTimer, inspect_physics
assert simulate().shape == (2, 3)
with WatchTimer('cpu'): pass
from types import SimpleNamespace as NS
native = NS(tick=0, settings=NS(tick_hz=120), lockstep=True, stats=lambda: {'tick_ms': 0.0})
assert inspect_physics(native)['stats'] == {'tick_ms': 0.0}
assert 'graaal.RenderDiagnostics' not in sys.modules
assert Assets.load_blue_noise().shape == (64, 64)
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_ragdoll_helper_has_no_ecs_or_rendering_dependency():
	result = subprocess.run([sys.executable, "-c", """
import sys
from graaal.PhysicsRagdoll import RagdollAsset
assert not {'graaal.ECS', 'graaal.EcsPhysics', 'graaal.RenderContext', 'pyjoltphysics'} & sys.modules.keys()
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_rendering_imports_do_not_load_ecs_and_legacy_exports_still_work():
	result = subprocess.run([sys.executable, "-c", """
import sys
import importlib.abc
class ForbidEcs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in {'graaal.ECS', 'graaal.EcsMaterials', 'graaal.MaterialTransform'}:
            raise AssertionError('Unexpected ECS dependency: ' + name)
blocker = ForbidEcs()
sys.meta_path.insert(0, blocker)
from graaal import MaterialRenderer, RenderDiagnostics, Diagnostics
assert Diagnostics.RenderMonitor is RenderDiagnostics.RenderMonitor
assert Diagnostics.get_draw_counts is RenderDiagnostics.get_draw_counts
assert Diagnostics.__dict__['RenderMonitor'] is RenderDiagnostics.RenderMonitor
sys.meta_path.remove(blocker)
from graaal.EcsMaterials import MaterialInstance, EcsMaterialAdapter
from graaal.MaterialTransform import MaterialTransform
assert MaterialRenderer.MaterialInstance is MaterialInstance
assert MaterialRenderer.EcsMaterialAdapter is EcsMaterialAdapter
assert MaterialRenderer.MaterialTransform is MaterialTransform
assert MaterialRenderer.__dict__['MaterialInstance'] is MaterialInstance
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr

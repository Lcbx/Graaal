import ast
import os
import subprocess
import sys
from pathlib import Path


def test_public_package_exports_resolve_to_implementation_objects():
	result = subprocess.run([sys.executable, "-c", '''
import importlib
exports = {
    'ECS': ('ECS', 'component', 'columns', 'Scheduler', 'CommandBuffer'),
    'DrawBatches': ('DrawBatches', 'Visibility', 'FrustumCuller'),
    'Materials': ('MaterialShader', 'MaterialBinding', 'MaterialLibrary'),
    'RenderContext': ('RenderContext', 'GpuBuffer', 'Shader', 'CommandContext'),
    'Diagnostics': ('WatchTimer', 'inspect_physics'),
}
for name, names in exports.items():
    package = importlib.import_module('graaal.' + name)
    for public_name in names:
        value = getattr(package, public_name)
        implementation = importlib.import_module('graaal.' + name + '.core')
        assert value is getattr(implementation, public_name)
        assert getattr(package, public_name) is value
'''], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_subsystem_packages_are_lazy_and_animation_needs_no_ecs_or_backends():
	result = subprocess.run([sys.executable, "-c", '''
import sys
import importlib
import importlib.abc
import graaal.ECS
import graaal.Animation
import graaal.DrawBatches
import graaal.Materials
import graaal.RenderContext
import graaal.RenderPasses
import graaal.EcsAdapters
import graaal.Diagnostics
import graaal.UI
packages = ('ECS', 'Animation', 'DrawBatches', 'Materials', 'RenderContext', 'RenderPasses', 'EcsAdapters', 'Diagnostics', 'UI')
assert not any(name.startswith(tuple('graaal.' + package + '.' for package in packages)) for name in sys.modules)
class ForbidDependencies(importlib.abc.MetaPathFinder):
	def find_spec(self, name, *args):
		if name.startswith('graaal.ECS.') or name.split('.')[0] in {'wgpu', 'glfw', 'jinja2', 'PIL', 'pygltflib', 'pyjoltphysics', 'imgui_bundle', 'psutil'}:
			raise AssertionError('Unexpected animation dependency: ' + name)
sys.meta_path.insert(0, ForbidDependencies())
for name in ('AnimationEvents', 'AnimationGraph', 'AnimationGraphPlayback', 'AnimationLod', 'AnimationStateMachine', 'AnimationSync', 'BakedClipPool', 'CrowdGraphSamples', 'CrowdGraphValidation', 'InverseKinematics', 'PosePalette', 'PoseTasks', 'RootMotion', 'SkeletalAnimation', 'SkeletalGltf'):
	importlib.import_module('graaal.Animation.' + name)
'''], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_inline_imports_are_confined_to_cold_paths():
	# Frame recording, simulation updates and array operations keep imports out.
	cold_paths = {
		"Assets": {"load_gltf_scene", "load_gltf_first_mesh", "load_blue_noise"},
		"Diagnostics/core": {"setup_memory_monitor"},
		"EcsPhysics": {"EcsPhysicsAdapter.__init__"},
		"PhysicsRagdoll": {"RagdollAsset.__init__"},
		"Geometry": {"make_cube_mesh"},
		"ImGui": {"ImGui.__init__"},
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
		module_key = path.relative_to(root).with_suffix("").as_posix()
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ".".join(scope) not in cold_paths.get(module_key, cold_paths.get(path.stem, ())):
			inline.extend(f"{path.name}:{item.lineno}" for item in ast.walk(node) if isinstance(item, (ast.Import, ast.ImportFrom)))
		for child in ast.iter_child_nodes(node): inspect(path, child, scope)
	for path in sorted(root.rglob("*.py")):
		inspect(path, ast.parse(path.read_text(encoding="utf-8-sig")))
	assert not inline, "Inline imports outside cold paths: " + ", ".join(inline)


def test_module_imports_work_in_either_order():
	for modules in (("ECS.core", "ECS.ecs_diagnostics"), ("ECS.ecs_diagnostics", "ECS.core"), ("RenderContext.core", "RenderContext.MeshArena"), ("RenderContext.MeshArena", "RenderContext.core")):
		result = subprocess.run([sys.executable, "-c", f"import graaal.{modules[0]}; import graaal.{modules[1]}"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
		assert result.returncode == 0, result.stderr


def test_public_gpu_profiler_export_is_stable_in_either_import_order():
	for first, second in (("GpuProfiler", "RenderMonitor"), ("RenderMonitor", "GpuProfiler")):
		result = subprocess.run([sys.executable, "-c", f'''
from graaal.Diagnostics import {first}, {second}
from graaal.Diagnostics.gpu_profiler import GpuProfiler as implementation
assert GpuProfiler is implementation
assert callable(GpuProfiler)
'''], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
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
from graaal.ECS import ECS
from graaal import Math, Assets
from graaal.DrawBatches import InstanceData
from graaal.RenderPasses import TransparencyData
from graaal.DrawBatches import DrawList
from graaal.DrawBatches import draw_compiler, draw_layout
from graaal import utils
from graaal.Animation import SkeletalAnimation, SkeletalGltf, RootMotion, AnimationEvents, AnimationGraph
from graaal.EcsAdapters import EcsAnimation
from graaal.RenderContext import Geometry
from graaal.RenderContext.ShaderSyntax import strip_comments
from graaal.RenderPasses.pass_settings import AmbientOcclusionSettings, AntialiasingSettings
from graaal.UI.diagnostics_visualizer import DiagnosticsVisualizer
from graaal.EcsAdapters import EcsPhysics
from graaal import PhysicsRagdoll
assert PhysicsRagdoll.joint_spring_torques([-1], [[[0, 0, 0, 1]]], [[[0, 0, 0]]], [[0, 0, 0, 1]], stiffness=1, damping=1, max_torque=1).tolist() == [[[0, 0, 0]]]
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
world = ECS()
world.register(MaterialTransform)
entity = world.create(1)
world.add(entity, MaterialTransform())
assert world.get(MaterialTransform)[entity].rotation.tolist() == [[0, 0, 0, 1]]
assert 'graaal.EcsAdapters.EcsMaterials' not in sys.modules
assert Math.Quaternion.from_y_rotation(0.5).shape == (4,)
assert strip_comments('fn main() {} // comment').startswith('fn main() {}')
assert AmbientOcclusionSettings().radius > 0
assert AntialiasingSettings().faaa_quality > 0
from types import SimpleNamespace as NS
assert not PhysicsRagdoll.joint_limit_torques([-1], [[[0, 0, 0, 1]]], [[[0, 0, 0]]], [[0, 0, 0, 1]], [[0, 0, 0, 1]], twist=(-0.6, 0.6), swing=(0.6, 0.6), margin=0.1, stiffness=1, damping=1, max_torque=1).any()
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
from graaal import Assets, PhysicsRagdoll, Diagnostics
from graaal.RenderContext import Geometry
from graaal.EcsAdapters import EcsPhysics
from graaal.Animation import SkeletalGltf
from graaal.__main__ import simulate
from graaal.Diagnostics import WatchTimer, inspect_physics
assert simulate().shape == (2, 3)
with WatchTimer('cpu'): pass
from types import SimpleNamespace as NS
native = NS(tick=0, settings=NS(tick_hz=120), lockstep=True, stats=lambda: {'tick_ms': 0.0})
assert inspect_physics(native)['stats'] == {'tick_ms': 0.0}
assert 'graaal.Diagnostics.RenderDiagnostics' not in sys.modules
assert Assets.load_blue_noise().shape == (64, 64)
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_ragdoll_helper_has_no_ecs_or_rendering_dependency():
	result = subprocess.run([sys.executable, "-c", """
import sys
from graaal.PhysicsRagdoll import RagdollAsset
assert not {'graaal.ECS', 'graaal.ECS', 'graaal.EcsAdapters.EcsPhysics', 'graaal.RenderContext', 'pyjoltphysics'} & sys.modules.keys()
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr


def test_rendering_imports_do_not_load_ecs_or_adapters():
	result = subprocess.run([sys.executable, "-c", """
import sys
import importlib.abc
class ForbidEcs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name == 'graaal.ECS' or name.startswith(('graaal.ECS.', 'graaal.EcsAdapters')):
            raise AssertionError('Unexpected ECS dependency: ' + name)
blocker = ForbidEcs()
sys.meta_path.insert(0, blocker)
from graaal.Materials import MaterialRenderer
from graaal.RenderPasses.CrowdPose import CrowdPosePass, GraphPosePass
from graaal import Diagnostics
from graaal.Diagnostics import RenderDiagnostics
assert Diagnostics.RenderMonitor is RenderDiagnostics.RenderMonitor
assert Diagnostics.get_draw_counts is RenderDiagnostics.get_draw_counts
assert Diagnostics.__dict__['RenderMonitor'] is RenderDiagnostics.RenderMonitor
assert not any(name.startswith('graaal.EcsAdapters') for name in sys.modules)
"""], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
	assert result.returncode == 0, result.stderr

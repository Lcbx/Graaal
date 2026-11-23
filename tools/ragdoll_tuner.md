# Ragdoll tuner

Run from the repository root with a Python installation that includes Tk and
the Graaal physics extra.

```sh
python -m pip install -e ".[physics]"
python -m tools.ragdoll_tuner
python -m tools.ragdoll_tuner path/to/rig.tuning.json
```

The tool starts with the passive example’s 13-body humanoid. **Open rig** also
accepts JSON written from `pyjoltphysics.RagdollSettings.as_dict()`. That
format supplies the skeleton and hard constraints. An imported rig starts
with its rest pose as the neutral pose and zero passive gains.

Select a bone in the tree or click its visible collider in the depth-tested
viewport. Click and drag to orbit and scroll to zoom.

- **Collider** edits mass, box/sphere/capsule/cylinder dimensions, and the
  bone-local offset and rotation. Box dimensions are full side lengths. A
  capsule’s cylinder length excludes its hemispheres.
- **Joint** edits hard twist and swing, swing centre bend, joint friction,
  and optional twist and plane axes. Leave an axis blank for native defaults.
- **Passive** edits the parent-relative neutral pose, neutral spring gains,
  and the soft-limit margin and gains. The soft-limit margin is measured
  inward from the hard limits. A zero torque cap disables that joint’s
  corresponding passive force.

Angle fields use degrees. Rotation fields take XYZ angles with matrix
composition `Rx @ Ry @ Rz`. Separate values with spaces or commas. The root
has collider settings, since each joint connects a child to its parent.

**Apply to bone** validates the edits and restarts the paused preview from
tick zero. **Revert fields** restores the last applied values for the selected
bone. Choose **Drop**, **Stairs**, or **Zero gravity**, then play, pause, step
one 120 Hz tick, or reset. Stairs opens from the downhill side, matching the
passive example viewer. Applying changes and resetting within a scenario keep
your orbit and zoom. Clear **Passive torques** to compare the same rig
using only the native hard constraints. **Push bone** applies a world-space
impulse in N·s to the selected body.

**Save** writes a versioned JSON project containing native settings, neutral
rotations, and passive gains. Importing plain native settings starts a separate
project file. Load the saved project in the [passive ragdoll example](../scenes/passive_ragdoll_example.py).

```sh
python -m scenes.passive_ragdoll_example --tuning rig.tuning.json --scenario stairs
```

Other scenes and standalone native applications can read the same project.

```python
from tools.ragdoll_project import read_project

rig = read_project("rig.tuning.json")
# rig.settings is a native RagdollSettings.
# Pass rig.neutral and **rig.spring to joint_spring_torques.
# Pass **rig.limit_parameters() to joint_limit_torques.
```

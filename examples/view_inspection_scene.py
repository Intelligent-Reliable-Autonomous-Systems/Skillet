# examples/view_inspection_scene.py
import mujoco
import mujoco.viewer
from skillet_tasks.mj_tasks.planning.inspection_pick_and_place.scene_factory import make_inspection_scene

spec = make_inspection_scene([False, True, False])  # clean, defective, clean
data = mujoco.MjData(spec.model)
mujoco.mj_resetData(spec.model, data)

with mujoco.viewer.launch_passive(spec.model, data) as v:
    while v.is_running():
        mujoco.mj_step(spec.model, data)
        v.sync()

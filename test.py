from skillet.scene import THREE_CUBE_SCENE
from skillet.scene.base import Scene
from skillet.scene.abstract.abstract_model import AbstractModel
import pickle
from skillet import DEVICE

domain = "skillet/scene/abstract/assets/blocks.domain.pddl"
task = None

ab_model = AbstractModel(domain_file=domain)

with open("data/test/vlm_out_holding.pkl", "rb") as f:
    scene: Scene = pickle.load(f)

b = scene.get_objects_from_name(["red_block"])[0]
scene.tcp_pose = b.pose.clone()
scene.gripper_pos = 0.8
ab_model.initialize(scene, task)

ab_state = ab_model.get_abstract_state()
result, plan = ab_model.plan(ab_state)

print(plan)

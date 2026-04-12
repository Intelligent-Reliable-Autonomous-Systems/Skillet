from skillet.scene.abstract.abstract_model import AbstractModel
from skillet.scene import THREE_CUBE_SCENE

domain = "skillet/scene/abstract/assets/blocks.domain.pddl"
task = None  # "skillet/scene/abstract/assets/empty-problem.problem.pddl"

ab_model = AbstractModel(domain_file=domain)

ab_model.initialize(THREE_CUBE_SCENE, task)

ab_state = ab_model.get_abstract_state()
result, plan = ab_model.plan(ab_state)

print(plan)

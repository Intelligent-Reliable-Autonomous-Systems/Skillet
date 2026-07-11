from unified_planning.engines import PlanGenerationResultStatus as PGResultStatus
from unified_planning.io import PDDLReader
from unified_planning.model import Problem
from unified_planning.shortcuts import OneshotPlanner

domain = "skillet_tasks/spongeworld-clean/simple-sponge.domain.pddl"
task = "skillet_tasks/spongeworld-clean/simple-sponge.problem.pddl"

reader = PDDLReader()
problem: Problem = reader.parse_problem(domain, task)

with OneshotPlanner(name="fast-downward") as planner:
    result = planner.solve(problem, timeout=30)

    status = result.status

    if status not in (PGResultStatus.SOLVED_SATISFICING, PGResultStatus.SOLVED_OPTIMALLY):
        print("FAIL")

print(result.plan.actions)

# class TAMPAgent(Agent):

#     def execute(env: Environment) -> None:
#         """A Task and Motion Planner executor for running an agent in an environment."""

#         # Get the initial observation

#         # 1. Do high level planning -> Get a sequence of skills to execute
#         # 2. Do parameter sampling -> Get a set of parameters for each skill
#         obs = env.get_observation(agent.task_observation_spec)
#         skill_sequence, skill_params: list[Skill], list[SkillParams] = agent.get_skill_sequence(obs, task)
#         skill_params: list[SkillParams] = agent.get_skill_params(skill_sequence, obs, task)

#         # 3. Execute the skills:
#         for skill, params in zip(skill_sequence, skill_params):
#             obs = env.get_observation(skill.observation_spec)
#             #   3.2. Check if skill can be initiated, else fail
#             if not skill.can_initiate(obs):
#                 return SkillStatus.FAILED
#             skill.initiate(obs, params)
#             #   3.3. While not terminated, query the skill controller to get the next action and take a step in the environment
#             action_or_status = skill.get_action(obs)
#             while not isinstance(action_or_status, SkillStatus):
#                 _, reward, term, trunc, info = env.step(action)
#                 obs = env.get_observation(skill.observation_spec)
#                 action_or_status = skill.get_action(obs)
#             #   3.4. If skill reports success, move to next skill, else fail
#             if action_or_status != SkillStatus.SUCCESS:
#                 return SkillStatus.FAILED

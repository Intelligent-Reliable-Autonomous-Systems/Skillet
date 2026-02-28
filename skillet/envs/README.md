# Environments

All of the different environment formats get pretty crazy.
- gym.Env / gym.EnvWrapper: classic unbatched environment
- gym.vector.VectorEnv / gym.vector.VectorWrapper: new gymnasium 1.0 vector format
- DirectRLEnv / ManagerBasedRLEnv: IsaacLab environment formats
- rsl.env.VecEnv environment format: Part of RSL_RL reinforcement learning library
- skillet.core.Environment / skillet.core.BatchedEnvironment: our own custom environment that adds observation and action specifications.
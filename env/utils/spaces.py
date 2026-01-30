"""utils.py

Utils for configuring a space
"""

import json
from typing import TypeVar

import gymnasium as gym
import numpy as np

SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)


def serialize_space(space: SpaceType) -> str:
    """Serialize a space specification as JSON.

    Args:
        space: Space specification.

    Returns:
        Serialized JSON representation.

    """
    # Gymnasium spaces
    if isinstance(space, gym.spaces.Discrete):
        return json.dumps({"type": "gymnasium", "space": "Discrete", "n": int(space.n)})
    if isinstance(space, gym.spaces.Box):
        return json.dumps(
            {
                "type": "gymnasium",
                "space": "Box",
                "low": space.low.tolist(),
                "high": space.high.tolist(),
                "shape": space.shape,
            }
        )
    if isinstance(space, gym.spaces.MultiDiscrete):
        return json.dumps({"type": "gymnasium", "space": "MultiDiscrete", "nvec": space.nvec.tolist()})
    if isinstance(space, gym.spaces.Tuple):
        return json.dumps({"type": "gymnasium", "space": "Tuple", "spaces": tuple(map(serialize_space, space.spaces))})
    if isinstance(space, gym.spaces.Dict):
        return json.dumps(
            {"type": "gymnasium", "space": "Dict", "spaces": {k: serialize_space(v) for k, v in space.spaces.items()}}
        )
    # Python data types
    # Box
    if isinstance(space, int) or (isinstance(space, list) and all(isinstance(x, int) for x in space)):
        return json.dumps({"type": "python", "space": "Box", "value": space})
    # Discrete
    if isinstance(space, set) and len(space) == 1:
        return json.dumps({"type": "python", "space": "Discrete", "value": next(iter(space))})
    # MultiDiscrete
    if isinstance(space, list) and all(isinstance(x, set) and len(x) == 1 for x in space):
        return json.dumps({"type": "python", "space": "MultiDiscrete", "value": [next(iter(x)) for x in space]})
    # composite spaces
    # Tuple
    if isinstance(space, tuple):
        return json.dumps({"type": "python", "space": "Tuple", "value": [serialize_space(x) for x in space]})
    # Dict
    if isinstance(space, dict):
        return json.dumps(
            {"type": "python", "space": "Dict", "value": {k: serialize_space(v) for k, v in space.items()}}
        )
    raise ValueError(f"Unsupported space ({space})")


def deserialize_space(string: str) -> gym.spaces.Space:
    """Deserialize a space specification encoded as JSON.

    Args:
        string: Serialized JSON representation.

    Returns:
        Space specification.

    """
    obj = json.loads(string)
    # Gymnasium spaces
    if obj["type"] == "gymnasium":
        if obj["space"] == "Discrete":
            return gym.spaces.Discrete(n=obj["n"])
        if obj["space"] == "Box":
            return gym.spaces.Box(low=np.array(obj["low"]), high=np.array(obj["high"]), shape=obj["shape"])
        if obj["space"] == "MultiDiscrete":
            return gym.spaces.MultiDiscrete(nvec=np.array(obj["nvec"]))
        if obj["space"] == "Tuple":
            return gym.spaces.Tuple(spaces=tuple(map(deserialize_space, obj["spaces"])))
        if obj["space"] == "Dict":
            return gym.spaces.Dict(spaces={k: deserialize_space(v) for k, v in obj["spaces"].items()})
        raise ValueError(f"Unsupported space ({obj['spaces']})")
    # Python data types
    if obj["type"] == "python":
        if obj["space"] == "Discrete":
            return {obj["value"]}
        if obj["space"] == "Box":
            return obj["value"]
        if obj["space"] == "MultiDiscrete":
            return [{x} for x in obj["value"]]
        if obj["space"] == "Tuple":
            return tuple(map(deserialize_space, obj["value"]))
        if obj["space"] == "Dict":
            return {k: deserialize_space(v) for k, v in obj["value"].items()}
        raise ValueError(f"Unsupported space ({obj['spaces']})")
    raise ValueError(f"Unsupported type ({obj['type']})")


def replace_env_cfg_spaces_with_strings(env_cfg: object) -> object:
    """Replace spaces objects with their serialized JSON representations in an environment config.

    Args:
        env_cfg: Environment config instance.

    Returns:
        Environment config instance with spaces replaced if any.

    """
    for attr in ["observation_space", "action_space", "state_space"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, serialize_space(getattr(env_cfg, attr)))
    for attr in ["observation_spaces", "action_spaces"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: serialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg


def replace_strings_with_env_cfg_spaces(env_cfg: object) -> object:
    """Replace spaces objects with their serialized JSON representations in an environment config.

    Args:
        env_cfg: Environment config instance.

    Returns:
        Environment config instance with spaces replaced if any.

    """
    for attr in ["observation_space", "action_space", "state_space"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, deserialize_space(getattr(env_cfg, attr)))
    for attr in ["observation_spaces", "action_spaces"]:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: deserialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg

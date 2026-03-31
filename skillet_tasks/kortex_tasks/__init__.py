"""Top level init file for tasks."""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``tasks``: These include single-file implementations of tasks.
- ``utils``: These include utility functions for the tasks.

"""

##
# Register Gym environments.
##
import collections
import collections.abc

collections.MutableMapping = collections.abc.MutableMapping
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableSequence = collections.abc.MutableSequence
collections.Callable = collections.abc.Callable

from skillet.envs.util import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

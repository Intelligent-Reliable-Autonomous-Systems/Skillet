"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``tasks``: These include single-file implementations of tasks.
- ``utils``: These include utility functions for the tasks.

"""

##
# Register Gym environments.
##

from skillet.envs.util import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

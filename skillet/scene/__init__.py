import contextlib

from .scene_objs import Cube as Cube
from .scene_objs import Location as Location
from .scene_objs import Sponge as Sponge
from .scene_objs import Table as Table
from .scene_objs import Target as Target
from .scenes import CUBE_SIZE as CUBE_SIZE
from .scenes import EMPTY_SCENE as EMPTY_SCENE
from .scenes import FIVE_CUBE_SCENE as FIVE_CUBE_SCENE
from .scenes import FOUR_CUBE_SCENE as FOUR_CUBE_SCENE
from .scenes import LOC_CUBE_SCENE as LOC_CUBE_SCENE
from .scenes import SIX_CUBE_APRIL_SCENE as SIX_CUBE_APRIL_SCENE
from .scenes import SIX_CUBE_SCENE as SIX_CUBE_SCENE
from .scenes import SPONGE_SCENE as SPONGE_SCENE
from .scenes import THREE_CUBE_APRIL_SCENE as THREE_CUBE_APRIL_SCENE
from .scenes import THREE_CUBE_SCENE as THREE_CUBE_SCENE

with contextlib.suppress(ImportError):
    from .visualization import Open3DVisualizer as Open3DVisualizer

# Scene Objects
from .objects import DiscardLocation as DiscardLocation
from .objects import InspectableCube as InspectableCube
from .objects import Platform as Platform

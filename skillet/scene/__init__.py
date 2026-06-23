import contextlib

from .scene_objs import Cube as Cube
from .scene_objs import Location as Location
from .scene_objs import Spill as Spill
from .scene_objs import Sponge as Sponge
from .scene_objs import Table as Table
from .scene_objs import Target as Target
from .scenes import CUBE_SIZE as CUBE_SIZE
from .scenes import SPILL_SIZE as SPILL_SIZE
from .scenes import SPONGE_SIZE as SPONGE_SIZE
from .scenes import TARGET_SIZE as TARGET_SIZE
from .scenes import empty_scene_loader as empty_scene_loader
from .scenes import five_cube_scene_loader as five_cube_scene_loader
from .scenes import four_cube_scene_loader as four_cube_scene_loader
from .scenes import one_cube_scene_loader as one_cube_scene_loader
from .scenes import seven_cube_scene_loader as seven_cube_scene_loader
from .scenes import six_cube_april_scene_loader as six_cube_april_scene_loader
from .scenes import six_cube_scene_loader as six_cube_scene_loader
from .scenes import sponge_scene_loader as sponge_scene_loader
from .scenes import three_cube_april_scene_loader as three_cube_april_scene_loader
from .scenes import three_cube_scene_loader as three_cube_scene_loader

with contextlib.suppress(ImportError):
    from .visualization import Open3DVisualizer as Open3DVisualizer

# Scene Objects
from .objects import DiscardLocation as DiscardLocation
from .objects import InspectableCube as InspectableCube
from .objects import Platform as Platform

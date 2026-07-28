import contextlib

from .scene_objs import Bin as Bin
from .scene_objs import Can as Can
from .scene_objs import Cube as Cube
from .scene_objs import Location as Location
from .scene_objs import Plate as Plate
from .scene_objs import Spill as Spill
from .scene_objs import Sponge as Sponge
from .scene_objs import Table as Table
from .scene_objs import Target as Target
from .scenes import BIN_SIZE as BIN_SIZE
from .scenes import CAN_SIZE as CAN_SIZE
from .scenes import CUBE_SIZE as CUBE_SIZE
from .scenes import PLATE_SIZE as PLATE_SIZE
from .scenes import SPILL_SIZE as SPILL_SIZE
from .scenes import SPONGE_SIZE as SPONGE_SIZE
from .scenes import TARGET_SIZE as TARGET_SIZE
from .scenes import load_scene as load_scene

with contextlib.suppress(ImportError):
    from .visualization import Open3DVisualizer as Open3DVisualizer

# Scene Objects
from .objects import DiscardLocation as DiscardLocation
from .objects import InspectableCube as InspectableCube
from .objects import Platform as Platform

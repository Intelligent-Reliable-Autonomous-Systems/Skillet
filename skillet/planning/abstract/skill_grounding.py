from skillet.planning.abstract.spatial_grounding import (
    _gripper_closed,
    _is_above_loc,
    _is_at,
    _is_grasping,
    _is_north_of,
    _is_obstructed_above,
    _is_occupied,
    _is_on,
    _is_on_table,
)
from skillet.scene import Cube, Table
from skillet.scene.base import Scene, SceneObject


def _pick_skill_4_grounding(scene_objs: list[SceneObject], scene: Scene) -> bool:
    """Ground the pick 4 skill.

    (:action pick-block
    :parameters (?target - block ?support - surface ?targetloc ?supportloc - location)
    :precondition (and
        (at-loc ?target ?targetloc)
        (at-loc ?support ?supportloc)
        (loc-above ?targetloc ?supportloc)
        (on ?target ?support)

        (not (obstructed-above ?targetloc)) - TODO
        (or (not (gripper-full)) (grasping ?target))
    )

    """
    target, support, targetloc, supportloc = scene_objs[:4]
    at_grd = _is_at(target, targetloc) and (
        (_is_at(support, supportloc) and isinstance(support, Cube)) or isinstance(support, Table)
    )
    above_grd = _is_above_loc(targetloc, supportloc)
    if isinstance(support, Cube):
        on_grd = _is_on(target, support)
    elif isinstance(support, Table):
        on_grd = _is_on_table(target, support)
    else:
        on_grd = False
    grasp_grd = not _is_grasping(target, scene)
    full_grd = not _gripper_closed(scene)
    obs_above_grd = not _is_obstructed_above(targetloc, scene)

    return bool(at_grd and above_grd and on_grd and grasp_grd and full_grd and obs_above_grd)


def _place_skill_4_grounding(scene_objs: list[SceneObject], scene: Scene) -> bool:
    """Ground the place 4 skill.

    (:action place-block
    :parameters (?grasped - block ?target - surface ?freeloc ?targetloc - location)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        ; this disjunction is necessary in the case where the block is already at the location but still being held
        (or (not (occupied ?freeloc)) (at-loc ?grasped ?freeloc))
        (at-loc ?target ?targetloc)
        (loc-above ?freeloc ?targetloc)
    )

    """
    grasped, target, freeloc, targetloc = scene_objs[:4]

    grasp_grd = _is_grasping(grasped, scene)
    full_grd = _gripper_closed(scene)
    at_grd = (not _is_at(grasped, freeloc)) and (
        (_is_at(target, targetloc) and isinstance(target, Cube)) or isinstance(target, Table)
    )
    above_grd = _is_above_loc(freeloc, targetloc)
    occupied_grd = not _is_occupied(freeloc, scene)

    return bool(grasp_grd and full_grd and at_grd and above_grd and occupied_grd)


def _drag_skill_5_grounding(scene_objs: list[SceneObject], scene: Scene) -> bool:
    """Ground the drag 5 skill.

    (:action drag-block
    :parameters (?grasped - block ?fromloc ?toloc ?belowfromloc ?belowtoloc - location)
    :precondition (and
        (at-loc ?grasped ?fromloc)
        (not (occupied ?toloc))
        (not (obstructed-above ?fromloc))
        (or (loc-north-of ?toloc ?fromloc) (loc-north-of ?fromloc ?toloc))
        (loc-above ?toloc ?belowtoloc)
        (loc-above ?fromloc ?belowfromloc)
        (not (gripper-full))
        (occupied ?belowtoloc)
    )

    """
    grasped, fromloc, toloc, belowfromloc, belowtoloc = scene_objs[:5]

    at_grd = _is_at(grasped, fromloc)
    full_grd = not _gripper_closed(scene)
    grasp_grd = not _is_grasping(grasped, scene)
    above_grd = _is_above_loc(toloc, belowtoloc) and _is_above_loc(fromloc, belowfromloc)
    north_grd = _is_north_of(toloc, fromloc) or _is_north_of(fromloc, toloc)
    occupied_grd = _is_occupied(belowtoloc, scene) and (not _is_occupied(toloc, scene))
    obs_above_grd = not _is_obstructed_above(fromloc, scene)

    return bool(at_grd and full_grd and grasp_grd and above_grd and north_grd and occupied_grd and obs_above_grd)

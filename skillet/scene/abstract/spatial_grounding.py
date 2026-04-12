"""Geometric grounding of spatial predicates from scene object poses."""

from typing import Literal

from skillet.scene.base import Scene, SceneObject
from skillet.scene.cube import Cube, Table


def _is_on(a: Cube, b: Cube, height_tol_frac: float = 0.3, xy_slack_frac: float = 0.5) -> bool:
    """Return True if cube *a* is resting on top of cube *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.
        height_tol_frac: Tolerance for the vertical gap check, as a fraction of
            the smaller cube's side length.
        xy_slack_frac: Extra horizontal slack allowed beyond b's footprint
            edges, as a fraction of the smaller cube's side length.

    """
    if not (a.is_pose_known() and b.is_pose_known()):
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]
    aabb_b = b.aabb

    tol = min(a.size, b.size) * height_tol_frac
    slack = min(a.size, b.size) * xy_slack_frac

    # a's bottom should be sitting at roughly the height of b's top surface
    if not abs(aabb_a[2] - aabb_b[5]) < tol:
        return False

    # a's xy centre should fall within b's footprint (plus a little slack)
    a_cx = (aabb_a[0] + aabb_a[3]) / 2.0
    a_cy = (aabb_a[1] + aabb_a[4]) / 2.0

    within_x = (aabb_b[0] - slack) <= a_cx <= (aabb_b[3] + slack)
    within_y = (aabb_b[1] - slack) <= a_cy <= (aabb_b[4] + slack)

    return bool(within_x and within_y)


def _is_on_table(a: Cube, table: Table, height_tol_frac: float = 0.5) -> bool:
    """Return True if cube *a* is resting on the table.

    Args:
        a: The candidate cube.
        table: the table object in the scene
        height_tol_frac: Tolerance for the vertical gap check, as a fraction of
            the smaller cube's side length.

    """
    if not (a.is_pose_known()):
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]

    tol = a.size * height_tol_frac

    # a's bottom should be sitting at roughly the height of table surface
    return abs(aabb_a[2] - table.height) < tol


def ground_cube_on_relations(scene: Scene) -> list[tuple[Literal["on"], SceneObject, SceneObject]]:
    """Ground the on relations in the scene."""
    on_relations = []
    clear_relations = [("clear", scene.table)]
    table = scene.table

    cube_list = []
    for obj in scene.objects:
        if not isinstance(obj, Cube):
            continue
        cube_list.append(obj)
        if table is not None and _is_on_table(obj, table):
            on_relations.append(("on", obj, table))
        for other_obj in scene.objects:
            if not isinstance(other_obj, Cube):
                continue
            if obj.object_id != other_obj.object_id and _is_on(obj, other_obj):
                on_relations.append(("on", obj, other_obj))

    # Remove cubes from clear list if they have an object on top
    for o in on_relations:
        if o[2] in cube_list:
            cube_list.remove(o[2])
    [clear_relations.append(("clear", obj)) for obj in cube_list]
    return on_relations, clear_relations


def ground_gripper_relations(scene: Scene) -> tuple[bool, list[tuple[Literal["holding"], SceneObject]]]:
    """Grounding for if the gripper hand is empty.

    TODO: Compare against cube centers and tcp pose. Add TCP pose to scene.
    """
    return True, []

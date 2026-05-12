"""Geometric grounding of spatial predicates from scene object poses."""

from typing import Literal

from skillet.scene.base import Scene, SceneObject
from skillet.scene.scene_objs import Cube, Table


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


def _is_north_of(a: Cube, b: Cube, north_tol_frac: float = 0.3, yz_slack_frac: float = 0.5) -> bool:
    """Return True if cube *a* is north of cube *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.
        north_tol_frac: Tolerance for the horizontal +x gap check, as a fraction of
            the smaller cube's side length.
        yz_slack_frac: Extra horizontal slack allowed beyond b's footprint
            edges, as a fraction of the smaller cube's side length.

    """
    if not (a.is_pose_known() and b.is_pose_known()):
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]
    aabb_b = b.aabb

    tol = min(a.size, b.size) * north_tol_frac
    slack = min(a.size, b.size) * yz_slack_frac

    # a's south side should be sitting at roughly b's north side (along +x)
    if not abs(aabb_a[0] - aabb_b[3]) < tol:
        return False

    # a's yz centre should fall within b's footprint (plus a little slack)
    a_cy = (aabb_a[1] + aabb_a[4]) / 2.0
    a_cz = (aabb_a[2] + aabb_a[4]) / 2.0

    within_y = (aabb_b[1] - slack) <= a_cy <= (aabb_b[4] + slack)
    within_z = (aabb_b[2] - slack) <= a_cz <= (aabb_b[4] + slack)

    return bool(within_y and within_z)


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


def _is_holding(a: Cube, scene: Scene, xyz_slack_frac: float = 0.8, gripper_thresh: float = 0.5) -> bool:
    """Test if the gripper is holding a cube.

    The cube xyz position must be sufficiently close to the tcp pose
    in the scene and the gripper must be closed.

    Args:
        a: Cube to test against
        scene: scene containing tcp pose and gripper position
        xyz_slack_frac: slack around xyz to still be considered holding
        gripper_thresh: threshold for gripper to be considered closed

    """
    # Check that the scene has both gripper and tcp pose
    if scene.tcp_pose is None or scene.gripper_pos is None or not a.is_pose_known():
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]
    slack = a.size * xyz_slack_frac

    # tcp pose should be within a's footprint (plus a little slack)
    within_x = (aabb_a[0] - slack) <= scene.tcp_pose[0] <= (aabb_a[3] + slack)
    within_y = (aabb_a[1] - slack) <= scene.tcp_pose[1] <= (aabb_a[4] + slack)
    within_z = (aabb_a[1] - slack) <= scene.tcp_pose[2] <= (aabb_a[5] + slack)

    # To be holding must be within footprint and gripper must be closed
    return bool(within_x and within_y and within_z and scene.gripper_pos > gripper_thresh)


def ground_cube_on_relations(scene: Scene) -> list[tuple[Literal["on"], SceneObject, SceneObject]]:
    """Ground the on relations in the scene."""
    on_relations = []
    north_relations = []
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
            if obj.object_id != other_obj.object_id and _is_north_of(obj, other_obj):
                north_relations.append(("north-of", obj, other_obj))

    # Remove cubes from clear list if they have an object on top
    for o in on_relations:
        if o[2] in cube_list:
            cube_list.remove(o[2])
    [clear_relations.append(("clear", obj)) for obj in cube_list]
    return on_relations, clear_relations, north_relations


def ground_gripper_relations(scene: Scene) -> tuple[bool, list[tuple[Literal["holding"], SceneObject]]]:
    """Grounding for if the gripper hand is empty and the object it is holding."""
    holding_relations = []
    for obj in scene.objects:
        if not isinstance(obj, Cube):
            continue
        if _is_holding(obj, scene):
            holding_relations.append(("holding", obj))
    return len(holding_relations) == 0, holding_relations


# get relations for north of/south of above/below by hard coding locations on the table

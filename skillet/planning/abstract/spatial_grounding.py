"""Geometric grounding of spatial predicates from scene object poses."""

from typing import Literal

import torch

from skillet.scene.base import Scene, SceneObject
from skillet.scene.scene_objs import Cube, Location, Table


def _is_on(a: Cube | Location, b: Cube | Location, height_tol_frac: float = 0.3, xy_slack_frac: float = 0.5) -> bool:
    """Return True if cube *a* is resting on top of cube *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.
        height_tol_frac: Tolerance for the vertical gap check, as a fraction of
            the smaller cube's side length.
        xy_slack_frac: Extra horizontal slack allowed beyond b's footprint
            edges, as a fraction of the smaller cube's side length.

    """
    if isinstance(a, Cube) and isinstance(b, Cube) and not (a.is_pose_known() and b.is_pose_known()):
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


def _is_north_of(
    a: Cube | Location, b: Cube | Location, north_tol_frac: float = 0.3, yz_slack_frac: float = 0.2
) -> bool:
    """Return True if cube *a* is north of cube *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.
        north_tol_frac: Tolerance for the horizontal +x gap check, as a fraction of
            the smaller cube's side length.
        yz_slack_frac: Extra horizontal slack allowed beyond b's footprint
            edges, as a fraction of the smaller cube's side length.

    """
    if isinstance(a, Cube) and isinstance(b, Cube) and not (a.is_pose_known() and b.is_pose_known()):
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


def _is_north_of_loc(a: Location, b: Location) -> bool:
    """Return True if location *a* is north of location *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.

    """
    return bool(torch.isclose(a.pose[0], b.pose[0] + b.size).item() and torch.isclose(a.pose[2], b.pose[2]).item())


def _is_above_loc(a: Location, b: Location) -> bool:
    """Return True if location *a* is above location *b*.

    Args:
        a: The candidate upper cube.
        b: The candidate lower cube.

    """
    return bool(
        torch.isclose(a.pose[0], b.pose[0]).item() and torch.isclose(a.pose[2], b.pose[2] + 0.05).item()
    )  # NOTE hardcoded


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


def _is_grasping(
    a: Cube, scene: Scene, z_slack_frac: float = 0.5, xy_slack_frac: float = 0.3, gripper_thresh: float = 0.4
) -> bool:
    """Test if the gripper is holding a cube.

    The cube xyz position must be sufficiently close to the tcp pose
    in the scene and the gripper must be closed.

    Args:
        a: Cube to test against
        scene: scene containing tcp pose and gripper position
        z_slack_frac: the z fraction tolerance to be inside
        xy_slack_frac: slack around xy to still be considered holding
        gripper_thresh: threshold for gripper to be considered closed

    """
    # Check that the scene has both gripper and tcp pose
    if scene.tcp_pose is None or scene.gripper_pos is None or not a.is_pose_known():
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]
    xy_slack = a.size * xy_slack_frac
    z_slack = a.size * z_slack_frac

    # tcp pose should be within a's footprint (plus a little slack)
    within_x = (aabb_a[0] - xy_slack) <= scene.tcp_pose[0] <= (aabb_a[3] + xy_slack)
    within_y = (aabb_a[1] - xy_slack) <= scene.tcp_pose[1] <= (aabb_a[4] + xy_slack)
    within_z = (aabb_a[2] - z_slack) <= scene.tcp_pose[2] <= (aabb_a[5] + z_slack * 1.5)

    # To be holding must be within footprint and gripper must be closed
    return bool(within_x and within_y and within_z and scene.gripper_pos > gripper_thresh)


def _gripper_closed(scene: Scene, gripper_thresh: float = 0.4) -> bool:
    """Check if the gripper is closed."""
    if scene.gripper_pos is None:
        return False
    return bool((scene.gripper_pos > gripper_thresh).item())


def _is_lifted(scene: Scene, lift_height: float = 0.2) -> bool:
    if scene.tcp_pose is None or scene.gripper_pos is None:
        return False
    return bool((scene.tcp_pose[2] > lift_height).item())


def _is_at(a: Cube, l: Location, z_slack_frac: float = 0.00, xy_slack_frac: float = 0.1) -> bool:
    """Test if a cube is at a location.

    The cube xyz position must be sufficiently close to the tcp pose
    in the scene and the gripper must be closed.

    Args:
        a: Cube to test against
        l: Location to test against
        z_slack_frac: the z fraction tolerance to be inside
        xy_slack_frac: slack around xy to still be considered holding

    """
    # Check that the scene has both gripper and tcp pose
    if not a.is_pose_known():
        return False

    aabb_a = a.aabb  # [xmin, ymin, zmin, xmax, ymax, zmax]
    xy_slack = a.size * xy_slack_frac
    z_slack = a.size * z_slack_frac

    # tcp pose should be within a's footprint (plus a little slack)
    within_x = l.pose[0] - xy_slack <= a.pose[0] <= (l.pose[0] + l.size + xy_slack)
    within_y = True
    within_z = l.pose[2] - z_slack <= a.pose[2] <= (l.pose[2] + a.size) + z_slack
    # (aabb_a[1] - z_slack) <= l.pose[2] + l.size / 2 <= (aabb_a[5] + z_slack)

    # To be holding must be within footprint and gripper must be closed
    return bool(within_x and within_y and within_z)


def _is_occupied(l: Location, scene: Scene) -> bool:
    """Test if a location is occupied."""
    return any(isinstance(other_obj, Cube) and _is_at(other_obj, l) for other_obj in scene.objects)


def _is_obstructed_above(loc: Location, scene: Scene) -> bool:
    """Test if a location is obstructed above."""
    above_relations = []
    occupied_relations = set()
    obstructed_above_relations = []
    for obj in scene.objects:
        if isinstance(obj, Table):
            continue
        for other_obj in scene.objects:
            if isinstance(other_obj, Location):
                if obj.object_id != other_obj.object_id and _is_above_loc(obj, other_obj):
                    above_relations.append(("loc-above", obj, other_obj))

            elif isinstance(other_obj, Cube) and _is_at(other_obj, obj):
                occupied_relations.add(("occupied", obj))

    occupied_relations = list(occupied_relations)
    above = [a[1] for a in above_relations]
    for o in occupied_relations:
        l = o[1]
        if l in above:
            obstructed_above_relations.append(above_relations[above.index(l)][2])

    return loc in obstructed_above_relations


def ground_cube_relations(scene: Scene) -> tuple[list[tuple[str, SceneObject, SceneObject]], ...]:
    """Ground the on relations in the scene."""
    on_relations = []
    north_relations = []
    clear_relations = [("clear", scene.table)]
    color_relations = []
    material_relations = []
    table = scene.table

    cube_list = []
    for obj in scene.objects:
        if not isinstance(obj, Cube):
            continue
        cube_list.append(obj)
        if obj.material is not None:
            material_relations.append((obj.material, obj))
        if not obj.moveable:
            material_relations.append(("immovable", obj))
        if obj.color is not None:
            color_relations.append((obj.color, obj))
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
    return on_relations, clear_relations, north_relations, color_relations, material_relations


def ground_gripper_relations(scene: Scene) -> tuple[bool, list[tuple[Literal["holding"], SceneObject]]]:
    """Grounding for if the gripper hand is empty and the object it is holding."""
    grasping_relations = []
    two_held_relations = False
    three_held_relations = False
    for obj in scene.objects:
        if not isinstance(obj, Cube):
            continue
        # Check if we are grasping a block
        if _is_grasping(obj, scene):
            grasping_relations.append(("grasping", obj))
            for other_obj in scene.objects:
                if other_obj.object_id == obj.object_id:
                    continue
                # If we are grasping a block, check if there is another block below
                if isinstance(other_obj, Cube) and (other_obj != obj) and _is_on(obj, other_obj):
                    two_held_relations = True
                    # If there is another block below, check if there is a third block below that
                    for other_other_obj in scene.objects:
                        if other_other_obj.object_id in [other_obj.object_id, obj.object_id]:
                            continue
                        if isinstance(other_obj, Cube) and (other_obj != obj) and _is_on(other_obj, other_other_obj):
                            three_held_relations = True
    return (
        len(grasping_relations) == 0,
        grasping_relations,
        _is_lifted(scene),
        (two_held_relations and len(grasping_relations) != 0),
        (three_held_relations and len(grasping_relations) != 0),
    )


def ground_location_relations(scene: Scene) -> list[tuple[str, SceneObject, SceneObject]]:
    """Grounding of the location relations in the scene."""
    above_relations = []
    north_relations = []
    at_relations = []
    occupied_relations = set()
    obstructed_above_relations = []
    obstructed_south_relations = []
    obstructed_north_relations = []
    for obj in scene.objects:
        if isinstance(obj, Table):
            for other_obj in scene.objects:
                if isinstance(other_obj, Location) and other_obj.pose[2] < -0.05:
                    at_relations.append(("at-loc", obj, other_obj))
        if not isinstance(obj, Location):
            continue
        if obj.pose[2] < -0.05:
            occupied_relations.add(("occupied", obj))
        for other_obj in scene.objects:
            if isinstance(other_obj, Location):
                if obj.object_id != other_obj.object_id and _is_above_loc(obj, other_obj):
                    above_relations.append(("loc-above", obj, other_obj))
                if obj.object_id != other_obj.object_id and _is_north_of_loc(obj, other_obj):
                    north_relations.append(("loc-north-of", obj, other_obj))
            elif isinstance(other_obj, Cube):
                if _is_at(other_obj, obj):
                    at_relations.append(("at-loc", other_obj, obj))
                    occupied_relations.add(("occupied", obj))

    occupied_relations = list(occupied_relations)
    above = [a[1] for a in above_relations]
    north_of = [n[1] for n in north_relations]
    south_of = [s[2] for s in north_relations]
    for o in occupied_relations:
        l = o[1]
        if l in above:
            obstructed_above_relations.append(("obstructed-above", above_relations[above.index(l)][2]))
        if l in north_of:
            obstructed_north_relations.append(("obstructed-north", north_relations[north_of.index(l)][2]))
        if l in south_of:
            obstructed_south_relations.append(("obstructed-south", north_relations[south_of.index(l)][1]))
    return (
        above_relations,
        north_relations,
        at_relations,
        occupied_relations,
        obstructed_above_relations,
        obstructed_north_relations,
        obstructed_south_relations,
    )


def ground_sponge_relations(scene: Scene) -> tuple[list[tuple[str, SceneObject, SceneObject]], ...]:
    """Ground the sponge relations in the scene."""
    on_relations = []
    color_relations = []
    material_relations = []

    for obj in scene.objects:
        if obj.deformable is not None and obj.deformable:
            material_relations.append(("deformable", obj))
        if obj.supportable is not None and obj.supportable:
            material_relations.append(("supportable", obj))

        if obj.color is not None:
            color_relations.append((obj.color, obj))

        for other_obj in scene.objects:
            if not obj.supportable:
                continue
            if obj.object_id != other_obj.object_id and _is_on(obj, other_obj):
                on_relations.append(("on", obj, other_obj))

    return on_relations, material_relations, color_relations

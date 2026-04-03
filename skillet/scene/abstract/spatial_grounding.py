"""Geometric grounding of spatial predicates from scene object poses."""

from typing import Literal

from skillet.scene.base import Scene, SceneObject
from skillet.scene.cube import Cube


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


def ground_on_relations(scene: Scene) -> list[tuple[Literal['on'], SceneObject, SceneObject]]:
    """Ground the on relations in the scene."""
    on_relations = []
    for obj in scene.objects:
        if not isinstance(obj, Cube):
            continue
        for other_obj in scene.objects:
            if not isinstance(other_obj, Cube):
                continue
            if obj.object_id != other_obj.object_id and _is_on(obj, other_obj):
                on_relations.append(('on', obj, other_obj))
    return on_relations

"""Pinocchio-based near-collision monitor for low-level joint control.

Handles robot kinematics + collision geometry (URDF + SRDF filtering)
Projects 1 control step ahead from joint velocities to detect imminent collisions

"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import hppfcl
import numpy as np
import pinocchio as pin


@dataclass
class CollisionInfo:
    """Structured result for a single monitoring query."""

    near_collision: bool
    min_distance_now: float
    min_distance_predicted: float
    threshold: float
    safety_factor: float
    limiting_pair: tuple[str, str]
    projected_dt: float


class CollisionProximityMonitor:
    """URDF/SRDF collision proximity monitor using Pinocchio."""

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str | None = None,
        package_dirs: list[str] | None = None,
        distance_threshold: float = 0.02,
        dt: float = 0.10,
        safety_factor: float = 1.0,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.prediction_dt = dt
        self.safety_factor = safety_factor

        package_dirs = list(package_dirs) if package_dirs is not None else None
        self.model, self.collision_model, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs)
        self.data = self.model.createData()
        self.collision_model.addAllCollisionPairs()

        if srdf_path:
            pin.removeCollisionPairs(self.model, self.collision_model, srdf_path)

        self.collision_data = pin.GeometryData(self.collision_model)
        self._refresh_distance_requests()

        self.mimic_dict = self._parse_mimic_rules_from_urdf(urdf_path)

    @property
    def nq(self) -> int:
        """Return the number of joint positions in the model."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Return the number of joint velocities in the model."""
        return self.model.nv

    def add_box_obstacle(
        self,
        name: str,
        size_xyz: list[float],
        xyz: list[float],
        rpy: list[float] = [0.0, 0.0, 0.0],
    ) -> None:
        """Add a static box obstacle and pair it with all robot geometries."""
        sx, sy, sz = [float(v) for v in size_xyz]
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError("size_xyz must contain positive values")

        roll, pitch, yaw = [float(v) for v in rpy]
        rot = pin.rpy.rpyToMatrix(roll, pitch, yaw)
        placement = pin.SE3(rot, np.asarray(xyz, dtype=float))

        box = hppfcl.Box(sx, sy, sz)
        universe_joint = self.model.getJointId("universe")
        universe_frame = self.model.getFrameId("universe")

        geom_obj = pin.GeometryObject(name, universe_joint, universe_frame, placement, box)
        new_geom_id = self.collision_model.addGeometryObject(geom_obj)

        # Pair this obstacle against every other geometry.
        for gid in range(2, new_geom_id):
            self.collision_model.addCollisionPair(pin.CollisionPair(gid, new_geom_id))

        self.collision_data = pin.GeometryData(self.collision_model)
        self._refresh_distance_requests()

    def update_obstacle_pose(
        self,
        name: str,
        xyz: list[float],
        rpy: list[float] = [0.0, 0.0, 0.0],
    ) -> None:
        """Update pose of a previously added named obstacle."""
        geom_id = self.collision_model.getGeometryId(name)
        if geom_id < 0:
            raise ValueError(f"Obstacle '{name}' not found in collision model")

        roll, pitch, yaw = [float(v) for v in rpy]
        rot = pin.rpy.rpyToMatrix(roll, pitch, yaw)
        placement = pin.SE3(rot, np.asarray(xyz, dtype=float))
        self.collision_model.geometryObjects[geom_id].placement = placement

    def check_near_collision(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        joint_names: list[str],
        dt: float | None = None,
    ) -> CollisionInfo:
        """Return near-collision status for current and projected state.

        The monitor flags near-collision if either current or predicted minimum
        separation is below the threshold. Predicted state is built with first
        order integration: q_pred = integrate(q, dq * dt).
        """
        pos, vel = self._build_pos_and_vel(q, dq, joint_names)

        used_dt = float(self.prediction_dt if dt is None else dt)
        pos_pred = pin.integrate(self.model, pos, vel * used_dt)

        dist_now, pair_now = self._min_distance_and_pair(pos)
        dist_pred, pair_pred = self._min_distance_and_pair(pos_pred)

        threshold = self.distance_threshold * self.safety_factor
        near = (dist_now <= threshold) or (dist_pred <= threshold)
        limiting_pair = pair_now if dist_now <= dist_pred else pair_pred

        return CollisionInfo(
            near_collision=near,
            min_distance_now=dist_now,
            min_distance_predicted=dist_pred,
            threshold=self.distance_threshold,
            safety_factor=self.safety_factor,
            limiting_pair=limiting_pair,
            projected_dt=used_dt,
        )

    def _refresh_distance_requests(self) -> None:
        request = hppfcl.DistanceRequest(enable_nearest_points=True)
        for i in range(len(self.collision_data.distanceRequests)):
            self.collision_data.distanceRequests[i] = request

    def _min_distance_and_pair(self, q: np.ndarray) -> tuple[float, tuple[str, str]]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
        pin.computeDistances(self.model, self.data, self.collision_model, self.collision_data, q)

        min_dist = float("inf")
        min_pair = ("", "")
        for i, pair in enumerate(self.collision_model.collisionPairs):
            result = self.collision_data.distanceResults[i]
            d = float(result.min_distance)
            if d < min_dist:
                min_dist = d
                name_a = self.collision_model.geometryObjects[pair.first].name
                name_b = self.collision_model.geometryObjects[pair.second].name
                min_pair = (name_a, name_b)

        if not np.isfinite(min_dist):
            return float("inf"), ("", "")
        return min_dist, min_pair

    def _parse_mimic_rules_from_urdf(self, urdf_path: str) -> dict[str, dict[str, str | float]]:
        """Parse mimic multiplier and offset from URDF."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        mimic_rules = {}
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            mimic_joint_name = joint.attrib["name"]
            source_joint = mimic.attrib["joint"]
            multiplier = float(mimic.attrib.get("multiplier", 1.0))
            offset = float(mimic.attrib.get("offset", 0.0))
            mimic_rules[mimic_joint_name] = {
                "source_joint": source_joint,
                "multiplier": multiplier,
                "offset": offset,
            }
        return mimic_rules

    def _build_pos_and_vel(
        self, q: np.ndarray, dq: np.ndarray, joint_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build position and velocity from recorded joints and velocities to include mimic joints.

        Args:
            q: np.ndarray of joint positions observed from the robot
            dq: np.ndarray of joint velocities observed from the robot
            joint_names: list of joint names corresponding to the robot

        Returns:
            np.ndarrays of position and velocity of all joints

        """
        assert len(q) == len(dq) == len(joint_names)

        pos = np.zeros(self.nq)
        vel = np.zeros(self.nv)

        for i, n in enumerate(joint_names):
            iq = self.model.joints[self.model.getJointId(n)].idx_q
            iv = self.model.joints[self.model.getJointId(n)].idx_v
            pos[iq] = q[i]
            vel[iv] = dq[i]
        for mimic_joint, mim_info in self.mimic_dict.items():
            iq_m = self.model.joints[self.model.getJointId(mimic_joint)].idx_q
            iq_s = self.model.joints[self.model.getJointId(mim_info["source_joint"])].idx_q
            pos[iq_m] = mim_info["multiplier"] * pos[iq_s] + mim_info["offset"]

            iv_m = self.model.joints[self.model.getJointId(mimic_joint)].idx_v
            iv_s = self.model.joints[self.model.getJointId(mim_info["source_joint"])].idx_v
            vel[iv_m] = mim_info["multiplier"] * vel[iv_s]

        return pos, vel

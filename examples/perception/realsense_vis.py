"""Stream a RealSense camera through the Skillet perception pipeline and visualize it in Open3D.

Same perception + Open3D setup as `examples/kortex/kortex_demo.py`, but backed by `RealsenseEnv`
instead of the robot, so there is no arm, gripper or planner -- just the camera, SAM3 segmentation
and the reconstructed scene objects.

The camera is localized from the table AprilTag. Until the tag is detected the camera pose falls
back to the tag pose itself, so points may land outside the scene bounds; pass `--no-bounds` to
see the raw cloud anyway.

Examples:
    python examples/perception/realsense_vis.py
    python examples/perception/realsense_vis.py --reconstructor sam3_streaming --scene 1sponge_1plate_1can
    python examples/perception/realsense_vis.py --no-bounds --max-depth 1.5

Quit:
    Close the Open3D window, or Ctrl+C.

"""

import argparse

import numpy as np

from skillet.envs.realsense import RealsenseEnv
from skillet.perception.perception import SkilletPerception
from skillet.scene import Open3DVisualizer, load_scene

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--scene", type=str, default="2magnet_3wooden_3dots", help="`load_scene` name whose objects SAM3 segments"
)
parser.add_argument("--reconstructor", choices=["sam3", "sam3_streaming"], default="sam3", help="SAM3 backend")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--poll_rate_hz", type=int, default=10, help="Tick rate of the perception")
parser.add_argument("--max-depth", type=float, default=None, help="Drop depth beyond this many meters")
parser.add_argument("--no-bounds", action="store_true", help="Don't clip the point cloud to the scene bounds")
parser.add_argument("--segment", type=argparse.BooleanOptionalAction, default=True, help="Color points by mask")
parser.add_argument("--vis-perception", type=argparse.BooleanOptionalAction, default=True, help="Show the 2D masks")
parser.add_argument("--apriltag-id", type=int, default=1, help="ID of the table AprilTag")
parser.add_argument("--apriltag-family", type=str, default="tag36h11")
parser.add_argument("--apriltag-size", type=float, default=0.1, help="AprilTag edge length in meters")
parser.add_argument(
    "--apriltag-pose",
    type=float,
    nargs=7,
    default=[0.14, -0.01, 0.0, 0.0, 0.0, 0.7071068, -0.7071068],
    metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    help="AprilTag pose in the world frame (xyz + quaternion xyzw)",
)
args_cli = parser.parse_args()


def main() -> None:
    """Stream the RealSense through perception until the Open3D window closes."""
    scene = load_scene(args_cli.scene)
    domain = "sponge" if "sponge" in args_cli.scene else "blocks"
    if args_cli.no_bounds:
        scene.bounds = None

    env = RealsenseEnv(
        width=args_cli.width,
        height=args_cli.height,
        fps=args_cli.fps,
        apriltag_pose=np.asarray(args_cli.apriltag_pose),
        apriltag_size_m=args_cli.apriltag_size,
        apriltag_id=args_cli.apriltag_id,
        apriltag_fam=args_cli.apriltag_family,
    )

    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=env.obs_spec,
        reconstructor=args_cli.reconstructor,
        poll_rate_hz=args_cli.poll_rate_hz,
        device="cuda",
        max_depth_m=args_cli.max_depth,
        vis_perception=args_cli.vis_perception,
        domain=domain,
    )
    visualizer = Open3DVisualizer(scene, env, window_name="RealSense Scene", width=1280, height=720)
    perception.set_visualizer(visualizer, segment_point_cloud=args_cli.segment)
    perception.run_thread()

    try:
        # Blocks on the GUI event loop until the window is closed
        visualizer.run()
    except KeyboardInterrupt:
        pass
    finally:
        perception.stop()
        env.close()


if __name__ == "__main__":
    main()

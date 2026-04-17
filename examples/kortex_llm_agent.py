"""Run the LLM planning agent on Kortex hardware (no replan on failure)."""

import argparse
import copy
import time
from typing import TYPE_CHECKING

import torch

from skillet.agents.llm_agent import LLMPlanningAgent
from skillet.agents.llm_planner import LLMPlanner, OpenAIClient, GeminiLLMClient
from skillet.core import ObservationSpec
from skillet.core.env import BatchToSingleWrapper
from skillet.envs import RealsenseEnv, SkilletEnv
from skillet.perception import SkilletPerception
from skillet.scene import EMPTY_SCENE, Open3DVisualizer
from skillet.skill import PickBlockSkill, PickSkill, PlaceBlock2Skill, PlaceSkill

if TYPE_CHECKING:
    from skillet.envs.specs import RGBD_Obs

parser = argparse.ArgumentParser(description="LLM planning agent on Kortex hardware.")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
parser.add_argument("--task", type=str, default="Kortex-Gen3Lite-v0", help="Kortex environment.")
parser.add_argument("--poll_rate_hz", type=int, default=1, help="Perception poll rate.")
parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model for planning.")
parser.add_argument("--goal", type=str, default="Stack red_block on blue_block", help="Task goal.")
parser.add_argument(
    "--realsense_env", action=argparse.BooleanOptionalAction, default=False,
    help="Use standalone RealSense camera (no robot).",
)
parser.add_argument(
    "--vlm_cache_path", type=str, default=None,
    help=(
        "Pickle path for cached Gemini VLM output (image + bboxes + goal atoms). "
        "Loaded if the file exists, skipping the API call; otherwise written "
        "after the first VLM call. Leave unset to always call the API."
    ),
)
parser.add_argument(
    "--vlm_predicates", action=argparse.BooleanOptionalAction, default=False,
    help=(
        "Derive scene predicates for the planner from the workspace image via "
        "the planning VLM instead of from geometric grounding."
    ),
)


def main() -> None:
    args = parser.parse_args()

    scene = copy.deepcopy(EMPTY_SCENE)

    # --- 1. Create environment ---
    if args.realsense_env:
        env = RealsenseEnv(apriltag_size_m=0.1, apriltag_id=3)
    else:
        env_cfg = {
            "robot_ip": args.robot_ip,
            "device": args.device,
            "num_envs": 1,
        }
        from skillet_tasks.kortex_tasks.factory import create_kortex_env

        env = create_kortex_env(args.task, env_cfg)
        env = SkilletEnv(env)
        env = BatchToSingleWrapper(env)
        env.reset()

    obs_spec_name = "rgb-d" if args.realsense_env else "rgbd-gripper"
    rgbd_spec: ObservationSpec[RGBD_Obs] = env.coerce_obs_spec(obs_spec_name)

    # --- 2. Start perception (VLM + SAM → scene with 3D poses) ---
    perception = SkilletPerception(
        env=env,
        scene=scene,
        obs_spec=rgbd_spec,
        reconstructor="sam",
        poll_rate_hz=args.poll_rate_hz,
        device=args.device,
        vlm_cache_path=args.vlm_cache_path,
    )
    perception.task_instruction = args.goal
    perception.build_scene = True

    visualizer = Open3DVisualizer(scene, env)
    perception.set_visualizer(visualizer, segment_point_cloud=True)
    perception.run_thread()
    visualizer.run_thread()

    print(f"[KortexLLM] Waiting for perception to build scene...")
    timeout = 60
    start = time.time()
    while not scene.contains_objects and (time.time() - start) < timeout:
        time.sleep(0.5)

    if not scene.contains_objects:
        print("[KortexLLM] ERROR: Perception timed out — no objects detected.")
        perception.stop()
        env.close()
        return

    print(f"[KortexLLM] Scene built:\n{scene}")

    # --- 3. Create LLM planner ---
    planner = LLMPlanner(
        client=OpenAIClient(model=args.model),
        use_vlm_predicates=args.vlm_predicates,
    )

    if args.realsense_env:
        # RealSense only: no robot, just test perception + planning
        print(f"[KortexLLM] Goal: {args.goal}")
        print(f"[KortexLLM] Model: {args.model}")
        print("[KortexLLM] RealSense mode — planning only (no robot execution).")

        success, plan = planner.plan(perception.scene, args.goal)
        print(f"\n=== LLM Plan (success={success}) ===")
        if plan:
            for i, action in enumerate(plan.actions):
                print(f"  Step {i}: {action.action}({action.parameters})")
        else:
            print("  No plan generated.")

        input("\n[KortexLLM] Press Enter to stop...")
        perception.stop()
        env.close()
        return

    # --- 4. Create skills ---
    from skillet.policy import TwistPidPosePolicy

    arm_policy = TwistPidPosePolicy(
        env.batched_env.obs_spec_twist_tcp,
        env.batched_env.action_spec_twist_tcp,
    )
    skill_length = int(1e9)

    pick_skill = PickSkill(
        reach_policy=arm_policy, gripper_policy=None,
        lift_height=0.23, length=skill_length,
    )
    place_skill = PlaceSkill(
        reach_policy=arm_policy, gripper_policy=None,
        lift_height=0.23, length=skill_length,
    )

    pick_block_skill = PickBlockSkill(
        perception.scene, pick_skill, vis_target_pos=visualizer.set_target_pos,
    )
    place_block_skill = PlaceBlock2Skill(
        perception.scene, place_skill, vis_target_pos=visualizer.set_target_pos,
    )

    action_map = {
        "pick_block": pick_block_skill,
        "place_block": place_block_skill,
    }

    # --- 5. Create agent and run ---
    agent = LLMPlanningAgent(
        scene=perception.scene,
        planner=planner,
        action_to_skill_map=action_map,
        goal=args.goal,
        perception=perception,
    )

    print(f"[KortexLLM] Goal: {args.goal}")
    print(f"[KortexLLM] Model: {args.model}")
    input("[KortexLLM] Press Enter to start execution...")

    with torch.inference_mode():
        agent.execute(env)

    print("[KortexLLM] Done.")
    perception.stop()
    env.close()


if __name__ == "__main__":
    main()

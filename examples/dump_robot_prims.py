"""
dump_robot_prims.py

Loads Given USD into an Isaac Sim stage and prints every prim path.
Use this to find the exact paths.

Run with Isaac Sim's Python:
    python examples/dump_robot_prims.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print all prim paths in Kinova_Gen3.usd.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd

from kinova_tasks.assets.utils import KINOVA_ASSET_DIR

USD_PATH = f"{KINOVA_ASSET_DIR}/robots/kinova/Kinova_Gen3.usd"


def main():
    stage: Usd.Stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        print(f"[ERROR] Could not open {USD_PATH}")
        return

    print(f"\nAll prims in  {USD_PATH}\n" + "-" * 60)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        print(f"  {path}")

    print("-" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()

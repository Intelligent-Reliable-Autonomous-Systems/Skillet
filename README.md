# Robot Skills

A framework for robot task and motion planning with skills/options developed for the Intelligent and Reliable Autonomous Systems (IRAS) Lab (Dr. Sandhya Saisubramanian) at Oregon State University. 

Primary Developers: Jeff Jewett (jewettje@oregonstate.edu) and Will Solow (soloww@oregonstate.edu)

## Installation

1. Create a conda environment: `conda create -n skills python=3.11`
2. Activate conda environment: `conda activate skills`
3. Install requirements via pip: `pip install numpy torch gymnasium roslibpy`

### IsaacSim/IsaacLab integration
See [IsaacLab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for additional information

1. Install IsaacSim `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
2. Verify Installation `isaacsim` and accept EULA
3. Clone IsaacLab repo: Navigate to parent directory `cd ..` and then `git clone https://github.com/isaac-sim/IsaacLab.git`
4. Install IsaacLab: `cd IsaacLab`, `./isaaclab.sh --install`
5. Navigate back to Robot-Skills repository `cd ../Robot-Skills`

To run experiment with dummy task policy and low level policy: `python3 examples/ros2_dummy.py --num_envs 4 --task Kinova-Reach-Direct-v0`

## ROS2 Integration
See [ROS2 Installation](https://docs.ros.org/en/jazzy/Installation.html) to install ROS2. Be sure to install on system python (not venv/conda)

1. Navigate to parent directory and clone `https://github.com/Intelligent-Reliable-Autonomous-Systems/IRAS-Kinova`
2. Follow installation instructions in `IRAS-Kinova/README.md`
3. Navigate back to `Robot-Skills`

To run:
1. Open new terminal. Navigate to IRAS-Kinova. Ensure system python is active (no venv/conda).
   - Source ROS2 system installation: `source /opt/ros/jazzy/setup.bash`
   - Source IRAS-Kinova ROS2 overlay: `source install/setup.bash`
   - Laucn ROSBridge Node: `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`
2. Navigate back to Robot-Skills in another terminal. Ensure virtual env is active: `conda activate skills`
3. Run dummy task policy with ROS2/RViz: `python3 examples/ros2_dummy.py --num_envs 1 --task ROS2-Reach-Kinova-v0 --ros2_ws <absolute-path-to-IRAS/Kinova>`

## Current considerations
1. Should we assume that we are operating on GPU or CPU (torch or NumPy)?
2. Should we assume that operations are batched? 
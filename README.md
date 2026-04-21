# Robot Skills

A framework for robot task and motion planning with skills/options developed for the Intelligent and Reliable Autonomous Systems (IRAS) Lab (Dr. Sandhya Saisubramanian) at Oregon State University. 

Primary Developers: Jeff Jewett (jewettje@oregonstate.edu) and Will Solow (soloww@oregonstate.edu)

## Installation

1. Create a conda environment: `conda create -n skillet python=3.11`
2. Activate conda environment: `conda activate skillet`
3. Install requirements via pip: `pip install -e .`

### IsaacSim/IsaacLab integration
1. pip install -e ".[isaac]"

### Mujoco Integration
2. pip install -e ".[mujoco]"

### Perception installation
1. Make sure to activate conda environment: `conda activate skills`
2. Install `open3d`: `conda install -c conda-forge open3d`
3. Install perception python packages: `pip install -e ".[perception]"`
4. You must first request access for the model weights on the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3) and then, once approved, download the sam3.pt file. Place the downloaded sam3.pt file at `data/models/sam3.pt`.
5. Clone the SAM3 repository: `git clone https://github.com/facebookresearch/sam3.git third_party/sam3`
6. `cd third_party/sam3 && pip install -e .`

## Kortex Integration
Sometimes ROS2 is a pain. To run the robot through the Kortex API instead of ROS2, follow these directions:
1. `python3 -m pip install skillet/envs/kortex/kortex_api-2.6.0.post3-py3-none-any.whl`
2. `pip install -e ".[kortex]"

You should now be all set to run an experiment with `--env_id Kortex-Gen3Lite-v0`

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
3. Run a Pick Skill with ROS2/RViz: `python3 examples/ros2_pick.py --num_envs 1 --task ROS2-Gen3-v0 --ros2_ws <absolute-path-to-IRAS/Kinova>`

### Hardware Experiment with ROS
Launch Gen3
```bash
ros2 launch gen3_py gen3.launch.py robot_ip:=192.168.8.10 use_fake_hardware:=false gripper:=robotiq_2f_85
```

OR 

Launch Gen3Lite
```bash
ros2 launch gen3lite_py gen3_lite.launch.py robot_ip:=192.168.1.10 use_fake_hardware:=false
```

Launch rosbridge
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```


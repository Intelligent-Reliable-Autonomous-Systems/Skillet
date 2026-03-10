# Robot Skills

A framework for robot task and motion planning with skills/options developed for the Intelligent and Reliable Autonomous Systems (IRAS) Lab (Dr. Sandhya Saisubramanian) at Oregon State University. 

Primary Developers: Jeff Jewett (jewettje@oregonstate.edu) and Will Solow (soloww@oregonstate.edu)

## Installation

1. Create a conda environment: `conda create -n skills python=3.11`
2. Activate conda environment: `conda activate skills`
3. Install requirements via pip: `pip install -e .`

### IsaacSim/IsaacLab integration
See [IsaacLab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for additional information

1. Install IsaacSim `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
2. Verify Installation `isaacsim` and accept EULA
3. Clone IsaacLab repo: Navigate to parent directory `cd ..` and then `git clone https://github.com/isaac-sim/IsaacLab.git`
4. Install IsaacLab: `cd IsaacLab`, `./isaaclab.sh --install`
5. Navigate back to Robot-Skills repository `cd ../Robot-Skills`

### Perception installation

1. Make sure to activate conda environment: `conda activate skills`
2. Install `open3d`: `conda install -c conda-forge open3d`
3. Install perception python packages: `pip install -e ".[perception]"`
4. Unlike other Ultralytics models, SAM 3 weights (sam3.pt) are not automatically downloaded. You must first request access for the model weights on the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3) and then, once approved, download the sam3.pt file. Place the downloaded sam3.pt file at `data/models/sam3.pt`.


Perception relies on some additional modules.

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

### Hardware Experiment 1
Launch Gen3
```bash
ros2 launch gen3_py gen3.launch.py robot_ip:=192.168.8.10 use_fake_hardware:=false gripper:=robotiq_2f_85 use_table_camera:=true
```

OR 

Launch Gen3Lite
```bash
ros2 launch gen3lite_py gen3_lite.launch.py robot_ip:=192.168.1.10 use_fake_hardware:=false use_table_camera:=true
```

Launch rosbridge
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Put arm into a good initial position
```bash
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
    joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
    points: [
        { positions: [0, 0.523599, 0, 1.5708, 0, 1.0, 0], time_from_start: { sec: 5 } },
    ]
    }" -1
```

    ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
    joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6],
    points: [
        { positions: [0.2, -0.18, 2.16, -1.57, -0.6, -1.34], time_from_start: { sec: 5 } },
    ]
    }" -1


ros2 action send_goal /move_action moveit_msgs/action/MoveGroup "{request: {group_name: 'arm', start_state: {is_diff: true}, goal_constraints: [{name: 'pose_goal', position_constraints: [{header: {frame_id: 'base_link'}, link_name: 'tool_frame', constraint_region: {primitives: [{type: 2, dimensions: [0.02, 0.02, 0.02]}], primitive_poses: [{position: {x: 0.2, y: -0.257, z: 0.2300}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}]}, weight: 1.0}], orientation_constraints: [{header: {frame_id: 'base_link'}, link_name: 'tool_frame', orientation: {x: 0.9969, y: 0.0654, z: 0.0438, w: -0.0052}, absolute_x_axis_tolerance: 0.1, absolute_y_axis_tolerance: 0.1, absolute_z_axis_tolerance: 0.1, weight: 1.0}]}], num_planning_attempts: 10, allowed_planning_time: 5.0, max_velocity_scaling_factor: 0.2, max_acceleration_scaling_factor: 0.2}, planning_options: {plan_only: false, replan: true, replan_attempts: 3}}"
    
# Robot Skills

A framework for robot task and motion planning with skills/options developed for the Intelligent and Reliable Autonomous Systems (IRAS) Lab (Dr. Sandhya Saisubramanian) at Oregon State University. 

Primary Developers: Jeff Jewett (jewettje@oregonstate.edu) and Will Solow (soloww@oregonstate.edu)

## Installation

1. Create a conda environment: `conda create -n skills python=3.11`
2. Activate conda environment: `conda activate skills`
3. Install requirements via pip: `pip install numpy torch gymnasium`

### IsaacSim/IsaacLab integration
1. Install IsaacSim `pip install isaacsim`
2. Verify Installation `isaacsim` and accept EULA
3. Clone IsaacLab repo: Navigate to parent directory `cd ..` and then `git clone https://github.com/isaac-sim/IsaacLab.git`
4. Install IsaacLab: `cd IsaacLab`, `./isaaclab.sh --install`
5. Navigate back to Robot-Skills repository `cd ../Robot-Skills`

## Use

## Current considerations
1. Should we assume that we are operating on GPU or CPU (torch or NumPy)?
2. Should we assume that operations are batched? 
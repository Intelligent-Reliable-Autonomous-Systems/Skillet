# Robot Skills

A framework for robot task and motion planning with skills/options developed for the Intelligent and Reliable Autonomous Systems (IRAS) Lab (Dr. Sandhya Saisubramanian) at Oregon State University. 

Primary Developers: Jeff Jewett (jewettje@oregonstate.edu) and Will Solow (soloww@oregonstate.edu)

## Installation

1. Create a conda environment: `conda create -n skills python=3.12`
2. Activate conda environment: `conda activate skills`
3. Install requirements via pip: `pip install numpy torch`

## Use

## Current considerations
1. Should we assume that we are operating on GPU or CPU (torch or NumPy)?
2. Should we assume that operations are batched? 
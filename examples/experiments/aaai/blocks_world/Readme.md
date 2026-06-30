# Blocks Experiments Readme

This folder contains 3 files:
1. blocks_active.py - File for running active learning experiments
2. blocks_eval.py - File for running experiments on evaluation tasks
3. blocks_random.py - File for collecting random action data

Every one of these experiment files is set up with a logger and video recorder. A video will be saved to the 

## blocks_active.py
This file uses a ActiveLearningAgent() class to select actions. The active learning agent class
must contain a _learning_agent which has two functions (see tamp.py:ActiveLearningAgent)
1. sample_action() - this must return an AbstractAction and a UPAction (InstantaneousAction)
2. update() - given the executed action, the success of the action, the new state, the agent should update its state representation 

sample_action() and update() should be implemented by Jeff given the CSAM, OLAM, Jeff Method baselines.

After those are completed, it should be ready to go

## blocks_eval.py
This file uses a PlanningAgent() class to plan over a domain specified by --domain_file and a task specified by --eval_dir.

eval_dir/ should contain three files: 
1. g_nl.txt - a natural language goal 
2. g_pddl.txt - a JSON parsable PDDL goal in the form: 
    [{"predicate": "on", "args": ["red_block", "table_0"]},
    {"predicate": "on", "args": ["pink_block", "red_block"]}]
3. s_init.txt - a file describing the initial state, usually in a PDDL-like format:
    on(yellow block, table)
    on(red block, table) ; immovable
    on(green block, yellow block)

Using the --vlm flag will force the PDDL goal to be constructed by the VLM, otherwise it will be parsed from g_pddl.txt. 

## blocks_random.py 
Given a domain, a RandomTampAgent() will select random actions that are viable in the current percieved state.


sudo ip addr add 192.168.1.100/24 dev enx6c6e072d4846
sudo ip link set enx6c6e072d4846 up
sudo ip route add 192.168.1.0/24 dev enx6c6e072d4846

sam3_server --serve sam3
# Current considerations for task level policies

The task policy should handle the sequencing of skills. It would be helpful 
if this task policy could be written up to handle an RL policy as well. 

Ideally we want a configuration that defines:

1) The type of task policy to use
2) The skills that the task policy assumes to have access to
3) The formal definition of each skill (initiation, terminatation, low level controller, success, etc)
4) Which simulator we are interfacing with. 


Current considerations: where should the environment be placed within the controller so that we can both execute low level control
and high level planning
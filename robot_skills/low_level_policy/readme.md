# Current considerations for low level policies

I think it makes most sense to have these low level policy classes
simply take the current observation and output the next action.

They should not be responsible for if any skill is in fact "done." This should
be handled by the skills controller classes which interface between the 
task policy and the low level policy. 

However, we might want to make the low level policies interface with the simulator of choice
This might be the easiest way to pass the "done" or "next obs" up the chain to the skill controller
and task policy
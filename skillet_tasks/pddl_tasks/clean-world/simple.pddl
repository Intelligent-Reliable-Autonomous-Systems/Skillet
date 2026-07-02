(:action hover-over
    :parameters (?grasped - item ?target - surface)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (supportable ?target)
    )
    :effect (
    )
)



(:action clean-plate
    :parameters (?grasped - sponge ?target - plate)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (dirty ?target)
    )
    :effect (and
        (gripper-full)
        (grasping ?grasped)
        (not (dirty ?target))
    )
)
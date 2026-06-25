(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface movable - object
        table - surface
        spill ketchup sponge plate - movable
    )
    (:predicates
        ; static predicates
        (deformable ?m - movable) ; kinematic attribute - movable can be squeezed
        ; dynamic predicates
        (supportable ?m - object) ; if this object can support something
        (wet ?b - object) ; material attribute
        (dirty ?b - object) ; material attribute
        (gripper-lifted) ; the gripper is lifted in the air
        (blue ?b - sponge)
        (yellow ?b - sponge) ; colors for the sponge
        (grasping ?b - movable) ; the gripper is closed around movable b
        (on ?b - movable ?s - object) ; movable b is on surface s
        ; pseudo-derived predicates
        (gripper-full) ; the gripper is occupied <-> exists ?b. (grasping ?b)
    )
(:action pick-sponge
    :parameters (?target - sponge ?support - object)
    :precondition (and
        (on ?target ?support)
        (not (gripper-full))
        (supportable ?support)
    )
    :effect (and
        (gripper-full)
        (grasping ?target)
        (gripper-lifted)
        (not (on ?target ?support))
    )
)
(:action place-sponge
    :parameters (?grasped - sponge ?target - object)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (supportable ?target)
    )
    :effect (and
        (on ?grasped ?target)
        (not (gripper-full))
        (not (grasping ?grasped))
        (gripper-lifted)
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
(:action squeeze-sponge
    :parameters (?grasped - sponge)
    :precondition (and
        (grasping ?grasped)
        (gripper-full)
        (deformable ?grasped)
    )
    :effect (and
        (not (wet ?grasped))
    )
)
)
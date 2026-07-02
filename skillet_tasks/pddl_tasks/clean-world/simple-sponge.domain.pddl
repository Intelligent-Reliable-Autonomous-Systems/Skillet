(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface item - object
        plate location bin - surface
        sponge can - item
        
    )
    (:predicates


        ; static predicates
        (deformable ?m - item) ; kinematic attribute - item can be squeezed
        (supportable ?m - object) ; if this object can support something
        (blue ?b - sponge)
        (yellow ?b - sponge) ; colors for the sponge

        ; dynamic predicates
        (on ?b - item ?s - surface) ; item b is on surface s
        (grasping ?b - item) ; the gripper is closed around movable b


        (wet ?b - object) ; material attribute
        (dirty ?b - object) ; material attribute
        

        ; pseudo-derived predicates
        (gripper-lifted) ; the gripper is lifted in the air
        (gripper-full) ; the gripper is occupied <-> exists ?b. (grasping ?b)
    )


    
(:action pick
    :parameters (?target - item ?support - surface)
    :precondition (and
        (not (gripper-full))
        (on ?target ?support)
    )
    :effect (and
        (gripper-full)
        (grasping ?target)
        (gripper-lifted)
        (not (on ?target ?support))
    )
)

(:action place
    :parameters (?grasped - item ?target - surface)
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




(:action squeeze
    :parameters (?grasped - item)
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
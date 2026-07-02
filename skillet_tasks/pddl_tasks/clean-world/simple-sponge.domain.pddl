(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions :universal-preconditions)
    (:types
        surface item - object
        table plate location bin - surface
        sponge can spill - item
        
    )
    (:predicates
        ; static predicates
        (deformable ?m - item) ; kinematic attribute - item can be squeezed
        (graspable ?m - item) ; if this object can be grasped
        (supportable ?m - surface) ; if this surface can support something
        (hoverable ?m - surface) ; if this surface can be hovered over
        (blue ?b - sponge)
        (yellow ?b - sponge) ; colors for the sponge

        (wet ?b - item) ; material attribute
        (dirty ?b - surface) ; material attribute

        ; dynamic predicates
        (on ?b - item ?s - surface) ; item b is on surface s
        (grasping ?b - item) ; the gripper is closed around movable b
        (obstructed ?b - surface) ; the surface is obstructed
        (hover ?g - item ?s - surface); the surface being hovered over

        ; pseudo-derived predicates
        (gripper-full) ; the gripper is occupied <-> exists ?b. (grasping ?b)
    )

(:action pick
    :parameters (?target - item ?support - surface)
    :precondition (and
        (not (gripper-full))
        (on ?target ?support)
        (obstructed ?support)
        (graspable ?target)
    )
    :effect (and
        (gripper-full)
        (grasping ?target)
        (not (on ?target ?support))
        (not (obstructed ?support))

    )
)

(:action place
    :parameters (?grasped - item ?target - surface)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (supportable ?target)
        (not (obstructed ?target))
    )
    :effect (and
        (on ?grasped ?target)
        (not (gripper-full))
        (not (grasping ?grasped))
        (obstructed ?target)
        (forall (?s - surface) (and (not (hover ?grasped ?s))))
    )
)

(:action hover-over
    :parameters (?grasped - item ?target - surface)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (hoverable ?target)
    )
    :effect (and
        (hover ?grasped ?target)
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

(:action wipe
    :parameters (?grasped - item ?target - surface)
    :precondition (and
        (hover ?grasped ?target)
        (gripper-full)
        (grasping ?grasped)
        (supportable ?target)
        (deformable ?grasped)
    )
    :effect (and
        (hover ?grasped ?target)
        (not (dirty ?target))
    )
)
)
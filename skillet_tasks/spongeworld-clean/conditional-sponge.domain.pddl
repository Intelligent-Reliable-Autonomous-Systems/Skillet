(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions :universal-preconditions)
    (:types
        surface item - object
        table plate target - surface
        sponge can spill - item

    )
    (:predicates
        ; static predicates
        (graspable ?m - item) ; if this object can be grasped
        (deformable ?m - item) ; if the item can be deformed
        (supportable ?m - surface) ; if this surface can support something
        (wipeable ?m - item) ; if this object can be wiped
        (blue ?b - sponge)
        (yellow ?b - sponge) ; colors for the sponge

        (wet ?b - surface) ; material attribute
        (dirty ?b - surface) ; material attribute

        ; dynamic predicates
        (on ?b - object ?s - surface) ; item b is on surface s
        (grasping ?b - item) ; the gripper is closed around movable b
        (obstructed ?b - surface) ; the surface is obstructed
        (contains ?b - item ?s - spill) ; the item contains liquid

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
        (hover ?grasped ?target)
    )
    :effect (and
        (on ?grasped ?target)
        (not (gripper-full))
        (not (grasping ?grasped))
        (obstructed ?target)
        (not (hover ?grasped ?target))
        (when
            (wet ?grasped)
            (wet ?target) ; placing a wet item on a surface makes the surface wet
        )
    )
)


(:action wipe
    :parameters (?grasped - item ?mess - item ?target - surface)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (supportable ?target)
        (on ?mess ?target)
        (wipeable ?mess)
        (deformable ?grasped)
    )
    :effect (and
        (not (dirty ?target))
        (not (on ?mess ?target))
        (forall (?can - can ?can-spill - spill)
            (when
                (and (on ?can ?target) (contains ?can ?can-spill))
                (on ?can-spill ?target) ; cans that are supported by the target surface spill their water onto the surface
            )
        )
        (forall (?can - can ?can-spill - spill)
            (when
                (and (on ?can ?target) (contains ?can ?can-spill))
                (not (contains ?can ?can-spill)) ; cans that are supported by the target surface spill their water onto the surface
            )
        )
    )
)
)
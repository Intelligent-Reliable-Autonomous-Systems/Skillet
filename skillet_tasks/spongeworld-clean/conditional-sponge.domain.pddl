(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions :universal-preconditions)
    (:types
        surface item - object
        table plate target bin - surface
        sponge can spill - item

    )
    (:predicates
        ; static predicates
        (graspable ?m - item) ; if this object can be grasped
        (deformable ?m - item) ; if the item can be deformed
        (supportable ?m - surface) ; if this surface can support something
        (wipeable ?m - item) ; if this object can be wiped

        (wet ?b - item) ; material attribute
        (dirty ?b - surface) ; material attribute

        ; dynamic predicates
        (on ?b - object ?s - surface) ; item b is on surface s
        (grasping ?b - item) ; the gripper is closed around movable b
        (obstructed ?b - surface) ; the surface is obstructed

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

        ; wipe with wet sponge -> water on plate
        (forall (?water_spill - spill)
            (when
                (and (wet ?grasped) (not (on ?water_spill ?target))) ; replace marker_spill with water_spill
                (on ?water_spill ?target)
            )
        )
        ; knock off cans onto table
        (forall (?soda_can - can)
            (when
                (on ?soda_can ?target)
                (not (on ?soda_can ?target))
            )
        )
        (forall (?soda_can - can ?tab - table)
            (when
                (on ?soda_can ?target)
                (on ?soda_can ?tab)
            )
        )

    )
)
)
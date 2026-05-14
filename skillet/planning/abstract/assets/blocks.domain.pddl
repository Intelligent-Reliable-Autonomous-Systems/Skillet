(define (domain blocks)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface - object
        table target block - surface
    )

    (:predicates
        (small ?s - surface) ; the surface can only fit one object

        (gripper-full)
        (clear ?s - surface)
        (on ?b - block ?s - surface)
        (grasping ?b - block)
    )

(:action pick-block
    :parameters (?b - block ?s - surface)
    :precondition (and
        (clear ?b)
        (on ?b ?s)
        (not (gripper-full))
    )
    :effect (and
        (gripper-full)
        (grasping ?b)
        (not (on ?b ?s))
        (not (clear ?b))
        (when
            (small ?s)
            (clear ?s)
        )
    )
)


(:action place-block
    :parameters (?b - block ?s - surface)
    :precondition (and
        (gripper-full)
        (grasping ?b)
        (clear ?s)
    )
    :effect (and
        (not (gripper-full))
        (not (grasping ?b))
        (on ?b ?s)
        (clear ?b)
        (when
            (small ?s)
            (not (clear ?s))
        )
    )
)

(:action drag-block
    :parameters (?b - block ?s - surface)
    :precondition (and
        (clear ?b)
        (on ?b ?s)
        (not (gripper-full))
        (not (small ?s))
    )
    :effect (and
        (gripper-full)
        (grasping ?b)
        (not (on ?b ?s))
        (not (clear ?b))
    )
)

)
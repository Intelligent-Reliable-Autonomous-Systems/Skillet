(define (domain blocks)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface - object
        table target block - surface
    )

    (:predicates
        (small ?s - surface) ; the surface can only fit one object

        (handempty)
        (clear ?s - surface)
        (on ?b - block ?s - surface)
        (holding ?b - block)
    )

(:action pick-block
    :parameters (?b - block ?s - surface)
    :precondition (and
        (clear ?b)
        (on ?b ?s)
        (handempty)
    )
    :effect (and
        (not (handempty))
        (holding ?b)
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
        (not (handempty))
        (holding ?b)
        (clear ?s)
    )
    :effect (and
        (handempty)
        (not (holding ?b))
        (on ?b ?s)
        (clear ?b)
        (when
            (small ?s)
            (not (clear ?s))
        )
    )
)

)
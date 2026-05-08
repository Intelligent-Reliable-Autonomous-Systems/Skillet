(define (domain inspection)
    (:requirements :typing :negative-preconditions)
    (:types
        block location - object
        table0 destination - location
        platform0 discard0 - destination
    )

    (:predicates
        (on ?b - block ?l - location)          ; block b is resting on location l
        (holding ?b - block)                   ; gripper is holding block b
        (gripper-empty)                        ; gripper holds nothing
        (gripper-above ?b - block)             ; wrist camera is positioned above block b
        (inspected ?b - block)                 ; block has been classified by DefectClassifier
        (defective ?b - block)                 ; ground-truth label: defective (set in initial state)
        (non-defective ?b - block)             ; ground-truth label: non-defective (set in initial state)
    )

    ; Move the wrist camera to an inspection pose above block ?b.
    ; Precondition requires the block to still be on the table and
    ; the gripper to be free.
    (:action approach-block
        :parameters (?b - block ?t - table0)
        :precondition (and
            (on ?b ?t)
            (gripper-empty)
        )
        :effect (gripper-above ?b)
    )

    ; Classify block ?b using the wrist-mounted camera.
    ; The arm must already be positioned above the block.
    (:action inspect-for-defects
        :parameters (?b - block)
        :precondition (gripper-above ?b)
        :effect (inspected ?b)
    )

    ; Grasp the block directly below the gripper.
    (:action pick
        :parameters (?b - block ?t - table0)
        :precondition (and
            (on ?b ?t)
            (gripper-above ?b)
            (gripper-empty)
        )
        :effect (and
            (not (on ?b ?t))
            (not (gripper-empty))
            (not (gripper-above ?b))
            (holding ?b)
        )
    )

    ; Place the held block at any destination (platform or discard region).
    ; The planner picks the destination that satisfies the goal.
    (:action place
        :parameters (?b - block ?dest - destination)
        :precondition (and
            (holding ?b)
            (inspected ?b)
        )
        :effect (and
            (not (holding ?b))
            (gripper-empty)
            (on ?b ?dest)
        )
    )
)
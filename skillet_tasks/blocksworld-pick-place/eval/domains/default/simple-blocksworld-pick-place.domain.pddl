(define (domain simple-blocks-pick-place-v1)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface location - object
        table target block - surface
    )

    (:predicates
        ; static predicates
        (loc-above ?l1 ?l2 - location) ; l1 is adjacent above l2

        (wooden ?b - surface) ; material attribute - latent property: inert
        (plastic ?b - surface) ; material attribute - latent property: magnetic

        ; dynamic predicates
        (at-loc ?s - surface ?l - location) ; block or target is at location b

        (grasping ?b - block) ; the gripper is closed around block b

        (on ?b - surface ?s - surface) ; block b is on surface s

        ; safety predicates (not currently used)
        (not-grid-aligned ?b - surface)
            ; block is not aligned to the grid.
            ; If an action knocks a block out of the grid, it fails
            ; e.g. if a tower gets knocked over

        ; pseudo-derived predicates
        (gripper-full) ; the gripper is grasping something ∃ [?b - block] [grasping ?b]
        (two-held) ; at least two blocks are held in the air (including ?grasped)
        (three-held) ; at least three blocks are held in the air (including ?grasped)
        (obstructed-above ?l - location) ; there is something occupying the location above ∃ [?l2 - location] when [loc-above ?l2 ?l1] [occupied ?l2]
    )


;;; pick block ?target from surface ?support at corresponding locations ?targetloc and ?supportloc
;;; the gripper must be empty
;;; the physical interpretation of this action is that the gripper will raise up,
;;;     move to ?target's location, lower down, close the gripper around it, then lift the block up
(:action pick-block
    :parameters (?target - block ?support - surface ?targetloc ?supportloc - location)
    :precondition (and
        (at-loc ?target ?targetloc)
        (at-loc ?support ?supportloc)
        (loc-above ?targetloc ?supportloc)
        (on ?target ?support)

        (not (obstructed-above ?targetloc))
        (not (gripper-full))
    )
    :effect (and
        ;; gripper
        (gripper-full)
        (grasping ?target)
        ;; target block
        (not (at-loc ?target ?targetloc))
        ;; support
        (not (obstructed-above ?supportloc)) ; the support location is now clear

        (not (on ?target ?support)) ; if either target or support is non-magnetic, they are separated
    )
)

;;; place the grasped block ?grasped onto a free location ?freeloc above surface ?target at location ?targetloc
;;; the physical interpretation of this action is that the arm will move over
;;;     the target location ?freeloc, descend to the ?target surface, and release
;;; if there are blocks hanging under ?grasped, they will be placed in order
;;; precondition: it needs to already be grasping and lifted in the air
(:action place-block
    :parameters (?grasped - block ?target - surface ?freeloc ?targetloc - location)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (not (obstructed-above ?targetloc))
        (not (obstructed-above ?freeloc))
        (at-loc ?target ?targetloc)
        (loc-above ?freeloc ?targetloc)
    )
    :effect (and
        ;; gripper
        (not (grasping ?grasped))
        (not (gripper-full))
        (obstructed-above ?targetloc)

        (on ?grasped ?target)
        (at-loc ?grasped ?freeloc)
    )
)

)
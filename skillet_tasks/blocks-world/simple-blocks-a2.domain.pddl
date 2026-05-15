(define (domain blocks-location-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface location - object
        table target block - surface
    )

    (:predicates
        ; static predicates
        (loc-north-of ?l1 ?l2 - location) ; l1 is adjacent north of l2
        (loc-above ?l1 ?l2 - location) ; l1 is adjacent above l2

        (wooden ?b - block) ; material attribute - latent property: inert
        (plastic ?b - block) ; material attribute - latent property: magnetic
        (immovable ?b - block) ; kinematic attribute - block cannot be moved

        ; dynamic predicates
        (at-loc ?s - surface ?l - location) ; block or target is at location b

        (gripper-lifted) ; the gripper is lifted in the air
        (grasping ?b - block) ; the gripper is closed around block b

        (on ?b - block ?s - surface) ; block b is on surface s
        (north-of ?b1 ?b2 - block) ; block b1 is immediately north of b2

        ; safety predicates
        (not-grid-aligned ?b - block)
            ; block is not aligned to the grid.
            ; If an action knocks a block out of the grid, it fails
            ; e.g. if a tower gets knocked over

        ; pseudo-derived predicates
        (gripper-full) ; the gripper is not grasping anything ∀ [?b - block] [not [grasping ?b]]
        (occupied ?l - location) ; There is a surface occupying this location. i.e. ∃ [?s - surface] [at-loc ?s ?l]
        (obstructed-above ?l - location) ; there is nothing occupying the location above ∀ [?l2 - location] when [loc-above ?l2 ?l1] [not [occupied ?l2]]
        (obstructed-north ?l - location) ; there is nothing occupying the location north of ?l. i.e. ∀ [?l2 - location] when [loc-north-of ?l2 ?l] [not [occupied ?l2]]
        (obstructed-south ?l - location) ; there is nothing occupying the location north of ?l. i.e. ∀ [?l2 - location] when [loc-north-of ?l ?l2] [not [occupied ?l2]]
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
        (or (not (gripper-full)) (grasping ?target))
    )
    :effect (and
        (gripper-full)
        (grasping ?target)
        (gripper-lifted)
        (not (obstructed-above ?supportloc)) ; the support location is now free
        (not (at-loc ?target ?targetloc))
    )
)

;;; place the grasped block ?target onto a free location ?freeloc above surface ?target at location ?targetloc
;;; the physical interpretation of this action is that the arm will move over
;;;     the target location ?freeloc, descend to the ?target surface, and release
;;; precondition: it needs to already be grasping and lifted in the air
(:action place-block
    :parameters (?grasped - block ?target - surface ?freeloc ?targetloc - location)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        ; this disjunction is necessary in the case where the block is already at the location but still being held
        (or (not (occupied ?freeloc)) (at-loc ?grasped ?freeloc))
        (at-loc ?target ?targetloc)
        (loc-above ?freeloc ?targetloc)
    )
    :effect (and
        (on ?grasped ?target)
        (not (gripper-full))
        (at-loc ?grasped ?freeloc)
        (not (obstructed-above ?freeloc))
        (obstructed-above ?targetloc)
        (gripper-lifted)
    )
)

;;; drag the grasped block horizontally 1 unit north or south
;;; the physical interpretation of this action is that the block will be grasped
;;;     and then the arm will drag it to a new location
(:action drag-block
    :parameters (?grasped - block ?fromloc ?toloc ?belowfromloc ?belowtoloc - location)
    :precondition (and
        (at-loc ?grasped ?fromloc)
        (not (occupied ?toloc))
        (or (loc-north-of ?toloc ?fromloc) (loc-north-of ?fromloc ?toloc))
        (loc-above ?toloc ?belowtoloc)
        (loc-above ?fromloc ?belowfromloc)
        (not (gripper-full))
        (occupied ?belowfromloc)
    )
    :effect (and
        ; fromloc
        (not (at-loc ?grasped ?fromloc))
        (not (occupied ?fromloc))
        ; toloc
        (at-loc ?grasped ?toloc)
        (occupied ?toloc)
        ; locations below change obstructed
        (not (obstructed-above ?belowfromloc))
        (obstructed-above ?belowtoloc)
    )
)

)
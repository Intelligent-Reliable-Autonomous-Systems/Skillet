(define (domain blocks)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface location - object
        table target block - surface
    )

    (:predicates
        ; static predicates
        (loc-north-of ?l1 ?l2 - location)

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
        (grid-aligned ?b - block)
            ; block is aligned to the grid.
            ; If an action knocks a block out of the grid, it fails
            ; e.g. if a tower gets knocked over

        ; pseudo-derived predicates
        (gripper-empty) ; the gripper is not grasping anything ∀ [?b - block] [not [grasping ?b]]
        (clear-above ?s - surface ?l - location) ; there is nothing immediately above the surface at this location [no blocks or targets covering]
        (clear-north ?b - block) ; there are no blocks immediately north of this block [∀ [?b2 - block] [not [north-of ?b ?b2]]]
        (clear-south ?b - block) ; there are no blocks immediately south of this block [∀ [?b2 - block] [not [north-of ?b2 ?b]]]
    )


;;; pick block ?b from surface ?s at location ?l
;;; the gripper must be empty
;;; the physical interpretation of this action is that the gripper will raise up,
;;;     move to ?b's location, lower down, close the gripper around it, then lift the block up
(:action pick-block
    :parameters (?target - block ?support - surface ?targetloc - location)
    :precondition (and
        (clear-above ?target ?targetloc)
        (on ?target ?support)
        (or (gripper-empty) (grasping ?target))
    )
    :effect (and
        (not (gripper-empty))
        (grasping ?target)
        (gripper-lifted)
        (not (clear-above ?target ?targetloc))
        (not (at-loc ?target ?targetloc)) ; ignore the held object's location
    )
)

;;; place the grasped block ?b onto surface ?s at location ?l
;;; the physical interpretation of this action is that the arm will move over
;;;     the target location ?loc, descend to the ?target surface, and release
;;; precondition: it needs to already be grasping and lifted in the air
(:action place-block
    :parameters (?grasped - block ?target - surface ?toloc - location)
    :precondition (and
        (not (gripper-empty))
        (grasping ?grasped)
        (or (clear-above ?target ?toloc) (on ?grasped ?target))
    )
    :effect (and
        (not (gripper-lifted))
        (on ?grasped ?target)
        (at-loc ?grasped ?toloc)
        (clear-above ?grasped ?toloc)
        (not (clear-above ?target ?toloc))
    )
)


)
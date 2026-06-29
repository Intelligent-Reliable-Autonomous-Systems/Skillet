(define (domain magnet-blocks-pick-place-v1)
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

        ;; CASE 1: TARGET AND SUPPORT NOT MAGNETICALLY CONNECTED
        (when
            (or (not (plastic ?target)) (not (plastic ?support)))
            (not (on ?target ?support)) ; if either target or support is non-magnetic, they are separated
        )

        ;; CASE 2: TARGET AND SUPPORT MAGNETICALLY CONNECTED
        ;;                    (when (and (plastic ?target) (plastic ?support)))
        (when
            (and (plastic ?target) (plastic ?support))
            (two-held) ; support was picked up magnetically, so at least 2 blocks are held
        )
        (forall (?surface-below - surface) ; the surface below support
            (when
                (and
                    (on ?support ?surface-below)
                    (plastic ?target) (plastic ?support) (not (plastic ?surface-below))
                )
                (not (on ?support ?surface-below)) ; the support has been picked up magnetically, but ?surface-below is not magnetic, so they separate
            )
        )
        (when
            (and (plastic ?target) (plastic ?support))
            (not (at-loc ?support ?supportloc)) ; support has been picked up magnetically - remove from location
        )
        (forall (?loc-below - location) ; the location below support
            (when
                (and (plastic ?target) (plastic ?support) (loc-above ?supportloc ?loc-below))
                (not (obstructed-above ?loc-below)) ; the support has been removed, so ?loc-below is clear above
            )
        )

        ;; CASE 3: TARGET, SUPPORT, AND SURFACE-BELOW ARE MAGNETIC
        (when
            (exists (?surface-below ?surface-below-below - surface) ; the block below support and the block below that
                (and
                    (on ?support ?surface-below) (on ?surface-below ?surface-below-below)
                    (plastic ?target) (plastic ?support) (plastic ?surface-below) ; only when surface-below is magnetically connected
                )
            )
            (three-held) ; magnetic 3-chain exists, so three are held
        )
        (forall (?surface-below ?surface-below-below - surface) ; the block below support and the block below that
            (when
                (and
                    (on ?support ?surface-below) (on ?surface-below ?surface-below-below)
                    (plastic ?target) (plastic ?support) (plastic ?surface-below) ; only when surface-below is magnetically connected
                )
                (not (on ?surface-below ?surface-below-below)) ; surface-below is picked up, surface-below-below is disconnected
            )
        )
        (forall (?surface-below - surface ?loc-below - location) ; the block below support and its location
            (when
                (and
                    (on ?support ?surface-below) (at-loc ?surface-below ?loc-below)
                    (plastic ?target) (plastic ?support) (plastic ?surface-below) ; only when surface-below is magnetically connected
                )
                (not (at-loc ?surface-below ?loc-below)) ; surface-below is picked up
            )
        )
        (forall (?surface-below - surface ?loc-below ?loc-below-below - location) ; the block below support and its location, plus the location below that
            (when
                (and
                    (on ?support ?surface-below) (at-loc ?surface-below ?loc-below) (loc-above ?loc-below ?loc-below-below)
                    (plastic ?target) (plastic ?support) (plastic ?surface-below) ; only when surface-below is magnetically connected
                )
                (not (obstructed-above ?loc-below-below)) ; surface-below is picked up, so loc-below-below becomes unobstructed
            )
        )

        ;; TODO: 4 magnetic blocks connected
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
        (not (two-held))
        (not (three-held))

        ;; CASE 1: NO HANGING BLOCK
        ;; block at free loc
        (when
            (not (two-held))
            (on ?grasped ?target) ; there are no hanging blocks, so grasped goes directly on target
        )
        (when
            (not (two-held))
            (at-loc ?grasped ?freeloc) ; there are no hanging blocks, so grasped goes directly at freeloc
        )

        ;; CASE 2: EXISTS 1 HANGING BLOCK
        ;;               (on ?grasped ?hanging-block)
        (forall (?hanging-block - surface)
            (when
                (and
                    (on ?grasped ?hanging-block)
                    (two-held) (not (three-held))
                )
                (on ?hanging-block ?target) ; connect hanging block to target
            )
        )
        (forall (?hanging-block - surface)
            (when
                (and
                    (on ?grasped ?hanging-block)
                    (two-held) (not (three-held))
                )
                (at-loc ?hanging-block ?freeloc) ; hanging block goes at freeloc
            )
        )
        (forall (?above-loc - location) ; location above freeloc
            (when
                (and
                    (loc-above ?above-loc ?freeloc)
                    (two-held) (not (three-held))
                )
                (at-loc ?grasped ?above-loc) ; grasped block is 1 block above freeloc
            )
        )
        (when
            (two-held) ; at least two blocks
            (obstructed-above ?freeloc) ; ?above-loc is occupied, so ?freeloc is obstructed
        )

        ;; CASE 3: THERE ARE 2 HANGING blocks
        ;;                  (on ?grasped ?hanging-block) (on ?hanging-block ?hanging-hanging-block)
        (forall (?hanging-block ?hanging-hanging-block - surface)
            (when
                (and
                    (on ?grasped ?hanging-block)
                    (on ?hanging-block ?hanging-hanging-block)
                    (three-held)
                )
                (on ?hanging-hanging-block ?target) ; connect hanging hanging block to target
            )
        )
        (forall (?hanging-block ?hanging-hanging-block - surface)
            (when
                (and
                    (on ?grasped ?hanging-block)
                    (on ?hanging-block ?hanging-hanging-block)
                    (three-held)
                )
                (at-loc ?hanging-hanging-block ?freeloc) ; hanging hanging block goes at freeloc
            )
        )
        (forall (?hanging-block - surface ?above-loc - location) ; location above freeloc
            (when
                (and
                    (on ?grasped ?hanging-block)
                    (loc-above ?above-loc ?freeloc)
                    (three-held)
                )
                (at-loc ?hanging-block ?above-loc) ; hanging block is 1 block above freeloc
            )
        )
        (forall (?above-loc ?above-above-loc - location) ; location 2 above freeloc
            (when
                (and
                    (loc-above ?above-loc ?freeloc) (loc-above ?above-above-loc ?above-loc)
                    (three-held)
                )
                (at-loc ?grasped ?above-above-loc) ; grasped block is 2 blocks above freeloc
            )
        )
        (forall (?above-loc - location) ; location above freeloc
            (when
                (and
                    (loc-above ?above-loc ?freeloc)
                    (three-held)
                )
                (obstructed-above ?above-loc) ; above-above-loc is occupied, so above-loc is obstructed
            )
        )
    )
)

)
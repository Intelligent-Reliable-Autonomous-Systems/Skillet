(define (domain sponge-alpha2)
    (:requirements :typing :conditional-effects :negative-preconditions)
    (:types
        surface location movable - object
        table bin spill - surface
        sponge plate - movable
    )

    (:predicates
        ; static predicates
        (deformable ?m - movable) ; kinematic attribute - movable can be squeezed

        ; dynamic predicates
        (at-loc ?s - surface ?l - location) ; block or target is at location b
        (damp ?b - plate) ; material attribute - latent property: inert

        (gripper-lifted) ; the gripper is lifted in the air
        (grasping ?b - movable) ; the gripper is closed around movable b

        (on ?b - movable ?s - surface) ; movable b is on surface s
        (clear ?s - surface) ; the surface is clear

        ; pseudo-derived predicates
        (gripper-full) ; the gripper is not grasping anything ∀ [?b - block] [not [grasping ?b]]
    )


;;; pick movable ?target from surface ?support 
;;; the gripper must be empty
;;; the physical interpretation of this action is that the gripper will raise up,
;;;     move to ?target's location, lower down, close the gripper around it, then lift the movable up
(:action pick-movable
    :parameters (?target - movable ?support - surface ?targetloc - location)
    :precondition (and
        (on ?target ?support)
        (at-loc ?target ?targetloc)

        (clear ?support)
        (or (not (gripper-full)) (grasping ?target))
    )
    :effect (and
        (gripper-full)
        (grasping ?target)
        (gripper-lifted)
        (clear ?support)) ; the support location is now free
        (not (at-loc ?target ?targetloc))
    )
)

;;; place the movable object ?target onto a free location ?freeloc above surface ?target at location ?targetloc
;;; the physical interpretation of this action is that the arm will move over
;;;     the target location ?freeloc, descend to the ?target surface, and release
;;; precondition: it needs to already be grasping and lifted in the air
(:action place-movable
    :parameters (?grasped - movable ?target - surface ?targetloc - location)
    :precondition (and
        (gripper-full)
        (grasping ?grasped)
        (at-loc ?target ?targetloc)
        (clear ?targetloc)
    )
    :effect (and
        (on ?grasped ?target)
        (not (gripper-full))
        (at-loc ?grasped ?freeloc)
        (gripper-lifted)
    )
)

;;; grasp the grasped movable and wipe at a location
;;; the physical interpretation of this action is that the block will be grasped
;;;     and then the arm will drag it to a new location
(:action wipe-movable
    :parameters (?grasped - movable ?fromloc - location ?wipeloc - location)
    :precondition (and
        (at-loc ?grasped ?fromloc)
        (not (gripper-full))
    )
    :effect (and
        (not (at-loc ?grasped ?fromloc))
        (at-loc ?grasped ?toloc)
    )
)

;;; grasp the grasped movable and wipe at a location 
;;; the physical interpretation of this action is that the movable grasped
;;;     will be squeezed
(:action squeeze-movable     
    :parameters (?grasped - movable)     
    :precondition (and         
        (grasping ?grasped)        
        (gripper-full)     
        (deformable grasped)     
    )     
    :effect (and
         (not (damp grasped))
    ) 
)

;;; grasp the grasped movable and dry at a location 
;;; the physical interpretation of this action is that the block will be grasped 
;;;     and then the arm will drag it to a new location 
(:action dry-movable     
    :parameters (?grasped - movable ?dryloc - location)     
    :precondition (and         
        (grasping ?grasped)        
        (gripper-full)     
        (deformable grasped)     
    )     
    :effect (and
         (not (damp movable))
    ) 
)

)
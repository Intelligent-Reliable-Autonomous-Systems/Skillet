# Magnet Blocks World

This is an experiment with a robotic arm manipulating blocks on a table.

## Initial PDDL Model

An expert has crafted an initial PDDL model that corresponds with the robot skills. This model was crafted with wooden blocks in mind.

### Objects and Types

Locations are spatial constructs that define where things can be placed.

Blocks are the main type. They can be moved around and placed on surfaces, including other blocks.

Targets are immobile surfaces that can be multiple units wide.

A table is an immobile surface that can hold blocks and targets.

### Predicates

**Static predicates**: These predicates do not changed and are defined by the task.
- `(loc-north-of ?l1 ?l2 - location)`: Locations are spatially related by (loc-north-of), which says that ?l1 is immediately north of ?l2.

- `(wooden ?b - block)`: A material attribute to represent wooden blocks.
- `(plastic ?b - block)`: A material attribute to represent plastic blocks. These have a latent property of being magnetic. These can introduce conditional effects.
- `(immovable ?b - block)`: A kinematic attribute that the block cannot be moved. It is mounted to the table.

**Dynamic predicates**: These predicates change in response to actions.
- `(at-loc ?s - surface ?l - location)`: The block or target is at location ?l. Note that targets can be at multiple locations.

- `(gripper-lifted)`: The gripper is lifted in the air.
- `(grasping ?b - block)`: The gripper is closed around block ?b.

- `(on ?b - block ?s - surface)`: Block ?b is on surface ?s.
- `(north-of ?b1 ?b2 - block)`: Block ?b1 is immediately north of ?b2.

**Safety predicates**: These predicates define safety conditions that must never be violated. If the agent should enter a recovery mode to rectify as soon as possible.

- `(not-grid-aligned ?b - block)`: The block is not aligned to the grid. This can happen when a tower gets knocked over. A block might overlap multiple locations or be at a weird angle.

**Pseudo-derived predicates**: These predicates track *forall not* patterns.
- `(gripper-empty)`: The gripper is not grasping anything ∀ [?b - block] [not [grasping ?b]]
- `(clear-above ?s - surface ?l - location)`: There is nothing immediately above the surface at this location [no blocks or targets covering]. The location is included to disambiguate surfaces that are multiple locations wide.
- `(clear-north ?b - block)`: There are no blocks immediately north of this block [∀ [?b2 - block] [not [north-of ?b ?b2]]].
- `(clear-south ?b - block)`: There are no blocks immediately south of this block [∀ [?b2 - block] [not [north-of ?b2 ?b]]].

### Actions

**Pick-Block**: Pick block ?b from surface ?s at location ?l. Precondition: The gripper must be empty
 - The physical interpretation of this action is that the gripper will raise up, move to ?b's location, lower down, close the gripper around it, then lift the block up.

```lisp
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
```

**Place-Block**: Place the grasped block ?b onto surface ?s at location ?l
- The physical interpretation of this action is that the arm will move over the target location ?loc, descend to the ?target surface, and release.
- Precondition: it needs to already be grasping and lifted in the air

```
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
        (clear-above ?grasped ?loc)
        (not (clear-above ?target ?loc))
    )
)
```

**Drag-Block**: Drag the grasped block horizontally 1 unit north or south
- the physical interpretation of this action is that the block will be grasped and then the arm will drag it to a new location.

```
(:action drag-block
    :parameters (?grasped - block ?fromloc ?toloc - location)
    :precondition (and
        (at-loc ?grasped ?fromloc)
        (or (loc-north-of ?toloc ?fromloc) (loc-north-of ?fromloc ?toloc))
    )
    :effect (and
        (grasp ?grasped)
        (not (at-loc ?grasped ?fromloc))
        (at-loc ?grasped ?toloc)
    )
)
```

## Hardware Implementation

The Kinvova Gen3 (Lite) arm will rearrange and stack blocks according to a set of natural language tasks. The blocks will move along a 2d plane. North is defined as the direction away from the robot base. The arm carries blocks above the 'stack limit' which is 3 or 4 blocks high.

## Objects and Types

Locations are a purely spatial construct. Divide the table into n rows, each marginally longer than a block width (~1.7" for placement tolerance). If they are too big, then magnet properties will not apply correctly. A location is a rectangular prism with a square base that extends up to the stack limit.

We will have wooden and plastic (magnet) blocks. We can hardcode their corresponding material attributes as long as we can uniquely identify them.

We can use the circular rubber mats as targets if the perception system can identify them. They are roughly 2 block locations wide. Position them to fill 2 locations.

There is one table in the scene.

## Predicates

**Static predicates**: These predicates do not changed and are defined by the task.
- `(loc-north-of ?l1 ?l2 - location)`: We have this by construction. Define for each pair of neighbor locations.

- `(wooden ?b - block)`: A property associated with the block ids we assign to wooden blocks.
- `(plastic ?b - block)`: A property associated with the block ids we assign to magnet blocks.
- `(immovable ?b - block)`: A property associated with the block ids we assign to blocks mounted to the table.

**Dynamic predicates**: These predicates change in response to actions.
- `(at-loc ?s - surface ?l - location)`: For blocks, this is assigned True to any location that the block occupies a significant area of (say 10%). Ideally, the block should be centered on the location and occupy only one location. But if it is straddling a boundary, we wouldn't want to try to place another block in that partially filled space. See `(not-grid-aligned)` for discussion on this.
For targets, similar logic applies. True whereever the target covers a significant percentage.

- `(gripper-lifted)`: True when the gripper is lifted in the air above a predefined height threshold.
- `(grasping ?b - block)`: True when gripper is in closed mode, but the gripper is not fully closed because something is blocking it, and the TCP is in the boundary of block ?b.

- `(on ?b - block ?s - surface)`: True when the height of the bottom plane of block ?b is within some tolerance of the top plane of surface ?s and there is substantial horizontal overlap.
- `(north-of ?b1 ?b2 - block)`: True when the southmost plane of block ?b1 is within some tolerance of the northmost plane of block ?b1. This indicates the ?b1 is on the north side of ?b2.

**Safety predicates**: These predicates define safety conditions that must never be violated. If the agent should enter a recovery mode to rectify as soon as possible.

- `(not-grid-aligned ?b - block)`: Every block below the stack limit must be contained in exactly 1 location. The orientation of the block should be in the cardinal directions. Otherwise, set this property to true. This should trigger a recovery mode instead of active exploration.

**Pseudo-derived predicates**: These predicates track *forall not* patterns. *I am considering whether these should be defined as purely derived predicates in PDDL rather than defining as spatial predicates.*
- `(gripper-empty)`: True when the gripper is open or full closed (not gripping anything).
- `(clear-above ?s - surface ?l - location)`: True when the area right above a block/target/table within the horizontal bounds of a location is free space.
- `(clear-north ?b - block)`: True when the area just north of a block (same height) is free space.
- `(clear-south ?b - block)`: Same but for south.

## Skills

### Pick-Block

The Pick-Block skill lifts a block into the air. Pick-Block is invoked as `pick-block(?target-block)`. The PDDL action has additional parameters `?surface` `?location` which are used during planning, but not needed for the skill.

**Behavior**
1. Pick-Block finds the coordinate of `?target-block`. Rotate gripper perpendicular to block plane.
2. If it is not already above that location, then
    a. it should lift the arm above the stack limit
    b. move over the target location
3. Descend down to the height of the block
4. Close the gripper around the block.
5. Lift the gripper above the stack limit with (hopefully) the block in the gripper.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If the gripper is already holding a block.
- If there is no clear space next to the target object to allow the grasp.
- After each step, check if any block becomes not grid aligned `(not-grid-aligned ?b - block)`
- If the arm exceeds some resistance threshold while moving.
- If the gripper cannot grasp the object or loses its grasp on the object.

### Place-Block

The Place-Block skill sets down a block on a target surface/location. It is invoked as `Place-Block(?grasped-block, ?target-surface, ?target-location)`. If the location is not specified, randomly select a location that satisfies `(at-loc ?target-surface ?target-location)`.

**Precondition**
The gripper should already be holding an object.

**Behavior**
1. If it is not already above `?target-location`, then
    a. it should lift the arm above the stack limit
    b. move over the target location
2. Descend until the bottom of `?grasped-block` reaches the top of `?target-surface` or until it meets physical resistance.
3. Release the grasped block.
4. Lift the arm above stack limit.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If it is not holding an object to begin with
- After each step, check if any block becomes not grid aligned `(not-grid-aligned ?b - block)`
- If the arm exceeds some resistance threshold while moving, except during step 2---early resistance while moving downward is expected if there is a hanging block. The arm should still *stop immediately*. But don't report a failure, unless...
- when the block is released at a certain height, if it falls down that is unsafe.

### Drag-Block

The Drag-Block skill is indended to move a block horizontally without lifting it high in the air. Due to restrictions in the PDDL planning, this can only move one location at a time. Chain them together if you want to drag multiple spaces.

It is invoked as `Drag-Block(?dragged-block, ?from-location, ?target-location)`, indicating that `?dragged-block` will be moved horizontally from `?from-location` to `?target-location`.

**Precondition**
The gripper cannot be holding anything, unless it is already holding `?dragged-block`. `?target-location` must be adjacent to `?from-location` (immediately north or south).

**Behavior**
The first part of Drag-Block is like Pick-Block
1. Find the coordinate of `?dragged-block`. Rotate gripper perpendicular to block plane.
2. If it is not already above that location, then
    a. it should lift the arm above the stack limit
    b. move over the target location
3. Descend down to the height of the block
4. Close the gripper around the block.
5. Instead of lifting up, move horizontally toward `?target-location`.
6. *Do not release*, in case we want to chain another Drag-Block.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If the gripper is already holding a block besides `?dragged-block`.
- If `?target-location` is not adjacent to `?from-location` (immediately north or south).
- If there is no clear space next to the target object to allow the grasp.
- After each step, check if any block becomes not grid aligned `(not-grid-aligned ?b - block)`
- If the arm exceeds some resistance threshold while moving (this includes horizontal resistance, such as if it goes against something immobile)
- If the gripper cannot grasp the object or loses its grasp on the object.

## Tasks

We want to be able to stack the blocks in a variety of configurations according to natural language prompts. Here I define several families of prompts.

*"Place the red block on top of the blue block"*: place a block of a certain color on another block.
- PDDL goal: (on ?red-block ?blue-block)
- Setup: Have several blocks on the table.
- Harder setup 1: The blue block is on the red block to start.
- Harder setup 2: There are other blocks on top
- Harder setup 3: Red and blue blocks are magnetic, so have to be careful not to stack them in wrong order.

*"Stack the blocks 3 high"*: achieve a tower of blocks that is 3 high in any order.
- PDDL goal: ∃ ?b1 ?b2 ?b2 s.t. (on ?b1, ?b2) ^ (on ?b2, ?b3) ^ (on ?b3, ?table)
- Setup: have at least 3 blocks on the table in stacks of 1-2.
- Harder setup: to make it more challenge, one or more of the blocks is immovable, so it has to be the base.

*"Place the green block between the red and yellow blocks"*: lay out the blocks horizontally such that green is in the middle.
- PDDL goal: ((north-of ?red-block ?green-block) ^ (north-of ?green-block ?yellow-block)) ∨ ((north-of ?yellow-block ?green-block) ^ (north-of ?green-block ?red-block))
- Setup: Have several blocks on the table in stacks of 1-2
- Harder setup: Have the green and/or red block be immovable

*"Unstack all the blocks"*: lay out all the blocks on the table without stacks.
- PDDL goal: ∀ (?b - block) (on ?b ?table)
- Setup: Have several stacks 2-3 high
- Harder setup 1: Include 2 magnet blocks, introducing a hazard of accidentally sticking 2 blocks together.
- Harder setup 2: Start with 2 magnet blocks stuck together. Include an immovable block to allow the drag skill to separate them.

*"Stack the blocks into rainbow order, with red on the bottom"*: requires semantic understanding in order to construct the goal
- PDDL goal: (on ?blue ?green) ^ (on ?green ?yellow) ^ (on ?yellow ?red) ^ (on ?red ?table)
- Setup: stacks of blocks 1-2 high
- Harder setup 1: green and yellow are magnetic. Once they are together, they cannot be separated.
- Harder setup 2: blue and red are magnetic. They must never be put together.

*"Reverse the order of the stack(s)"*: task goal must be constructed relative to the initial state.
- PDDL goal: [assume initial state has (on ?red ?blue), (on ?blue ?yellow)] then the goal is (on ?yellow ?blue) ^ (on ?blue ?red) ^ (on ?red ?table)
- Setup: a single stack 2-3 high
- Harder setup 1: multiple stacks
- Harder setup 2: red and yellow are magnet blocks. Be careful not to stick them together.


# Magnet Blocks World V3

This is an experiment with a robotic arm manipulating blocks on a table.

## Initial PDDL Model

An expert has crafted an initial PDDL model that corresponds with the robot skills. This model was crafted with wooden blocks in mind.

### Objects and Types

Locations are spatial constructs that define where things can be placed. These are defined in the x=North-South axis and the z=Up-Down axis.

Blocks are the main type. They can be moved around and placed at locations and on surfaces, including other blocks.

Targets are immobile surfaces that can be multiple units wide.

A table is an immobile surface that can hold blocks and targets.

### Predicates

**Static predicates**: These predicates do not changed and are defined by the task.
- `(loc-north-of ?l1 ?l2 - location)`: Locations are spatially related by (loc-north-of), which says that ?l1 and ?l2 are adjacent, with ?l1 north of ?l2.
- `(loc-above ?l1 ?l2 - location)`: Locations are also related by (loc-above), which says that ?l1 and ?l2 are adjacent, with ?l1 above ?l2.

Here is an example of how to initialize a problem with a 3x2 location grid. loc0 is the table level x=0, z=0. loc2_1 is the first block level at x=2.
```lisp
(loc-north-of loc1 loc0)
(loc-north-of loc2 loc1)
(loc-north-of loc1_1 loc0_1)
(loc-north-of loc2_1 loc1_1)

(loc-above loc0_1 loc0)
(loc-above loc0_2 loc0_1)
```

- `(wooden ?b - block)`: A material attribute to represent wooden blocks.
- `(plastic ?b - block)`: A material attribute to represent plastic blocks. These have a latent property of being magnetic. These can introduce conditional effects.
- `(immovable ?b - block)`: A kinematic attribute that the block cannot be moved. It is mounted to the table.

**Dynamic predicates**: These predicates change in response to actions.
- `(at-loc ?s - surface ?l - location)`: The block or target is at location ?l. Note that targets can cover multiple locations.

- `(gripper-lifted)`: The gripper is lifted in the air.
- `(grasping ?b - block)`: The gripper is closed around block ?b.

- `(on ?b - block ?s - surface)`: Block ?b is on surface ?s.
- `(north-of ?b1 ?b2 - block)`: Blocks ?b1 and ?b2 are adjacent with ?b1 north of ?b2.

**Safety predicates**: These predicates define safety conditions that must never be violated. If the agent should enter a recovery mode to rectify as soon as possible.

- `(not-grid-aligned ?b - block)`: The block is not aligned to the grid. This can happen when a tower gets knocked over. A block might overlap multiple locations or be at a weird angle.

**Pseudo-derived predicates**: These predicates track *quantified* patterns.
- `(gripper-full)`: The gripper is not grasping anything ∃ [?b - block] [grasping ?b]
- `(occupied ?l - location)`: There is a surface occupying this location. i.e. ∃ [?s - surface] [at-loc ?s ?l]
- `(obstructed-above ?l - location)`: There is nothing immediately above the location. The location is included to disambiguate surfaces that are multiple locations wide. i.e. ∀ [?l2 - location] when [loc-above ?l2 ?l] [not [occupied ?l2]]
- `(obstructed-north ?b - block)`: There are no blocks immediately north of this block. i.e. ∀ [?l2 - location] when [loc-north-of ?l2 ?l] [not [occupied ?l2]]
- `(obstructed-south ?b - block)`: There are no blocks immediately south of this block. i.e. ∀ [?l2 - location] when [loc-north-of ?l ?l2] [not [occupied ?l2]]

### Actions

**Pick-Block**: Pick block ?target from surface ?support at corresponding locations ?targetloc and ?supportloc. Precondition: The gripper must be empty
 - The physical interpretation of this action is that the gripper will raise up, move to ?target's location, lower down, close the gripper around it, then lift the block up.

```lisp
(:action pick
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
```

**Place-Block**: Place the grasped block ?target onto a free location ?freeloc above surface ?target at location ?targetloc
- The physical interpretation of this action is that the arm will move over the target location ?freeloc, descend to the ?target surface, and release.
- Precondition: it needs to already be grasping

```
(:action place
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
```

**Drag-Block**: Drag the grasped block ?grasped horizontally 1 unit north or south, from ?fromloc to ?toloc. The locations below are also included.
- the physical interpretation of this action is that the block will be grasped and then the arm will drag it to a new location.

```
(:action drag
    :parameters (?grasped - block ?fromloc ?toloc ?belowfromloc ?belowtoloc - location)
    :precondition (and
        (at-loc ?grasped ?fromloc)
        (not (occupied ?toloc))
        (or (loc-north-of ?toloc ?fromloc) (loc-north-of ?fromloc ?toloc))
        (loc-above ?toloc ?belowtoloc)
        (loc-above ?fromloc ?belowfromloc)
    )
    :effect (and
        ; gripper
        (grasping ?grasped)
        (gripper-full)
        (not (gripper-lifted))
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
```

## Hardware Implementation

The Kinvova Gen3 (Lite) arm will rearrange and stack blocks according to a set of natural language tasks. The blocks will move along a 2d plane. North is defined as the direction away from the robot base. The arm carries blocks above the 'stack limit' which is 3 or 4 blocks high.

## Objects and Types

**Locations**: Locations are a purely spatial construct. Divide the table into a voxel grid, with the horizontal dimension marginally longer than a block width (~1.8" for placement tolerance). If they are too big, then magnet properties will not apply correctly. The vertical dimensions should closely match the height of the blocks. A location is a voxel in this grid.
- Locations are predefined in the workspace.
- At startup, generate a set of symbols (e.g. loc1_1, loc2_3, etc) corresponding to the horizontal and vertical location.
- Level 1 is where the blocks are placed on the table. Stacked blocks go to Level 2, then 3, etc.
- There is also Level 0, which are the locations that the table occupies. These level 0 locations allow us to control where on the table things can be placed.

**Blocks**: We will have wooden and plastic (magnet) blocks each identified by a color. We can hardcode a mapping of color to their corresponding material attributes.

**Targets**: We can use the circular rubber mats as targets. They are roughly 2 block locations wide. Position them to fill 2 locations.
- Though targets are technically above the table, because they are thin, assign them to locations at Level 0, as though they are in the table.

**Table**: There is one table in the scene.

## Predicates

**Static predicates**: These predicates do not changed and are defined by the task.
- `(loc-north-of ?l1 ?l2 - location)`: We have this by construction of the voxel grid. Statically define for each pair of neighbor locations.
- `(loc-above ?l1 ?l2)`: We have this by construction of the voxel grid. Statically define for each pair of neighbor locations.

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
- `(gripper-full)`: True when the gripper is closed around a block.
- `(obstructed-above ?l - location)`: True when a location has a block or other obstruction above it. Not free space.
- `(obstructed-north ?b - block)`: True when the area just north of a location is obstructed by a block or otherwise. Not free space.
- `(obstructed-south ?b - block)`: Same but for south.

## Skills

### Pick-Block

The Pick-Block skill lifts a block into the air. Pick-Block is invoked as `pick(?target)`. The PDDL action has additional parameters `?support` `?targetloc ?supportloc` which are used during planning, but not needed for the skill.
- validate that `?target` is on `?support`
- validate that `?targetloc` and `?supportloc` correspond to the locations of `?target` and `?support`.

**Behavior**
1. Pick-Block finds the coordinate of `?target`. Rotate gripper perpendicular to block plane.
2. If it is not already above that location, then
    a. it should lift the arm above the stack limit
    b. move over the target location
3. Descend down to the height of the block
4. Close the gripper around the block.
5. Lift the gripper above the stack limit with (hopefully) the block in the gripper.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If `?target` is not on `?support`.
- If `?targetloc` and `?supportloc` do not correspond to the locations of `?target` and `?support`.
- If the gripper is already holding a block.
- If there is no clear space next to the target object to allow the grasp.
- After each step, check if any block becomes not grid aligned `(not-grid-aligned ?b - block)`
- If the arm exceeds some resistance threshold while moving.
- If the gripper cannot grasp the object or loses its grasp on the object.

### Place-Block

The Place-Block skill sets down a block on a target surface/location. It is invoked as `place(?grasped, ?target, ?freeloc ?targetloc)`. The locations `?freeloc` and `?targetloc` are constructs useful for planning. If the location is not specified, randomly select a location on `?target` that satisfies `(at-loc ?target-surface ?target-location)`.
- validate that `?freeloc` is above `?targetloc` and that `?targetloc` is the location of `?target`.

**Precondition**
The gripper should already be holding an object.

**Behavior**
1. If it is not already above `?target`, then
    a. it should lift the arm above the stack limit
    b. move over the target location
2. Descend until the bottom of `?grasped` reaches the top of `?target` or until it meets physical resistance.
3. Release the grasped block.
4. Lift the arm above stack limit.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If it is not holding an object to begin with
- After each step, check if any block becomes not grid aligned `(not-grid-aligned ?b - block)`
- If the arm exceeds some resistance threshold while moving, except during step 2---early resistance while moving downward is expected if there is a hanging block. The arm should still *stop immediately*. But don't report a failure, unless...
- when the block is released at a certain height, if it falls down that is unsafe.

### Drag-Block

The Drag-Block skill is indended to move a block horizontally without lifting it high in the air. Due to restrictions in the PDDL planning, this can only move one location at a time. Chain them together if you want to drag multiple spaces.

It is invoked as `drag(?grasped, ?from-loc, ?to-loc ?below-from-loc ?below-to-loc)`, indicating that the `?grasped` block will be moved horizontally from `?from-loc` to `?to-loc`. The locations `?below-from-loc` and `?below-to-loc` are useful planning constructs, but irrelevant for the skill execution.

**Precondition**
- validate that the gripper is not holding anything, unless it is already holding `?grasped`.
- validate that `?from-loc` is horizontally adjacent to `?to-loc` and that `?grasped` is at `?from-loc`.
- validate that `?below-from-loc` and `?below-to-loc` are the locations below `?from-loc` and `?to-loc` respectively.

**Behavior**
The first part of Drag-Block is like Pick-Block
1. Find the coordinate of `?grasped`. Rotate gripper perpendicular to block plane.
2. If it is not already above that location, then
    a. it should lift the arm above the stack limit
    b. move over the target location
3. Descend down to the height of the block
4. Close the gripper around the block.
5. Instead of lifting up, move horizontally toward `?to-loc`.
6. *Do not release*, in case we want to chain another Drag-Block.

**Failure conditions**: If any of these conditions are violated, *lift the arm straight up to a safe position* and return a *failure*.
- If any of the precondition validation fails.
- If there is no clear space next to the target object to allow the grasp. (shouldn't be a problem on a 2d cross section)
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


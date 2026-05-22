**Setup**

Place a blue target cover x=0 and x=1. Place a pink target covering x=2 and x=3. Place the red block on the blue target at x=0. Place the green block on the pink target at x=2. Place the blue block on the table at x=4.

**Natural Language Task**

Move the red block off of the blue target. Move the green block onto the blue target. Stack the blue block on the red block.

**PDDL Goal**

```lisp
    (:goal (and
            (not (on red blue_target))
            (on green blue_target)
            (on blue red)
        )
    )
```
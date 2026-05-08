**Setup**

Place 3 blocks on the table. Red at x=0, green at x=2, and blue at x=4.

**Natural Language Task**

Stack the blocks 3 tall. The red block should be above the blue block. The blue block should be on the green block.

**PDDL Goal**

```lisp
(:goal (and
        (on red blue)
        (on blue green)
    )
)
```
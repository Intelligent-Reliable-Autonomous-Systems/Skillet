**Setup**

Place the blocks in a stack at x=1. Green on the table, then red, and blue on top.

**Natural Language Goal**

Rearrange the blocks into a stack with blue on top of green on top of red.

**PDDL Goal**

```lisp
(:goal (and
        (on green red)
        (on blue green)
    )
)
```
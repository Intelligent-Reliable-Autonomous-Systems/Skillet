**Setup**

There are 3 blocks: red, green, and blue. Red is at x=2 on the table. Green is at x=1 on the table. Blue is at x=2 on red.

**Natural Language Goal**

Place all three blocks adjacent on the table in a row. From south to north, the blocks should go red, green, blue.

**PDDL Goal**

```
(:goal (or
        (and
            (at-loc red loc0_1)
            (at-loc green loc1_1)
            (at-loc blue loc2_1)
        )
        (and
            (at-loc red loc1_1)
            (at-loc green loc2_1)
            (at-loc blue loc3_1)
        )
        (and
            (at-loc red loc2_1)
            (at-loc green loc3_1)
            (at-loc blue loc4_1)
        )
        (and
            (at-loc red loc3_1)
            (at-loc green loc4_1)
            (at-loc blue loc5_1)
        )
    )
)
```
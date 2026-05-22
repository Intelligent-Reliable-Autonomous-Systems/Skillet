(define (problem three-block-table-horizontal)
    (:domain blocks)
    (:objects
        loc0 loc0_1 loc0_2 loc0_3
        loc1 loc1_1 loc1_2 loc1_3
        loc2 loc2_1 loc2_2 loc2_3
        loc3 loc3_1 loc3_2 loc3_3
        loc4 loc4_1 loc4_2 loc4_3 - location ; 5 columns stacked 3 high (plus the table level)
        red green blue - block
        table0 - table
    )
    (:init
        (loc-north-of loc1 loc0)
        (loc-north-of loc2 loc1)
        (loc-north-of loc3 loc2)
        (loc-north-of loc4 loc3)
        (loc-north-of loc1_1 loc0_1)
        (loc-north-of loc2_1 loc1_1)
        (loc-north-of loc3_1 loc2_1)
        (loc-north-of loc4_1 loc3_1)
        (loc-north-of loc1_2 loc0_2)
        (loc-north-of loc2_2 loc1_2)
        (loc-north-of loc3_2 loc2_2)
        (loc-north-of loc4_2 loc3_2)
        (loc-north-of loc1_3 loc0_3)
        (loc-north-of loc2_3 loc1_3)
        (loc-north-of loc3_3 loc2_3)
        (loc-north-of loc4_3 loc3_3)

        (loc-above loc0_1 loc0)
        (loc-above loc0_2 loc0_1)
        (loc-above loc0_3 loc0_2)
        (loc-above loc1_1 loc1)
        (loc-above loc1_2 loc1_1)
        (loc-above loc1_3 loc1_2)
        (loc-above loc2_1 loc2)
        (loc-above loc2_2 loc2_1)
        (loc-above loc2_3 loc2_2)
        (loc-above loc3_1 loc3)
        (loc-above loc3_2 loc3_1)
        (loc-above loc3_3 loc3_2)
        (loc-above loc4_1 loc4)
        (loc-above loc4_2 loc4_1)
        (loc-above loc4_3 loc4_2)

        ; table-level locations. assign table or target to each slot
        (at-loc table0 loc0)
        (at-loc table0 loc1)
        (at-loc table0 loc2)
        (at-loc table0 loc3)
        (at-loc table0 loc4)

        (wooden red)
        (wooden green)
        (wooden blue)

        (at-loc red loc2_1)
        (at-loc green loc1_1)
        (at-loc blue loc2_2)

        (on red table0)
        (on green table0)
        (on blue red)

        (gripper-lifted)
    )
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
    )
)
)

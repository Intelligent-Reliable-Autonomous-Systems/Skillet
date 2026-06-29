(define (problem blocksworld-train-07)
    (:domain blocks)
    (:objects
        loc_a0 loc_a1 loc_a2 loc_a3 loc_a4
        loc_b0 loc_b1 loc_b2 loc_b3 loc_b4
        loc_c0 loc_c1 loc_c2 loc_c3 loc_c4
        loc_d0 loc_d1 loc_d2 loc_d3 loc_d4 - location ; 4 columns stacked 4 high (plus the table level)
        yellow pink red cyan - block
        red_target blue_target green_target - target
        table0 - table
    )
    (:init
        (loc-above loc_a1 loc_a0)
        (loc-above loc_a2 loc_a1)
        (loc-above loc_a3 loc_a2)
        (loc-above loc_a4 loc_a3)
        (loc-above loc_b1 loc_b0)
        (loc-above loc_b2 loc_b1)
        (loc-above loc_b3 loc_b2)
        (loc-above loc_b4 loc_b3)
        (loc-above loc_c1 loc_c0)
        (loc-above loc_c2 loc_c1)
        (loc-above loc_c3 loc_c2)
        (loc-above loc_c4 loc_c3)
        (loc-above loc_d1 loc_d0)
        (loc-above loc_d2 loc_d1)
        (loc-above loc_d3 loc_d2)
        (loc-above loc_d4 loc_d3)

        (at-loc table0 loc_a0)
        (at-loc red_target loc_b0)
        (at-loc blue_target loc_c0)
        (at-loc green_target loc_d0)

        (wooden yellow)
        (plastic pink)
        (plastic red)
        (plastic cyan)

        (at-loc red loc_a1)
        (at-loc cyan loc_a2)
        (at-loc pink loc_b1)
        (at-loc yellow loc_d1)

        (on red table0)
        (on cyan red)
        (on pink red_target)
        (on yellow green_target)
    )
    (:goal
        (and
            (on cyan red)
            (on red pink)
            (on pink blue_target)
            (on yellow green_target)
        )
    )
)

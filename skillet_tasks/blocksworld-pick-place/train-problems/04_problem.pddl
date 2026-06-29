(define (problem blocksworld-train-04)
    (:domain blocks)
    (:objects
        loc_a0 loc_a1 loc_a2 loc_a3
        loc_b0 loc_b1 loc_b2 loc_b3
        loc_c0 loc_c1 loc_c2 loc_c3
        loc_d0 loc_d1 loc_d2 loc_d3 - location ; 4 columns stacked 3 high (plus the table level)
        yellow green pink red - block
        red_target blue_target green_target - target
        table0 - table
    )
    (:init
        (loc-above loc_a1 loc_a0)
        (loc-above loc_a2 loc_a1)
        (loc-above loc_a3 loc_a2)
        (loc-above loc_b1 loc_b0)
        (loc-above loc_b2 loc_b1)
        (loc-above loc_b3 loc_b2)
        (loc-above loc_c1 loc_c0)
        (loc-above loc_c2 loc_c1)
        (loc-above loc_c3 loc_c2)
        (loc-above loc_d1 loc_d0)
        (loc-above loc_d2 loc_d1)
        (loc-above loc_d3 loc_d2)

        (at-loc red_target loc_a0)
        (at-loc blue_target loc_b0)
        (at-loc green_target loc_c0)
        (at-loc table0 loc_d0)

        (wooden yellow)
        (wooden green)
        (plastic pink)
        (plastic red)

        (at-loc red loc_a1)
        (at-loc pink loc_a2)
        (at-loc yellow loc_b1)
        (at-loc green loc_c1)

        (on red red_target)
        (on pink red)
        (on yellow blue_target)
        (on green green_target)
    )
    (:goal
        (and
            (on green yellow)
            (on yellow table0)
            (on red pink)
            (on pink blue_target)
        )
    )
)

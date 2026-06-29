(define (problem blocksworld-train-09)
    (:domain blocks)
    (:objects
        loc_a0 loc_a1 loc_a2 loc_a3 loc_a4
        loc_b0 loc_b1 loc_b2 loc_b3 loc_b4
        loc_c0 loc_c1 loc_c2 loc_c3 loc_c4
        loc_d0 loc_d1 loc_d2 loc_d3 loc_d4
        loc_e0 loc_e1 loc_e2 loc_e3 loc_e4 - location ; 5 columns stacked 4 high (plus the table level)
        yellow green pink red cyan - block
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
        (loc-above loc_e1 loc_e0)
        (loc-above loc_e2 loc_e1)
        (loc-above loc_e3 loc_e2)
        (loc-above loc_e4 loc_e3)

        (at-loc table0 loc_a0)
        (at-loc red_target loc_b0)
        (at-loc blue_target loc_c0)
        (at-loc green_target loc_d0)
        (at-loc table0 loc_e0)

        (wooden yellow)
        (wooden green)
        (plastic pink)
        (plastic red)
        (plastic cyan)

        (at-loc pink loc_a1)
        (at-loc red loc_a2)
        (at-loc cyan loc_b1)
        (at-loc yellow loc_d1)
        (at-loc green loc_e2)

        (on pink table0)
        (on red pink)
        (on cyan red_target)
        (on yellow green_target)
        (on green yellow)
    )
    (:goal
        (and
            (on cyan red)
            (on red blue_target)
            (on green green_target)
            (on yellow table0)
            (on pink red_target)
        )
    )
)

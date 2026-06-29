(define (problem blocksworld-train-02)
    (:domain blocks)
    (:objects
        loc_a0 loc_a1 loc_a2 loc_a3
        loc_b0 loc_b1 loc_b2 loc_b3
        loc_c0 loc_c1 loc_c2 loc_c3 - location ; 3 columns stacked 3 high (plus the table level)
        yellow pink red - block
        red_target blue_target green_target - target
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

        (at-loc red_target loc_a0)
        (at-loc blue_target loc_b0)
        (at-loc green_target loc_c0)

        (wooden yellow)
        (plastic pink)
        (plastic red)

        (at-loc pink loc_a1)
        (at-loc red loc_b1)
        (at-loc yellow loc_c1)

        (on pink red_target)
        (on red blue_target)
        (on yellow green_target)
    )
    (:goal
        (and
            (on red pink)
            (on pink red_target)
            (on yellow green_target)
        )
    )
)

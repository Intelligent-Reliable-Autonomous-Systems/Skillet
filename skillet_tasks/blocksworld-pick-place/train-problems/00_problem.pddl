(define (problem blocksworld-train-00)
    (:domain magnet-blocks-pick-place)
    (:objects
        loc_a0 loc_a1 loc_a2 loc_a3
        loc_b0 loc_b1 loc_b2 loc_b3
        loc_c0 loc_c1 loc_c2 loc_c3 - location ; 3 columns stacked 3 high plus the table level
        red green cyan - block
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
        (at-loc table0 loc_a0)
        (at-loc table0 loc_b0)
        (at-loc table0 loc_c0)
        (at-loc red loc_a1)
        (at-loc cyan loc_b1)
        (at-loc green loc_c1)
        (plastic red)
        (plastic cyan)
        (wooden green)
        (on red table0)
        (on cyan table0)
        (on green table0)
    )
    (:goal
        (and
            (on red cyan)
            (on cyan green)
            (on green table0)
        )
    )
)
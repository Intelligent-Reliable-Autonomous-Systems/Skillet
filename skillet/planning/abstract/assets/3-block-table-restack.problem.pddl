(define (problem three-block-table-restack)
    (:domain blocks)
    (:objects
        red green blue - block
        table0 - table
    )
    (:init
        (small red)
        (small green)
        (small blue)

        (clear table0)
        (clear green)

        (on green blue)
        (on blue red)
        (on red table0)
    )
    (:goal (and
            (on red blue)
            (on blue green)
        )
    )
)

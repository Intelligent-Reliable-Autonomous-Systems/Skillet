(define (problem three-block-table)
    (:domain blocks)
    (:objects
        red green blue - block
        table0 - table
    )
    (:init
        (small red)
        (small green)
        (small blue)

        (handempty)
        (clear table0)
        (clear red)
        (clear green)
        (clear blue)

        (on red table0)
        (on green table0)
        (on blue table0)
    )
    (:goal (and
            (on red blue)
            (on blue green)
        )
    )
)

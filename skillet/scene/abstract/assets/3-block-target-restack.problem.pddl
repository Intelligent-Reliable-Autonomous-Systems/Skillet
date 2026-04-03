(define (problem three-block-target-restack)
    (:domain blocks)
    (:objects
        red green blue - block
        target0 target1 - target
        table0 - table
    )
    (:init
        (small red)
        (small green)
        (small blue)
        (small target0)
        (small target1)

        (handempty)
        (clear target1)
        (clear green)

        (on green blue)
        (on blue red)
        (on red target0)
    )
    (:goal (and
            (on red blue)
            (on blue green)
        )
    )
)

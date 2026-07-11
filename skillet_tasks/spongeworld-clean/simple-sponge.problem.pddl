(define (problem sponge-alpha2-prob1)
    (:domain sponge-alpha2)
    (:objects
        table1 - table
        plate1 - plate
        sponge1 sponge2 - sponge
        green_circle red_circle purple_circle - target
        marker_scribble - spill
        coke_can - can
    )
    (:init
        ; static attributes
        (graspable sponge1)
        (graspable sponge2)
        (graspable coke_can)
        (wipeable marker_scribble)
        (deformable sponge1)
        (deformable sponge2)

        (not (graspable marker_scribble))
        (on marker_scribble plate1)

        (supportable plate1)
        (supportable green_circle)
        (supportable red_circle)
        (supportable purple_circle)


        (blue sponge1)
        (yellow sponge2)

        (wet sponge1)
        (dirty plate1)

        (on sponge1 green_circle)
        (on sponge2 red_circle)
        (on coke_can purple_circle)
        (obstructed green_circle)
        (obstructed red_circle)
        (obstructed purple_circle)

    )
    (:goal (and
        (on coke_can plate1)
    ))
)
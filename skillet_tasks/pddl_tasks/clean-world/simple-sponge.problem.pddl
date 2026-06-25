(define (problem sponge-alpha2-prob1)
    (:domain sponge-alpha2)
    (:objects
        table1 - table
        plate1 - plate
        sponge1 sponge2 - sponge
        ketchup1 - ketchup
    )
    (:init
        ; static attributes
        (deformable sponge1)
        (deformable sponge2)
        (supportable table1)
        (supportable plate1)
        (blue sponge1)
        (yellow sponge2)
        (wet sponge1)
        ; plate sits on the table
        (on plate1 table1)
        ; two clean, dry sponges resting on the table
        (on sponge1 table1)
        (on sponge2 table1)
        ; ketchup is on the plate, and has made the plate dirty
        (on ketchup1 plate1)
        (dirty plate1)
        ; gripper starts empty, not lifted
    )
    (:goal (and
        (not (wet sponge1))
        (on sponge2 plate1)
        (on sponge1 table1)
    ))
)
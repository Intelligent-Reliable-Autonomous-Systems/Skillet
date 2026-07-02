(define (problem sponge-alpha2-prob1)
    (:domain sponge-alpha2)
    (:objects
        table1 - table
        plate1 - plate
        sponge1 sponge2 - sponge
        loc1 loc2 loc3 - location
        water_spill - spill
        orange_bin - bin 
        coke_can - can
    )
    (:init
        ; static attributes
        (deformable sponge1)
        (deformable sponge2)
        (graspable sponge1)
        (graspable sponge2)
        (graspable coke_can)
        (not (graspable water_spill))
        (not (deformable water_spill))

        (supportable plate1)
        (supportable loc1)
        (supportable loc2)
        (supportable loc3)

        (hoverable orange_bin)
        (hoverable plate1)

        (blue sponge1)
        (yellow sponge2)

        (wet sponge1)
        (dirty plate1)

        (on sponge1 loc1)
        (on sponge2 loc2)
        (on coke_can plate1)
        (obstructed plate1)
        (obstructed loc1)
        (obstructed loc2)

    )
    (:goal (and
        (on sponge1 plate1)
    ))
)
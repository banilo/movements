from more_datasets import fetch_abide_movements

bunch = fetch_abide_movements()

aut_target = bunch.pheno["DX_GROUP"]

color = {1 : "r", 2 : "b"}

import pylab as pl
pl.figure()
pl.subplot(1, 2, 1)
for mov, targ in zip(bunch.movement, aut_target):
    pl.plot(mov[:, :3], color[targ])
    pl.title("xyz movement")
    pl.xlabel("red: aut, blue: control")

pl.subplot(1, 2, 2)
for mov, targ in zip(bunch.movement, aut_target):
    pl.plot(mov[:, 3:], color[targ])
    pl.title("angular movement")
    pl.xlabel("red: aut, blue: control")

pl.show()


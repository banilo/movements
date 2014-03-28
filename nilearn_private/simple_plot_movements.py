import numpy as np
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


# Also look at first 2 PCs of detrend coefs
acquisition_lengths = [len(mov) for mov in bunch.movement]
min_acq_length = np.min(acquisition_lengths)
from features import trend_coef
trend_coefs = np.array([trend_coef(mov, axis=0).T.ravel() 
                        for mov in bunch.movement])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform((trend_coefs - trend_coefs.mean(0)) /
                        trend_coefs.std(0))

colors = [color[targ] for targ in aut_target]
pl.figure()
pl.scatter(pcs.T[0], pcs.T[1], c=colors, s=5, lw=0)

pl.show()


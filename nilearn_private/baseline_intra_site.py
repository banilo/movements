import numpy as np
from more_datasets import fetch_abide_movements

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit

whole_bunch = fetch_abide_movements()

sites = np.unique(whole_bunch.pheno["SITE_ID"])
dx_vals = np.unique(whole_bunch.pheno["DX_GROUP"])

pipeline = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(C=10., kernel="rbf"))])

all_scores = dict()

for site in sites:
    site_bunch = fetch_abide_movements(SITE_ID=site)
    print "\nSite %s" % site
    dx_group = site_bunch.pheno["DX_GROUP"]
    repartition = (dx_group[:, np.newaxis] == dx_vals[np.newaxis, :]).sum(0)
    print "Total subjects: %d" % len(dx_group)
    for d, r in zip(dx_vals, repartition):
        print "DX_GROUP %s: %s" % (str(d), str(r))

    cv = StratifiedShuffleSplit(dx_group, n_iter=200, test_size=0.25)
    min_acq_length = np.min([len(m) for m in site_bunch.movement])
    X = np.array([m[:min_acq_length]
                  for m in site_bunch.movement]).reshape(len(dx_group), -1)
    scores = cross_val_score(pipeline, X, dx_group, cv=cv, n_jobs=8)
    all_scores[site] = all_scores.get("site", dict())
    all_scores[site]["scores"] = scores
    all_scores[site]["n_samples"] = len(dx_group)

    dummy_scores_1 = cross_val_score(DummyClassifier("stratified"),
                                 X, dx_group, cv=cv, n_jobs=1)
    dummy_scores_2 = cross_val_score(DummyClassifier("most_frequent"),
                                 X, dx_group, cv=cv, n_jobs=1)
    dummy_scores_3 = cross_val_score(DummyClassifier("uniform"),
                                 X, dx_group, cv=cv, n_jobs=1)
    all_scores[site]["dummy_1"] = dummy_scores_1
    all_scores[site]["dummy_2"] = dummy_scores_2
    all_scores[site]["dummy_3"] = dummy_scores_3

    print "Score: %1.4f +- %1.4f" % (scores.mean(),
                                     scores.std() / np.sqrt(len(dx_group)))
    print "Dummies \n%1.4f +- %1.4f\n%1.4f +- %1.4f\n%1.4f +- %1.4f\n" % (
        dummy_scores_1.mean(), dummy_scores_1.std() / np.sqrt(len(dx_group)),
        dummy_scores_2.mean(), dummy_scores_2.std() / np.sqrt(len(dx_group)),
        dummy_scores_3.mean(), dummy_scores_3.std() / np.sqrt(len(dx_group)))

import pylab as pl
pl.figure()
width = 0.2
for i, (clf, col) in enumerate(zip(["scores",
                                    "dummy_1",
                                    "dummy_2",
                                    "dummy_3"],
                                   ["r", "b", "b", "b"])):
    mean_scores = [all_scores[site][clf].mean() for site in sites]
    se_scores = [all_scores[site][clf].std() /
                 np.sqrt(all_scores[site]["n_samples"])
                 for site in sites]

    pl.bar(np.arange(len(mean_scores)) + i * width, 
           mean_scores, width=width,
           yerr=se_scores, color=col)

pl.xticks(np.arange(len(sites)) + .5, sites, rotation=90)
pl.title("Accuracy scores of SVM vs 3 dummies")
pl.show()

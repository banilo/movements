"""A first attempt to decode conditions from motion detrending coefs"""
import numpy as np
from more_datasets import fetch_abide_movements

bunch = fetch_abide_movements()

# Get diagnosis group 1: autist, 2: control
# Transform to 1: autist, -1: control
aut_target = 2 * (1.5 - bunch.pheno["DX_GROUP"])

# There seems to be a different number of frames depending on site
# I am hoping at least TR is equal across sites. The way I implemented
# the detrending coefs will cause bias which will make acquisition
# length visible. So for the beginning I will crop everything to 
# the shortest acquisition length

acquisition_lengths = [len(mov) for mov in bunch.movement]
min_acquisition_length = np.min(acquisition_lengths)

movement = [mov[:min_acquisition_length] for mov in bunch.movement]

# extract some trend features
from features import trend_coef
trend_features = np.array([trend_coef(mov, polyorder=1, axis=0).T.ravel() 
            for mov in movement])


# Do a simple SVM on these features
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.dummy import DummyClassifier
svm = SVC(kernel="linear", C=1.)
logreg = LogisticRegression(C=1.)

# Watch out with cross validation!
# Our first approach will not stratify across sites
# But this absolutely needs to be tested

# We also have slightly unbalanced classes

cv = ShuffleSplit(len(trend_features), n_iter=100, test_size=0.1)

scores = cross_val_score(logreg, trend_features, aut_target, cv=cv,
                         n_jobs=8)

dummy_scores_1 = cross_val_score(DummyClassifier("stratified"), 
                                 trend_features, aut_target, cv=cv, n_jobs=8)
dummy_scores_2 = cross_val_score(DummyClassifier("most_frequent"), 
                                 trend_features, aut_target, cv=cv, n_jobs=8)
dummy_scores_3 = cross_val_score(DummyClassifier("uniform"), 
                                 trend_features, aut_target, cv=cv, n_jobs=8)


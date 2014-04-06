import numpy as np
import scipy
from more_datasets import fetch_abide_movements
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

# scikits
from scikits import bootstrap

bunch = fetch_abide_movements()

aut_target = bunch.pheno["DX_GROUP"]

nparticipants = len(aut_target)

FS = np.zeros((nparticipants,24))
for i in xrange(6):
	values = [np.mean(mov[:,i]) for mov in bunch.movement]
	FS[:,i] = values
	mins = [np.min(mov[:,i]) for mov in bunch.movement]
	FS[:,i+6] = mins
	maxs = [np.max(mov[:,i]) for mov in bunch.movement]
	FS[:,i+12] = maxs
	stds = [np.std(mov[:,i]) for mov in bunch.movement]
	FS[:,i+18] = stds

FS = np.log10(np.abs(FS)+0.1)
FS = StandardScaler().fit_transform(FS)

inds_kids = bunch.pheno['AGE_AT_SCAN']<18.
inds_adults = np.logical_not(inds_kids)
inds_males = bunch.pheno['SEX']==1
inds_females = np.logical_not(inds_males)
inds_eyesopen = bunch.pheno['EYE_STATUS_AT_SCAN']==1
inds_eyesclosed = bunch.pheno['EYE_STATUS_AT_SCAN']==2
inds_nomeds = bunch.pheno['CURRENT_MED_STATUS']=='0'
inds_meds = bunch.pheno['CURRENT_MED_STATUS']=='1'
inds_right = bunch.pheno['HANDEDNESS_CATEGORY']=='L'

"""
i = inds_females
FS = FS[i,:]
aut_target = aut_target[i,:]
"""


"""
#param_dict = {"C": [10000, 5000, 1000, 100, 10, 1, 0.1, 0.5, 0.001],
param_dict = {"C":np.linspace(0.1,1,100),
	"penalty":['l1', 'l2']}
grid = GridSearchCV(clf, param_dict, cv=cv)
realscores = grid.fit(FS, aut_target)
"""

n_bs = 10000

# bootstrap inference
all_scores = dict()
n_folds = 10
aut_target[aut_target==2] = 0

# do folding by hand to get classif decisions
folder = StratifiedKFold(aut_target, n_folds=10)
real_pred = []
real_acc = []
dumm1_pred = []
dummy1_acc,dummy2_acc,dummy3_acc = [], [], []
for train_index, test_index in folder:
	clf = LogisticRegression()
	clf.fit(FS[train_index,:],aut_target[train_index])
	labels = np.asarray(clf.predict(FS[test_index,:]))
	acc = np.mean(aut_target[test_index]==labels)
	real_pred += list(labels)
	real_acc += list(acc)

print "REAL ACCURACY MEAN: %.2f" % np.mean(real_acc)
print "DUMMY1 ACCURACY MEAN: %.2f" % np.mean(pred_acc)

# compute CI of predictions
empirical_sample = np.asarray(real_pred)
real_ci = bootstrap.ci(empirical_sample, statfunction=scipy.mean, n_samples=n_bs)
empirical_sample = np.asarray(dummy1_pred)
real_ci = bootstrap.ci(empirical_sample, statfunction=scipy.mean, n_samples=n_bs)



"""
cv = StratifiedKFold(aut_target, n_folds=5)
real_scores = cross_val_score(clf,X=FS,y=aut_target,cv=cv,n_jobs=4)
dummy_scores_1 = cross_val_score(DummyClassifier("stratified"), 
                                 X=FS,y=aut_target,cv=cv,n_jobs=4)
dummy_scores_2 = cross_val_score(DummyClassifier("most_frequent"), 
                                 X=FS,y=aut_target,cv=cv,n_jobs=4)
dummy_scores_3 = cross_val_score(DummyClassifier("uniform"), 
                                 X=FS,y=aut_target,cv=cv,n_jobs=4)
"""




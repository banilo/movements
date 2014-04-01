import numpy as np
from more_datasets import fetch_abide_movements

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


inds_kids = bunch.pheno['AGE_AT_SCAN']<18.
inds_adults = np.logical_not(inds_kids)
inds_males = bunch.pheno['SEX']==1
inds_females = np.logical_not(inds_males)
inds_eyesopen = bunch.pheno['EYE_STATUS_AT_SCAN']==1
inds_eyesclosed = bunch.pheno['EYE_STATUS_AT_SCAN']==2
inds_nomeds = bunch.pheno['CURRENT_MED_STATUS']=='0'
inds_meds = bunch.pheno['CURRENT_MED_STATUS']=='1'
inds_right = bunch.pheno['HANDEDNESS_CATEGORY']=='L'


i = inds_females
FS = FS[i,:]
aut_target = aut_target[i,:]


FS = np.log10(np.abs(FS)+0.1)
from sklearn.preprocessing import StandardScaler

FS = StandardScaler().fit_transform(FS)

from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(aut_target, n_folds=5)

foldacc = cross_val_score(LinearSVC(),X=FS,y=aut_target,cv=cv,n_jobs=1)
acc = np.mean(foldacc)
print(acc)



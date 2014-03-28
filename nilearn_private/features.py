import numpy as np
from scipy.linalg import lstsq


def skewness(array, axis=None):
    if axis is None:
        axis = 0
        array = array.ravel()

    array = np.rollaxis(array, axis)
    sd = array.std(axis=0)
    z_score = ((array - array.mean(axis=0)[np.newaxis, ...]) /
               sd[np.newaxis, ...])

    return (z_score ** 3).mean(axis=0)


def kurtosis(array, axis=None):
    if axis is None:
        axis = 0
        array = array.ravel()

    array = np.rollaxis(array, axis)
    sd = array.std(axis=0)
    z_score = ((array - array.mean(axis=0)[np.newaxis, ...]) /
               sd[np.newaxis, ...])

    return (z_score ** 4).mean(axis=0)


def trend_coef(array, polyorder=1, axis=None):

    if axis is None:
        axis = 0
        array = array.ravel()

    array = np.rollaxis(array, axis)
    remaining_shape = array.shape[1:]
    length = len(array)

    array = array.reshape(length, -1)
    x = np.arange(length)
    x -= x.mean()
    x /= x.max()

    regressors = np.array([x ** order for order in range(polyorder + 1)]).T
    coef, resids, rank, s = lstsq(regressors, array)

    return coef.reshape([polyorder + 1] + list(remaining_shape))


if __name__ == "__main__":
    import nibabel as nb
    from nilearn.datasets import fetch_haxby
    h = fetch_haxby(n_subjects=1)
    data = nb.load(h.func[0]).get_data().T

    skewmap = skewness(data, axis=0)
    kurtmap = kurtosis(data, axis=0)
    trendmap = trend_coef(data, polyorder=2, axis=0)

    

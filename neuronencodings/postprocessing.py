import numpy as np
from sklearn import decomposition, manifold


def hist_1d(arr):
    """ 1d histogram along axis for feature extraction """
    return np.histogram(arr / arr.max(), bins=np.linspace(0, 1, 11))[0]


def feat_stats(arr):
    """ Simple statistics """
    return np.concatenate([np.mean(arr, axis=0),
                           np.median(arr, axis=0),
                           np.var(arr, axis=0)])


def combine_views(views, use_hist=False):
    """ Simple statistics to combine views from same cell / entity """
    views = np.array(views).copy()

    feats = feat_stats(views)

    if use_hist:
        hist = np.apply_along_axis(hist_1d, axis=1, arr=views)
        feats = np.concatenate([feat_stats(hist), feats])

    return feats


def pca(features, ndims=10):
    """ Applies PCA to features / views """

    features_n = features.copy()
    std = np.std(features_n, axis=0)[None]
    std[std == 0] = 1
    features_n /= std
    features_n -= np.nanmean(features_n, axis=0)[None]

    pca = decomposition.PCA(n_components=ndims)
    features_t = pca.fit_transform(features_n)

    return features_t


def tsne(features, ndims=3):
    """ Fits and applies TSNE to features / views """

    features_n = features.copy()
    std = np.std(features_n, axis=0)[None]
    std[std == 0] = 1
    features_n /= std
    features_n -= np.nanmean(features_n, axis=0)[None]

    tsne = manifold.TSNE(n_components=ndims)
    features_t = tsne.fit_transform(features_n)

    return features_t


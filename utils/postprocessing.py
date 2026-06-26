import numpy as np
from sklearn.decomposition import PCA


def postprocess(well_embs, ctrl_mask):
    """Fit MAD scaler + sphering on control wells, transform all wells.

    Args:
        well_embs: (N, D) float32 array of well-level embeddings.
        ctrl_mask: boolean array of length N marking negative-control wells.
    Returns:
        (N, D) postprocessed embeddings (not yet L2-normalised).
    """
    ctrl_embs = well_embs[ctrl_mask]
    if len(ctrl_embs) < 10:
        print(f"Warning: only {len(ctrl_embs)} control wells; fitting on all wells.")
        ctrl_embs = well_embs
    mad = MADScaler().fit(ctrl_embs)
    sphere = SpheringTransform().fit(mad.transform(ctrl_embs))
    return sphere.transform(mad.transform(well_embs))


class MADScaler:
    """Robust per-dimension scaling: subtract median, divide by MAD."""

    def fit(self, X):
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)
        self.mad_ = np.where(self.mad_ < 1e-8, 1.0, self.mad_)
        return self

    def transform(self, X):
        return (X - self.median_) / self.mad_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SpheringTransform:
    """PCA whitening (sphering). Fit on negative controls, transform all wells."""

    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components, whiten=True)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

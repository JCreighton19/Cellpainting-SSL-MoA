import numpy as np
from sklearn.decomposition import PCA


def postprocess(well_embs, ctrl_mask, plates=None):
    """Fit per-plate MAD scaler + global sphering on control wells, transform all wells.

    Matches the Cell-DINO paper's "sphering + MAD robustize": MAD-normalize each
    plate independently using that plate's own control-well median/MAD, then fit
    a single sphering (PCA whitening) transform on the pooled, MAD-normalized
    controls across all plates.

    Args:
        well_embs: (N, D) float32 array of well-level embeddings.
        ctrl_mask: boolean array of length N marking negative-control wells.
        plates: array-like of length N giving each well's plate ID. If None,
            all wells are treated as a single plate (previous global-MAD behavior).
    Returns:
        (N, D) postprocessed embeddings (not yet L2-normalised).
    """
    plates = np.asarray(plates) if plates is not None else np.zeros(len(well_embs), dtype=int)

    mad_normed = np.empty_like(well_embs)
    for plate in np.unique(plates):
        plate_mask = plates == plate
        plate_ctrl = well_embs[plate_mask & ctrl_mask]
        if len(plate_ctrl) < 10:
            print(f"Warning: plate {plate} has only {len(plate_ctrl)} control wells; fitting MAD on all wells from this plate.")
            plate_ctrl = well_embs[plate_mask]
        mad = MADScaler().fit(plate_ctrl)
        mad_normed[plate_mask] = mad.transform(well_embs[plate_mask])

    sphere = SpheringTransform().fit(mad_normed[ctrl_mask])
    return sphere.transform(mad_normed)


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

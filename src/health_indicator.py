"""A single PCA HI; fitted anchors and loadings are persisted."""
import numpy as np
from sklearn.decomposition import PCA


class PCAHealthIndicator:
    def fit(self, standardized, metadata, config):
        self.pca = PCA(n_components=1).fit(standardized)
        scores = self.pca.transform(standardized).ravel()
        early = metadata.cycle <= config["hi_anchor_cycles"]
        late = metadata.cycle > metadata.groupby("engine_id").cycle.transform("max") - config["hi_terminal_cycles"]
        self.healthy_anchor = float(np.median(scores[early]))
        self.failed_anchor = float(np.median(scores[late]))
        if abs(self.healthy_anchor-self.failed_anchor) < 1e-8:
            raise ValueError("PCA does not separate early and terminal reference states")
        return self

    def transform(self, standardized):
        score = self.pca.transform(standardized).ravel()
        return np.clip(100*(score-self.failed_anchor)/(self.healthy_anchor-self.failed_anchor), 0, 100)

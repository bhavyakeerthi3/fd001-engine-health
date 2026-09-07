# Health indicator reasoning

One PCA component is used because it gives a compact sensor-only coordinate with auditable loadings and no learned nonlinear HI. PCA is fitted to standardized Keep sensors on fitting engines only; it optimizes variance, not prognostic accuracy. Explained variance ratio from the run is 0.640413. Exact loadings are in health_indicator.json, not hand-assigned weights.

Healthy anchor: median PC score during the first 20 cycles of fitting engines. Failed anchor: median PC score in the last 5 cycles. These training lifecycle labels orient/calibrate the score; it is not fully unsupervised calibration. Health = clip(100*(PC-failed)/(healthy-failed), 0, 100). This definition resolves PCA's arbitrary sign. It is a relative project score, not probability of failure; raw cycle-level health can fluctuate and is not forced monotonic. HI is shown independently rather than fed into the RUL models.


# Data and EDA reasoning

The official NASA archive is retained by reference and identity hashes. Parsing rejects wrong column counts, missing/non-finite values, duplicate engine/cycle pairs and noncontiguous cycles. Every raw column is documented in data/column_dictionary.csv. Sensor symbols and units follow Table 2 in the bundled NASA modeling paper; operational setting names/units remain unspecified where the numbered text schema does not identify their order. HPC/LPC mean high/low pressure compressor; HPT/LPT mean high/low pressure turbine. The source labels PCNfR_dmd in rpm; preserve that attribution rather than silently interpreting the stored scale.

Partition training engines before learning any sensor policy. Lifetimes for the supplied train file are descriptive; sensor decisions, slopes and starting-condition statistics use only the fitting partition. See engine_lifetimes.csv, engine_sensor_analysis.csv, operating_settings.csv and sensor_analysis.csv.

Each sensor's pooled variance and Pearson correlations with cycle, raw RUL and clipped RUL are reported. Pooled correlations can reflect engine offsets, so retention uses median within-engine absolute Spearman correlation and linear-fit R2. Monotonicity is the absolute mean sign of first differences after a causal rolling mean. Trend strength is within-engine linear R2. Noise proxy is std(first difference)/sqrt(2*variance); it includes some degradation signal and is not a measured sensor-noise calibration.

The recorded selection rules are {'minimum_unique': 3, 'minimum_median_abs_spearman': 0.3, 'minimum_trend_r2': 0.1}. These are project analysis rules, not NASA limits or optimized physical constants. Remove low-cardinality channels, Investigate weak trends, Keep supported trends. Normalize is recorded as a separate preprocessing action on every Keep sensor: physical scales should not determine PCA loadings or LSTM optimization. No sensor must receive each verdict just to fill the categories. Starting-condition variation is the between-engine standard deviation of the first 20 cycle means. Degradation rates are descriptive whole-life slopes, not constant-rate physical models.

| sensor | classification | preprocessing | reason |
| --- | --- | --- | --- |
| sensor_1 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_2 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.670, trend R2=0.468; retain and standardize on training engines. |
| sensor_3 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.648, trend R2=0.441; retain and standardize on training engines. |
| sensor_4 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.787, trend R2=0.625; retain and standardize on training engines. |
| sensor_5 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_6 | Remove | nan | Only 2 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_7 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.776, trend R2=0.614; retain and standardize on training engines. |
| sensor_8 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.769, trend R2=0.605; retain and standardize on training engines. |
| sensor_9 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.739, trend R2=0.557; retain and standardize on training engines. |
| sensor_10 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_11 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.820, trend R2=0.667; retain and standardize on training engines. |
| sensor_12 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.809, trend R2=0.660; retain and standardize on training engines. |
| sensor_13 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.762, trend R2=0.599; retain and standardize on training engines. |
| sensor_14 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.776, trend R2=0.615; retain and standardize on training engines. |
| sensor_15 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.722, trend R2=0.544; retain and standardize on training engines. |
| sensor_16 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_17 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.671, trend R2=0.467; retain and standardize on training engines. |
| sensor_18 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_19 | Remove | nan | Only 1 unique values; effectively constant or binary near-constant signal under this regime. |
| sensor_20 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.708, trend R2=0.523; retain and standardize on training engines. |
| sensor_21 | Keep | Normalize | Consistent within-engine trend: median /rho/=0.708, trend R2=0.533; retain and standardize on training engines. |

## Recorded lifetime distribution

| partition | count | min | median | mean | max | std |
| --- | --- | --- | --- | --- | --- | --- |
| train | 80 | 128 | 199.0000 | 207.0125 | 362 | 47.4564 |
| validation | 20 | 135 | 193.5000 | 203.5000 | 269 | 42.6139 |

## Recorded rate and initial-state variation

| sensor | minimum_slope | median_slope | maximum_slope | starting_mean_std |
| --- | --- | --- | --- | --- |
| sensor_11 | 0.0017 | 0.0033 | 0.0057 | 0.1444 |
| sensor_12 | -0.0163 | -0.0085 | -0.0040 | 0.4038 |
| sensor_13 | 0.0001 | 0.0007 | 0.0017 | 0.0461 |
| sensor_14 | -0.1786 | 0.0406 | 0.4848 | 7.2942 |
| sensor_15 | 0.0003 | 0.0004 | 0.0007 | 0.0170 |
| sensor_17 | 0.0096 | 0.0173 | 0.0261 | 0.6965 |
| sensor_2 | 0.0027 | 0.0053 | 0.0091 | 0.2357 |
| sensor_20 | -0.0036 | -0.0020 | -0.0012 | 0.0831 |
| sensor_21 | -0.0020 | -0.0012 | -0.0006 | 0.0501 |
| sensor_3 | 0.0318 | 0.0651 | 0.1003 | 2.4323 |
| sensor_4 | 0.0631 | 0.1093 | 0.1779 | 4.3533 |
| sensor_7 | -0.0182 | -0.0099 | -0.0046 | 0.4723 |
| sensor_8 | 0.0001 | 0.0007 | 0.0016 | 0.0451 |
| sensor_9 | -0.1273 | 0.0753 | 0.5888 | 7.3347 |

Slopes are in each sensor's own units, so compare engines within a sensor rather than slope magnitude across sensors. Nonzero starting-mean spread shows why one nominal reference trajectory cannot represent every engine. Whole-life slopes summarize different rates but also compress nonlinear degradation and measurement noise.

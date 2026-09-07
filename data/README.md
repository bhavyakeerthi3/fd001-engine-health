# FD001 source data

Official NASA archive: https://data.nasa.gov/docs/legacy/CMAPSSData.zip

See provenance.json for retrieval time, byte counts and locally calculated SHA-256 identities. These hashes are reproducibility checks, not an independent authenticity guarantee. Only FD001 and the source documentation are extracted. Raw files are immutable pipeline inputs.

| column | unit | description |
| --- | --- | --- |
| engine_id | Integer | Engine identifier within train or test; namespaces are separate. Never a model feature. |
| cycle | Cycles | Observed operational age, starts at one; used for ordering and targets, excluded from RUL features. |
| setting_1 | Unspecified | NASA operational setting 1; raw archive does not provide a named quantity/unit. Retained in raw data; excluded under FD001's single regime. |
| setting_2 | Unspecified | NASA operational setting 2; raw archive does not provide a named quantity/unit. Retained in raw data; excluded under FD001's single regime. |
| setting_3 | Unspecified | NASA operational setting 3; raw archive does not provide a named quantity/unit. Retained in raw data; excluded under FD001's single regime. |
| sensor_1 | deg R | T2: Total temperature at fan inlet. Source column 6; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_2 | deg R | T24: Total temperature at LPC outlet. Source column 7; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_3 | deg R | T30: Total temperature at HPC outlet. Source column 8; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_4 | deg R | T50: Total temperature at LPT outlet. Source column 9; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_5 | psia | P2: Pressure at fan inlet. Source column 10; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_6 | psia | P15: Total pressure in bypass duct. Source column 11; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_7 | psia | P30: Total pressure at HPC outlet. Source column 12; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_8 | rpm | Nf: Physical fan speed. Source column 13; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_9 | rpm | Nc: Physical core speed. Source column 14; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_10 | dimensionless | epr: Engine pressure ratio P50/P2. Source column 15; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_11 | psia | Ps30: Static pressure at HPC outlet. Source column 16; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_12 | pps/psi (source notation) | phi: Ratio of fuel flow to Ps30. Source column 17; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_13 | rpm | NRf: Corrected fan speed. Source column 18; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_14 | rpm | NRc: Corrected core speed. Source column 19; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_15 | dimensionless | BPR: Bypass ratio. Source column 20; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_16 | dimensionless | farB: Burner fuel-air ratio. Source column 21; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_17 | not specified (source dash) | htBleed: Bleed enthalpy. Source column 22; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_18 | rpm | Nf_dmd: Demanded fan speed. Source column 23; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_19 | rpm (as printed in source table) | PCNfR_dmd: Demanded corrected fan speed. Source column 24; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_20 | lbm/s | W31: HPT coolant bleed. Source column 25; mapping follows Table 2 in the bundled NASA modeling paper. |
| sensor_21 | lbm/s | W32: LPT coolant bleed. Source column 26; mapping follows Table 2 in the bundled NASA modeling paper. |

RUL_FD001.txt: one nonnegative number per test engine, in ascending engine-ID order. It is remaining life AFTER the final observed test cycle. Training trajectories reach failure; test trajectories are censored.

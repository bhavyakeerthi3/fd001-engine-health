# Anomaly detection reasoning

Isolation Forest fits last/mean/std/slope sensor-window features from the first 30 cycles of fitting engines, as a proxy for nominal operation. The forest has 200 trees and contamination 0.05; these are declared project settings. Including all terminal degradation in nominal training would weaken the intended early-life reference. Initial wear still varies, so early cycles are not certified healthy.

A negative decision_function marks unusual windows relative to this reference. It can flag expected degradation as well as other deviations. No labeled anomaly truth is supplied by FD001; anomaly precision, recall and fault-onset detection accuracy are therefore not reported. Flag rates and the learned offset appear in anomaly_summary.json. Anomaly output does not alter RUL or the maintenance-status rule.


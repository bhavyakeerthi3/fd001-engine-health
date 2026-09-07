"""Sensor order from Table 2 of the NASA paper bundled in the archive.

Units below transcribe that source; raw numbered columns remain canonical.
"""
SENSOR_DESCRIPTIONS = [
    ("T2", "Total temperature at fan inlet", "deg R"),
    ("T24", "Total temperature at LPC outlet", "deg R"),
    ("T30", "Total temperature at HPC outlet", "deg R"),
    ("T50", "Total temperature at LPT outlet", "deg R"),
    ("P2", "Pressure at fan inlet", "psia"),
    ("P15", "Total pressure in bypass duct", "psia"),
    ("P30", "Total pressure at HPC outlet", "psia"),
    ("Nf", "Physical fan speed", "rpm"),
    ("Nc", "Physical core speed", "rpm"),
    ("epr", "Engine pressure ratio P50/P2", "dimensionless"),
    ("Ps30", "Static pressure at HPC outlet", "psia"),
    ("phi", "Ratio of fuel flow to Ps30", "pps/psi (source notation)"),
    ("NRf", "Corrected fan speed", "rpm"),
    ("NRc", "Corrected core speed", "rpm"),
    ("BPR", "Bypass ratio", "dimensionless"),
    ("farB", "Burner fuel-air ratio", "dimensionless"),
    ("htBleed", "Bleed enthalpy", "not specified (source dash)"),
    ("Nf_dmd", "Demanded fan speed", "rpm"),
    ("PCNfR_dmd", "Demanded corrected fan speed", "rpm (as printed in source table)"),
    ("W31", "HPT coolant bleed", "lbm/s"),
    ("W32", "LPT coolant bleed", "lbm/s"),
]

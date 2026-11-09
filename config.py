"""Central configuration for EngineHealth.

Every path, constant and subset description lives here so that the rest of the
code never contains "magic numbers".  Change a value here and the whole
project (training, evaluation, website) follows.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset" / "raw"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

# --------------------------------------------------------------- columns ----
ID_COL = "unit"          # engine identifier
CYCLE_COL = "cycle"      # operating cycle (the "time" axis)
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLUMNS = [ID_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS   # 26 columns

# Physical meaning of the 21 sensors (NASA PHM08 documentation).
SENSOR_DESCRIPTIONS = {
    "s1": "T2 - Total temperature at fan inlet",
    "s2": "T24 - Total temperature at LPC outlet",
    "s3": "T30 - Total temperature at HPC outlet",
    "s4": "T50 - Total temperature at LPT outlet",
    "s5": "P2 - Pressure at fan inlet",
    "s6": "P15 - Total pressure in bypass duct",
    "s7": "P30 - Total pressure at HPC outlet",
    "s8": "Nf - Physical fan speed",
    "s9": "Nc - Physical core speed",
    "s10": "epr - Engine pressure ratio",
    "s11": "Ps30 - Static pressure at HPC outlet",
    "s12": "phi - Ratio of fuel flow to Ps30",
    "s13": "NRf - Corrected fan speed",
    "s14": "NRc - Corrected core speed",
    "s15": "BPR - Bypass ratio",
    "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy",
    "s18": "Nf_dmd - Demanded fan speed",
    "s19": "PCNfR_dmd - Demanded corrected fan speed",
    "s20": "W31 - HPT coolant bleed",
    "s21": "W32 - LPT coolant bleed",
}

# ---------------------------------------------------------------- subsets ----
SUBSETS = {
    "FD001": {"conditions": 1, "fault_modes": "HPC degradation",
              "note": "1 operating condition, 1 fault mode - simplest baseline"},
    "FD002": {"conditions": 6, "fault_modes": "HPC degradation",
              "note": "6 operating conditions, 1 fault mode"},
    "FD003": {"conditions": 1, "fault_modes": "HPC degradation + Fan degradation",
              "note": "1 operating condition, 2 fault modes"},
    "FD004": {"conditions": 6, "fault_modes": "HPC degradation + Fan degradation",
              "note": "6 operating conditions, 2 fault modes - hardest"},
}
DEFAULT_SUBSET = "FD001"

# ------------------------------------------------------------- modelling ----
RANDOM_STATE = 42
RUL_CAP = 125            # piece-wise linear RUL: early life is treated as "healthy"
WINDOW = 10              # rolling window (cycles) for temporal features
VALIDATION_FRACTION = 0.2  # share of *engines* held out for validation
MIN_ABS_SPEARMAN = 0.05  # sensors weaker than this are dropped (feature selection)

# Fault-risk classification + health categories (all in cycles of RUL)
FAULT_THRESHOLD = 30     # RUL <= 30 -> "imminent fault" (positive class)
WARNING_THRESHOLD = 70   # 30 < RUL <= 70 -> "Warning"; above -> "Healthy"
INTERVAL_COVERAGE = 0.80 # empirical prediction-interval coverage

# NASA score constants (Saxena et al., PHM08)
NASA_A1 = 13.0
NASA_A2 = 10.0

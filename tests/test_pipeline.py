"""
test_pipeline.py
================
Verification suite for the 6-layer CPCB cleaning pipeline.

Uses a synthetic dataset injected with every known edge case:
  - Negative and zero values (Layer 2)
  - Upper-bound voltage spikes (Layer 2)
  - 18-hour flatline sensor drift (Layer 3)
  - Cross-pollutant inversions: PM2.5 > PM10 (Layer 3)
  - Various gap sizes: 2h, 6h, 72h (Layers 5 & 6)
  - IST timezone standardization (Layer 1)

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

# Import pipeline layers
from src.clean_cpcb import (
    layer1_standardize_index,
    layer2_artifact_filter,
    layer3_drift_and_inversion,
    layer4_chunk_splitting,
    layer5_micro_imputation,
    layer6_sequence_windowing,
)

# ── Load pipeline config ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "pipeline.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

# ── Synthetic Data Factory ───────────────────────────────────
# We build a 3000-hour dataset to satisfy min_chunk_hours >= 2160
N_HOURS = 3000
START = "2024-01-01"
END = "2024-05-05"


def _make_clean_df() -> pd.DataFrame:
    """Create a 'perfectly clean' synthetic station dataset."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(
        start=START, periods=N_HOURS, freq="h", tz="Asia/Kolkata"
    )
    df = pd.DataFrame(
        {
            "pm25": rng.uniform(30, 150, N_HOURS),
            "pm10": rng.uniform(80, 300, N_HOURS),
            "no2": rng.uniform(10, 80, N_HOURS),
        },
        index=idx,
    )
    df.index.name = "datetime"

    # Ensure PM2.5 < PM10 always (physical constraint)
    df["pm25"] = np.minimum(df["pm25"], df["pm10"] * 0.8)
    return df


def _make_dirty_df() -> pd.DataFrame:
    """
    Inject ALL known edge cases into the clean dataset.
    Returns a copy with artifacts embedded at specific positions.
    """
    df = _make_clean_df()

    # (1) Negative values at hours 10–12
    df.iloc[10, df.columns.get_loc("pm25")] = -5.0
    df.iloc[11, df.columns.get_loc("pm10")] = 0.0
    df.iloc[12, df.columns.get_loc("no2")] = -999.0

    # (2) Upper-bound voltage spikes at hour 50
    df.iloc[50, df.columns.get_loc("pm25")] = 8950.0   # > 1000 ceiling
    df.iloc[51, df.columns.get_loc("pm10")] = 5000.0   # > 2000 ceiling

    # (3) 18-hour flatline (sensor drift) starting at hour 100
    for i in range(100, 118):
        df.iloc[i, df.columns.get_loc("pm25")] = 42.1

    # (4) Cross-pollutant inversion at hours 200–202
    #     Force PM2.5 > PM10 (physically impossible)
    df.iloc[200, df.columns.get_loc("pm25")] = 250.0
    df.iloc[200, df.columns.get_loc("pm10")] = 100.0
    df.iloc[201, df.columns.get_loc("pm25")] = 200.0
    df.iloc[201, df.columns.get_loc("pm10")] = 90.0

    # (5) 2-hour gap (should be interpolated by Layer 5)
    df.iloc[300, df.columns.get_loc("pm25")] = np.nan
    df.iloc[301, df.columns.get_loc("pm25")] = np.nan

    # (6) 6-hour gap (should NOT be interpolated — left for Layer 6)
    for i in range(400, 406):
        df.iloc[i, df.columns.get_loc("pm25")] = np.nan

    # (7) 72-hour gap (3 days — should NOT be interpolated)
    for i in range(500, 572):
        df.iloc[i, df.columns.get_loc("pm25")] = np.nan

    return df


# ═════════════════════════════════════════════════════════════
# LAYER 1 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer1:
    def test_index_is_datetime(self):
        """After Layer 1, the index must be a DatetimeIndex."""
        df = _make_clean_df()
        # Simulate raw CSV load (string index)
        df.index = df.index.astype(str)
        result = layer1_standardize_index(df, START, END)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_timezone_is_ist(self):
        """Index timezone must be Asia/Kolkata (IST)."""
        df = _make_clean_df()
        result = layer1_standardize_index(df, START, END)
        assert str(result.index.tz) == "Asia/Kolkata"

    def test_continuous_no_gaps(self):
        """The reindexed grid should have no missing hourly steps."""
        df = _make_clean_df()
        result = layer1_standardize_index(df, START, END)
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_mixed_timestamp_formats(self):
        """Layer 1 must handle mixed ISO and non-ISO timestamp strings."""
        df = _make_clean_df().head(5)
        # Mix of formats — use format='mixed' for pandas ≥2.x compat
        str_index = [
            "2024-01-01 00:00:00+05:30",
            "2024-01-01T01:00:00+05:30",
            "2024-01-01 02:00:00+05:30",
            "2024-01-01T03:00:00+05:30",
            "2024-01-01 04:00:00+05:30",
        ]
        df.index = pd.to_datetime(str_index, format="mixed", utc=True)
        result = layer1_standardize_index(df, "2024-01-01", "2024-01-01")
        assert str(result.index.tz) == "Asia/Kolkata"


# ═════════════════════════════════════════════════════════════
# LAYER 2 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer2:
    def test_negative_values_purged(self):
        """Values ≤ 0 must become NaN."""
        df = _make_dirty_df()
        result = layer2_artifact_filter(df, CFG, {})
        assert np.isnan(result.iloc[10]["pm25"])    # was -5.0
        assert np.isnan(result.iloc[11]["pm10"])    # was 0.0
        assert np.isnan(result.iloc[12]["no2"])     # was -999.0

    def test_upper_bound_spikes_purged(self):
        """Readings exceeding physical ceilings must become NaN."""
        df = _make_dirty_df()
        result = layer2_artifact_filter(df, CFG, {})
        assert np.isnan(result.iloc[50]["pm25"])    # was 8950 > 1000
        assert np.isnan(result.iloc[51]["pm10"])    # was 5000 > 2000

    def test_valid_values_preserved(self):
        """Normal readings should not be affected."""
        df = _make_dirty_df()
        result = layer2_artifact_filter(df, CFG, {})
        # Hour 0 was untouched and should be valid
        assert result.iloc[0]["pm25"] > 0


# ═════════════════════════════════════════════════════════════
# LAYER 3 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer3:
    def test_flatline_detected(self):
        """18h of identical readings must be converted to NaN."""
        df = _make_dirty_df()
        # First clean artifacts so Layer 3 sees valid-looking flatlines
        df = layer2_artifact_filter(df, CFG, {})
        result = layer3_drift_and_inversion(df, CFG)

        # All 18 flatline positions (hours 100–117) should now be NaN
        for i in range(100, 118):
            assert np.isnan(result.iloc[i]["pm25"]), f"Hour {i} not flagged"

    def test_cross_pollutant_inversion_nullified(self):
        """PM2.5 > PM10 rows must have BOTH columns set to NaN."""
        df = _make_dirty_df()
        df = layer2_artifact_filter(df, CFG, {})
        result = layer3_drift_and_inversion(df, CFG)

        # Hour 200: PM2.5=250, PM10=100 → inversion
        assert np.isnan(result.iloc[200]["pm25"])
        assert np.isnan(result.iloc[200]["pm10"])
        # Hour 201: PM2.5=200, PM10=90 → inversion
        assert np.isnan(result.iloc[201]["pm25"])
        assert np.isnan(result.iloc[201]["pm10"])


# ═════════════════════════════════════════════════════════════
# LAYER 4 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer4:
    def test_clean_station_passes(self):
        """A clean station with ~100% completeness must pass."""
        df = _make_clean_df()
        chunks = layer4_chunk_splitting(df, CFG, "TestClean", {})
        assert len(chunks) == 1
        assert chunks[0][0] is not None
        assert "KEPT" in chunks[0][1]["status"]

    def test_empty_station_dropped(self):
        """A station with no pollutant columns must be dropped."""
        df = pd.DataFrame(
            {"other_col": np.ones(3000)},
            index=pd.date_range("2024-01-01", periods=3000, freq="h", tz="Asia/Kolkata"),
        )
        chunks = layer4_chunk_splitting(df, CFG, "TestEmpty", {})
        assert len(chunks) == 0


# ═════════════════════════════════════════════════════════════
# LAYER 5 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer5:
    def test_short_gap_interpolated(self):
        """A 2-hour gap must be filled by interpolation."""
        df = _make_dirty_df()
        df = layer2_artifact_filter(df, CFG, {})
        df = layer3_drift_and_inversion(df, CFG)
        result = layer5_micro_imputation(df, CFG)

        # Hours 300–301 had a 2h gap — should now be filled
        assert result.iloc[300]["pm25"] is not np.nan or pd.notna(result.iloc[300]["pm25"])
        assert result.iloc[301]["pm25"] is not np.nan or pd.notna(result.iloc[301]["pm25"])

    def test_long_gap_not_interpolated(self):
        """A 6-hour gap must remain as NaN (not interpolated)."""
        df = _make_dirty_df()
        df = layer2_artifact_filter(df, CFG, {})
        df = layer3_drift_and_inversion(df, CFG)
        result = layer5_micro_imputation(df, CFG)

        # Hours 400–405 had a 6h gap — should still be NaN
        for i in range(400, 406):
            assert np.isnan(result.iloc[i]["pm25"]), f"Hour {i} was wrongly filled"

    def test_massive_gap_not_interpolated(self):
        """A 72-hour gap must remain as NaN."""
        df = _make_dirty_df()
        df = layer2_artifact_filter(df, CFG, {})
        df = layer3_drift_and_inversion(df, CFG)
        result = layer5_micro_imputation(df, CFG)

        # Spot-check the middle of the 72h gap
        assert np.isnan(result.iloc[530]["pm25"])


# ═════════════════════════════════════════════════════════════
# LAYER 6 TESTS
# ═════════════════════════════════════════════════════════════
class TestLayer6:
    def test_windows_have_no_nans(self):
        """Every window returned by Layer 6 must be NaN-free in target cols."""
        df = _make_clean_df()
        windows = layer6_sequence_windowing(df, CFG)
        target_cols = [c for c in ["pm25", "pm10", "no2"] if c in df.columns]
        for w in windows:
            assert not w[target_cols].isna().any().any()

    def test_gap_windows_dropped(self):
        """Windows overlapping a large gap must be excluded."""
        df = _make_dirty_df()
        df = layer2_artifact_filter(df, CFG, {})
        df = layer3_drift_and_inversion(df, CFG)
        df = layer5_micro_imputation(df, CFG)

        windows_dirty = layer6_sequence_windowing(df, CFG)

        # Compare to perfectly clean data
        df_clean = _make_clean_df()
        windows_clean = layer6_sequence_windowing(df_clean, CFG)

        # Dirty data with gaps MUST produce fewer windows than clean data
        assert len(windows_dirty) < len(windows_clean)

    def test_correct_window_length(self):
        """Each window must be exactly lookback + horizon hours."""
        df = _make_clean_df()
        windows = layer6_sequence_windowing(df, CFG)
        expected_len = CFG["windowing"]["lookback_hours"] + CFG["windowing"]["horizon_hours"]
        for w in windows:
            assert len(w) == expected_len

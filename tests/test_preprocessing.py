"""
Unit tests for data preprocessing, cleaning, and vital signs summarization.
"""

import pandas as pd
import pytest

from src.preprocessing import clean_sensor_data, compute_vital_summary


def test_clean_sensor_data_field_mapping():
    """Verify that field1, field2, field3 are correctly mapped."""
    raw_feeds = [
        {
            "created_at": "2025-09-28T17:19:13Z",
            "entry_id": 1,
            "field1": "99.0",  # SpO2
            "field2": "74",    # Pulse / Heart Rate
            "field3": "36.9",  # Body Temperature
        },
        {
            "created_at": "2025-09-28T17:19:33Z",
            "entry_id": 2,
            "field1": "98.5",
            "field2": "76",
            "field3": "37.0",
        },
    ]

    df = clean_sensor_data(raw_feeds)

    assert "spo2" in df.columns
    assert "heart_rate" in df.columns
    assert "temperature" in df.columns

    # Verify values match expectations
    assert df.iloc[0]["spo2"] == 99.0
    assert df.iloc[0]["heart_rate"] == 74.0
    assert df.iloc[0]["temperature"] == 36.9


def test_clean_sensor_data_handles_corrupt_entries():
    """Verify that invalid strings are coerced to NaN and handled properly."""
    raw_feeds = [
        {
            "created_at": "invalid-time",
            "entry_id": 1,
            "field1": "not_a_number",
            "field2": "80",
            "field3": "36.8",
        },
        {
            "created_at": "2025-09-28T17:20:00Z",
            "entry_id": 2,
            "field1": "null",
            "field2": "",
            "field3": None,
        }
    ]

    df = clean_sensor_data(raw_feeds)
    # The second row has all null vitals and should be dropped
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["spo2"])
    assert df.iloc[0]["heart_rate"] == 80.0
    assert df.iloc[0]["temperature"] == 36.8


def test_clean_sensor_data_empty():
    """Verify clean_sensor_data returns empty DataFrame on empty input."""
    df = clean_sensor_data([])
    assert df.empty
    assert "spo2" in df.columns
    assert "heart_rate" in df.columns
    assert "temperature" in df.columns


def test_compute_vital_summary_valid():
    """Verify that mean, min, max calculations are mathematically sound."""
    data = {
        "created_at": ["2025-09-28T17:00:00Z", "2025-09-28T17:01:00Z", "2025-09-28T17:02:00Z"],
        "spo2": [96.0, 98.0, 100.0],
        "heart_rate": [70.0, 80.0, 90.0],
        "temperature": [36.0, 37.0, 38.0],
    }
    df = pd.DataFrame(data)

    summary = compute_vital_summary(df)

    assert summary.reading_count == 3
    assert pytest.approx(summary.avg_temp, 0.01) == 37.0
    assert summary.min_temp == 36.0
    assert summary.max_temp == 38.0

    assert pytest.approx(summary.avg_hr, 0.01) == 80.0
    assert summary.min_hr == 70.0
    assert summary.max_hr == 90.0

    assert pytest.approx(summary.avg_spo2, 0.01) == 98.0
    assert summary.min_spo2 == 96.0
    assert summary.max_spo2 == 100.0

    assert "Python Health Data Analysis" in summary.formatted_text


def test_compute_vital_summary_empty():
    """Verify compute_vital_summary handles empty DataFrame gracefully."""
    df = pd.DataFrame(columns=["spo2", "heart_rate", "temperature"])
    summary = compute_vital_summary(df)
    assert summary.reading_count == 0
    assert summary.avg_temp is None
    assert "No telemetry readings available" in summary.formatted_text

"""
Data preprocessing, cleaning, and statistical aggregation for IoT vital telemetry.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VitalSummary:
    """Summary statistics aggregated across a telemetry window."""
    reading_count: int
    avg_temp: Optional[float]
    min_temp: Optional[float]
    max_temp: Optional[float]
    avg_hr: Optional[float]
    min_hr: Optional[float]
    max_hr: Optional[float]
    avg_spo2: Optional[float]
    min_spo2: Optional[float]
    max_spo2: Optional[float]
    formatted_text: str


def clean_sensor_data(feeds: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transform raw ThingSpeak feeds into a clean, typed pandas DataFrame.

    Corrects the channel mapping bug found in the original notebook:
      - field1: Oxygen Saturation (SpO2 %)
      - field2: Heart Rate / Pulse (BPM)
      - field3: Body Temperature (°C)
    """
    if not feeds:
        return pd.DataFrame(columns=["created_at", "entry_id", "spo2", "heart_rate", "temperature"])

    df = pd.DataFrame(feeds)

    # Standardize column names
    column_mapping = {
        "field1": "spo2",
        "field2": "heart_rate",
        "field3": "temperature",
        "Pulse(HR)": "heart_rate",
        "Pulse": "heart_rate",
        "Heart Rate": "heart_rate",
        "Temperature": "temperature",
        "Temp": "temperature",
        "SpO2": "spo2",
    }
    df = df.rename(columns=column_mapping)

    # Required columns guarantee
    for col in ["spo2", "heart_rate", "temperature"]:
        if col not in df.columns:
            df[col] = np.nan

    # Parse timestamps
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")

    # Coerce numeric measurements
    for metric in ["spo2", "heart_rate", "temperature"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Drop records where all vitals are null
    df = df.dropna(subset=["spo2", "heart_rate", "temperature"], how="all").reset_index(drop=True)

    return df


def compute_vital_summary(df: pd.DataFrame) -> VitalSummary:
    """
    Compute statistical aggregations (mean, min, max) for cleaned vital signs.
    """
    if df.empty:
        return VitalSummary(
            reading_count=0,
            avg_temp=None, min_temp=None, max_temp=None,
            avg_hr=None, min_hr=None, max_hr=None,
            avg_spo2=None, min_spo2=None, max_spo2=None,
            formatted_text="No telemetry readings available."
        )

    temps = df["temperature"].dropna().tolist()
    hrs = df["heart_rate"].dropna().tolist()
    spo2s = df["spo2"].dropna().tolist()

    avg_temp = float(np.mean(temps)) if temps else None
    min_temp = float(np.min(temps)) if temps else None
    max_temp = float(np.max(temps)) if temps else None

    avg_hr = float(np.mean(hrs)) if hrs else None
    min_hr = float(np.min(hrs)) if hrs else None
    max_hr = float(np.max(hrs)) if hrs else None

    avg_spo2 = float(np.mean(spo2s)) if spo2s else None
    min_spo2 = float(np.min(spo2s)) if spo2s else None
    max_spo2 = float(np.max(spo2s)) if spo2s else None

    # Human-readable formatted summary string
    lines = [
        "📊 Python Health Data Analysis:",
        "-------------------------------",
        f"• Number of readings: {len(df)}"
    ]

    if temps:
        lines.append(
            f"• Body Temperature: Avg {avg_temp:.2f} °C (Min {min_temp:.2f} °C, Max {max_temp:.2f} °C)"
        )
    if hrs:
        lines.append(
            f"• Heart Rate / Pulse: Avg {avg_hr:.1f} BPM (Min {int(min_hr)} BPM, Max {int(max_hr)} BPM)"
        )
    if spo2s:
        lines.append(
            f"• Oxygen Saturation: Avg {avg_spo2:.1f}% (Min {min_spo2:.1f}%, Max {max_spo2:.1f}%)"
        )

    return VitalSummary(
        reading_count=len(df),
        avg_temp=avg_temp, min_temp=min_temp, max_temp=max_temp,
        avg_hr=avg_hr, min_hr=min_hr, max_hr=max_hr,
        avg_spo2=avg_spo2, min_spo2=min_spo2, max_spo2=max_spo2,
        formatted_text="\n".join(lines)
    )

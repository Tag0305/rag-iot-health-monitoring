"""
Unit tests for FAISS vector store indexing and semantic retrieval.
"""

from pathlib import Path
import pytest

from src.config import Settings
from src.vector_store import PatientRAGRetriever, RetrievedCase


def test_vector_store_initialization_and_search(tmp_path):
    """Test building FAISS index and retrieving top-k similar records."""
    # Create small mock patient CSV
    sample_csv = tmp_path / "test_patients.csv"
    sample_csv.write_text(
        "Patient ID,Age,Gender,Heart Rate,Body Temperature,Oxygen Saturation,Systolic Blood Pressure,Diastolic Blood Pressure,Risk Category,Textual Data\n"
        "PID-101,25,Male,70,36.6,99.0,120,80,Low,\"Patient ID: PID-101, Age: 25, Gender: Male, Heart Rate: 70 bpm, Temp: 36.60 C, SpO2: 99.00%, BP: 120/80, Risk: Low\"\n"
        "PID-102,68,Female,110,38.8,91.0,145,95,High,\"Patient ID: PID-102, Age: 68, Gender: Female, Heart Rate: 110 bpm, Temp: 38.80 C, SpO2: 91.00%, BP: 145/95, Risk: High\"\n"
        "PID-103,45,Male,82,37.2,97.5,128,84,Low,\"Patient ID: PID-103, Age: 45, Gender: Male, Heart Rate: 82 bpm, Temp: 37.20 C, SpO2: 97.50%, BP: 128/84, Risk: Low\"\n",
        encoding="utf-8"
    )

    settings = Settings(patient_records_path=sample_csv, top_k_retrieval=2)
    retriever = PatientRAGRetriever(settings=settings)
    retriever.initialize(records_path=sample_csv)

    assert retriever.index is not None
    assert retriever.index.ntotal == 3
    assert retriever.dimension == 384

    # Query for PID-102 profile
    query = "Patient ID: PID-102, Age: 68, Gender: Female, Heart Rate: 110 bpm, Temp: 38.80 C, SpO2: 91.00%, BP: 145/95, Risk: High"
    results = retriever.search(query, top_k=2)

    assert len(results) == 2
    assert isinstance(results[0], RetrievedCase)
    assert results[0].patient_id == "PID-102"
    assert results[0].risk_category == "High"
    assert results[0].similarity_distance >= 0.0


def test_vector_store_format_for_prompt():
    """Verify prompt formatting utility."""
    cases = [
        RetrievedCase(
            patient_id="PID-201",
            age=40,
            gender="Female",
            heart_rate=75.0,
            body_temperature=36.7,
            spo2=98.5,
            blood_pressure="120/80",
            risk_category="Low",
            textual_data="Sample",
            similarity_distance=0.25,
        )
    ]
    formatted = PatientRAGRetriever.format_for_prompt(cases)
    assert "Case #1 [ID: PID-201]" in formatted
    assert "Distance: 0.250" in formatted
    assert "SpO₂ 98.5%" in formatted

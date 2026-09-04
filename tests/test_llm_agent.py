"""
Unit tests for GeminiAgent prompt formatting and deterministic offline mode.
"""

from src.config import Settings
from src.llm_agent import GeminiAgent
from src.preprocessing import VitalSummary
from src.rules_engine import ClinicalDiagnosis, SeverityLevel
from src.vector_store import RetrievedCase


def test_gemini_agent_mock_initial_summary():
    """Verify deterministic summary generation when running offline."""
    agent = GeminiAgent(settings=Settings(gemini_api_key=""))
    summary = VitalSummary(
        reading_count=50,
        avg_temp=38.2, min_temp=37.5, max_temp=38.9,
        avg_hr=105.0, min_hr=90.0, max_hr=120.0,
        avg_spo2=93.5, min_spo2=91.0, max_spo2=96.0,
        formatted_text="Mock vitals summary"
    )

    output = agent.generate_initial_summary(summary)
    assert "Clinical Telemetry Summary" in output
    assert "pyrexia" in output.lower() or "elevated temperature" in output.lower()
    assert "sub-optimal oxygen saturation" in output.lower()


def test_gemini_agent_mock_rag_analysis():
    """Verify explainable RAG mock synthesis with historical cases."""
    agent = GeminiAgent(settings=Settings(gemini_api_key=""))

    profile = {"Age": 60, "Gender": "Male", "Weight": 75.0, "Height": 1.75}
    summary = VitalSummary(
        reading_count=30,
        avg_temp=37.0, min_temp=36.7, max_temp=37.2,
        avg_hr=75.0, min_hr=70.0, max_hr=80.0,
        avg_spo2=98.0, min_spo2=97.0, max_spo2=99.0,
        formatted_text="Normal summary"
    )
    diagnosis = ClinicalDiagnosis(
        conditions=[],
        severity=SeverityLevel.NORMAL,
        is_abnormal=False,
        explanation="All measured vital signs are within normal clinical thresholds."
    )
    cases = [
        RetrievedCase(
            patient_id="PID-1005",
            age=58,
            gender="Male",
            heart_rate=72.0,
            body_temperature=36.8,
            spo2=98.0,
            blood_pressure="120/80",
            risk_category="Low",
            textual_data="Sample case",
            similarity_distance=0.15,
        )
    ]

    output = agent.generate_rag_analysis(profile, summary, diagnosis, cases)
    assert "Explainable RAG Diagnostic Synthesis" in output
    assert "PID-1005" in output
    assert "NORMAL" in output


def test_gemini_agent_mock_chat():
    """Verify conversational fallback responses for specific queries."""
    agent = GeminiAgent(settings=Settings(gemini_api_key=""))
    summary = VitalSummary(
        reading_count=10,
        avg_temp=38.0, min_temp=37.8, max_temp=38.2,
        avg_hr=85.0, min_hr=80.0, max_hr=90.0,
        avg_spo2=94.0, min_spo2=93.0, max_spo2=95.0,
        formatted_text="Vitals text"
    )
    diagnosis = ClinicalDiagnosis()

    reply_fever = agent._mock_chat_response("Do I have a fever?", summary, diagnosis)
    assert "38.0" in reply_fever
    assert "fever" in reply_fever.lower()

    reply_spo2 = agent._mock_chat_response("What is my oxygen level?", summary, diagnosis)
    assert "94.0" in reply_spo2
    assert "oxygen" in reply_spo2.lower()

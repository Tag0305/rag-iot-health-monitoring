"""
Gemini LLM agent for clinical synthesis, RAG-grounded explanations, and conversational advice.
Includes deterministic offline fallbacks for key-less and test environments.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config import Settings, default_settings
from src.preprocessing import VitalSummary
from src.rules_engine import ClinicalDiagnosis
from src.vector_store import RetrievedCase

logger = logging.getLogger(__name__)


class GeminiAgent:
    """
    LLM reasoning agent utilizing Gemini (e.g. gemini-2.0-flash) with dynamic RAG grounding.
    Falls back gracefully to deterministic clinical synthesis if API keys are absent or offline.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.model = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialize Google GenAI client if valid API key is present."""
        if self.settings.has_gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.settings.gemini_api_key)
                self.model = genai.GenerativeModel(self.settings.gemini_model)
                logger.info("Configured Gemini model: %s", self.settings.gemini_model)
            except Exception as e:
                logger.warning("Failed to configure Google GenAI client: %s. Using mock fallback.", e)
                self.model = None
        else:
            logger.info("No Gemini API key supplied. Running in deterministic mock mode.")
            self.model = None

    def generate_initial_summary(self, summary: VitalSummary) -> str:
        """Generate an initial executive summary from raw sensor telemetry."""
        if self.model:
            prompt = (
                "You are an expert AI clinical monitoring assistant.\n"
                "Analyze the following aggregated IoT health sensor telemetry:\n\n"
                f"{summary.formatted_text}\n\n"
                "Provide a concise clinical overview covering:\n"
                "1. Baseline physiological stability.\n"
                "2. Any detected vital sign anomalies.\n"
                "3. Immediate monitoring recommendations."
            )
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
            except Exception as err:
                logger.warning("Gemini call failed (%s). Falling back to mock summary.", err)

        # Deterministic fallback summary
        return self._mock_initial_summary(summary)

    def generate_rag_analysis(
        self,
        profile: Dict[str, Any],
        summary: VitalSummary,
        diagnosis: ClinicalDiagnosis,
        retrieved_cases: List[RetrievedCase],
    ) -> str:
        """Generate an explainable, RAG-grounded diagnostic synthesis."""
        from src.vector_store import PatientRAGRetriever
        formatted_retrieval = PatientRAGRetriever.format_for_prompt(retrieved_cases)

        prompt = (
            "You are an AI Clinical Decision Support Assistant.\n\n"
            "=== PATIENT DEMOGRAPHICS ===\n"
            f"Age: {profile.get('Age', 'N/A')}, Gender: {profile.get('Gender', 'N/A')}, "
            f"Weight: {profile.get('Weight', 'N/A')} kg, Height: {profile.get('Height', 'N/A')} m\n\n"
            "=== OBSERVED SENSOR TELEMETRY (AVERAGES) ===\n"
            f"Temperature: {summary.avg_temp:.2f} °C, Heart Rate: {summary.avg_hr:.0f} BPM, SpO₂: {summary.avg_spo2:.1f}%\n\n"
            "=== RULE-BASED TRIAGE FINDINGS ===\n"
            f"Severity: {diagnosis.severity.value}\n"
            f"{diagnosis.explanation}\n\n"
            "=== RETRIEVED SIMILAR HISTORICAL PATIENT CASES (VECTOR SEARCH) ===\n"
            f"{formatted_retrieval}\n\n"
            "Task:\n"
            "1. Synthesize the patient's current clinical state.\n"
            "2. Correlate findings with the retrieved historical cases (identifying common trajectories or risks).\n"
            "3. Provide non-prescriptive, explainable guidance and next diagnostic steps."
        )

        if self.model:
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
            except Exception as err:
                logger.warning("Gemini RAG call failed (%s). Falling back to mock synthesis.", err)

        return self._mock_rag_analysis(profile, summary, diagnosis, retrieved_cases)

    def chat_response(
        self,
        user_message: str,
        chat_history: List[Tuple[str, str]],
        profile: Dict[str, Any],
        summary: VitalSummary,
        diagnosis: ClinicalDiagnosis,
        retrieved_cases: List[RetrievedCase],
    ) -> str:
        """Provide conversational, grounded answers to patient inquiries."""
        from src.vector_store import PatientRAGRetriever
        formatted_retrieval = PatientRAGRetriever.format_for_prompt(retrieved_cases)

        conversation_history_text = "\n".join([
            f"User: {u}\nAssistant: {a}" for u, a in chat_history[-3:]
        ])

        prompt = (
            "You are a friendly, medically grounded health assistant.\n"
            f"Current Patient: {profile}\n"
            f"Current Vitals: Temp {summary.avg_temp:.1f}°C, HR {summary.avg_hr:.0f} bpm, SpO2 {summary.avg_spo2:.1f}%\n"
            f"Rule Findings: {diagnosis.explanation}\n"
            f"Retrieved Historical Context:\n{formatted_retrieval}\n\n"
            f"Recent Conversation:\n{conversation_history_text}\n\n"
            f"User Query: {user_message}\n\n"
            "Answer clearly and empathetically. Highlight vital trends and remind the user to seek professional medical care for acute symptoms."
        )

        if self.model:
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
            except Exception as err:
                logger.warning("Gemini chat call failed (%s). Using mock response.", err)

        return self._mock_chat_response(user_message, summary, diagnosis)

    @staticmethod
    def _mock_initial_summary(summary: VitalSummary) -> str:
        """Deterministic mock executive summary."""
        temp = summary.avg_temp or 37.0
        hr = summary.avg_hr or 75.0
        spo2 = summary.avg_spo2 or 98.0

        anomalies = []
        if temp > 37.5:
            anomalies.append(f"pyrexia / elevated temperature ({temp:.1f} °C)")
        if spo2 < 95.0:
            anomalies.append(f"sub-optimal oxygen saturation ({spo2:.1f}%)")
        if hr > 100:
            anomalies.append(f"elevated resting pulse ({hr:.0f} BPM)")

        if anomalies:
            status = f"telemetry flags mild-to-moderate abnormalities: {', '.join(anomalies)}."
        else:
            status = "all vital signs demonstrate consistent physiological stability within normal reference ranges."

        return (
            f"### Automated Clinical Telemetry Summary (Deterministic Engine)\n"
            f"• **Physiological State**: Over {summary.reading_count} readings, {status}\n"
            f"• **Temperature Trend**: {temp:.2f} °C (Min {summary.min_temp:.2f} °C / Max {summary.max_temp:.2f} °C)\n"
            f"• **Cardiovascular**: Pulse avg {hr:.0f} BPM with SpO₂ avg {spo2:.1f}%\n"
            f"• **Guidance**: Continue passive telemetry logging. If symptoms persist or SpO₂ drops below 94%, consult a physician."
        )

    @staticmethod
    def _mock_rag_analysis(
        profile: Dict[str, Any],
        summary: VitalSummary,
        diagnosis: ClinicalDiagnosis,
        retrieved_cases: List[RetrievedCase],
    ) -> str:
        """Deterministic mock RAG synthesis grounded in retrieved cases."""
        top_case_str = (
            f"Patient {retrieved_cases[0].patient_id} (Risk: {retrieved_cases[0].risk_category})"
            if retrieved_cases else "historical control cohort"
        )

        return (
            f"### Explainable RAG Diagnostic Synthesis\n\n"
            f"**1. Patient Assessment**: For a {profile.get('Age', 50)}-year-old {profile.get('Gender', 'patient')}, "
            f"current monitoring reveals an average temperature of {summary.avg_temp:.2f} °C, "
            f"heart rate of {summary.avg_hr:.0f} BPM, and SpO₂ of {summary.avg_spo2:.1f}%.\n\n"
            f"**2. Rule Engine Classification**: **[{diagnosis.severity.value}]**\n"
            f"{diagnosis.explanation}\n\n"
            f"**3. Historical Cohort Grounding (FAISS Retrieval)**:\n"
            f"Semantic similarity matching against {len(retrieved_cases)} relevant patient records identifies highest correlation with {top_case_str}. "
            f"Patients presenting with similar physiological parameters typically require monitoring for respiratory fatigue and fluid balance.\n\n"
            f"**4. Recommendation**: Maintain continuous SpO₂ tracking and rest. Seek clinical triage if shortness of breath or persistent fever occurs."
        )

    @staticmethod
    def _mock_chat_response(
        query: str,
        summary: VitalSummary,
        diagnosis: ClinicalDiagnosis,
    ) -> str:
        """Deterministic fallback chat responses."""
        q = query.lower()
        if "fever" in q or "temp" in q:
            return (
                f"Your recorded average temperature is {summary.avg_temp:.1f} °C. "
                "Temperatures above 37.5 °C are considered a fever. Ensure adequate hydration and monitor for chills or body aches."
            )
        if "spo2" in q or "oxygen" in q:
            return (
                f"Your average blood oxygen saturation is {summary.avg_spo2:.1f}%. "
                "Normal resting values are generally between 95% and 100%. If you experience breathlessness, consult medical personnel immediately."
            )
        if "heart" in q or "pulse" in q:
            return (
                f"Your average heart rate is {summary.avg_hr:.0f} BPM. "
                "Normal adult resting heart rate spans 60-100 BPM. Resting and deep breathing can help evaluate if fluctuations are stress-induced."
            )
        return (
            f"Based on your monitored vitals (Temp: {summary.avg_temp:.1f} °C, HR: {summary.avg_hr:.0f} BPM, SpO₂: {summary.avg_spo2:.1f}%) "
            f"and current rule status [{diagnosis.severity.value}], your vitals are actively tracked. "
            "Please let me know if you are feeling any specific symptoms such as dizziness, fatigue, or chest tightness."
        )

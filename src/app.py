"""
Interactive Web Application for RAG-Enhanced IoT Health Monitoring.
Built with Gradio, featuring patient profile controls, telemetry inspection,
rule-based clinical alerts, FAISS historical case retrieval, and conversational AI.
"""

import logging

from src.config import Settings, default_settings
from src.iot_client import ThingSpeakClient
from src.llm_agent import GeminiAgent
from src.preprocessing import clean_sensor_data, compute_vital_summary
from src.rules_engine import RulesEngine, SeverityLevel
from src.vector_store import PatientRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthMonitoringApp:
    """End-to-end application orchestrator."""

    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        self.iot_client = ThingSpeakClient(settings)
        self.rules_engine = RulesEngine(settings.thresholds)
        self.retriever = PatientRAGRetriever(settings)
        self.agent = GeminiAgent(settings)
        self._retriever_ready = False

    def ensure_retriever(self) -> None:
        """Lazily initialize vector store."""
        if not self._retriever_ready:
            self.retriever.initialize()
            self._retriever_ready = True

    def process_patient_pipeline(
        self,
        age: int,
        gender: str,
        weight: float,
        height: float,
        channel_id: str,
        read_key: str,
        force_mock: bool,
    ) -> tuple[str, str, str, str, str]:
        """
        Execute full clinical monitoring pipeline:
        1. Ingest telemetry from ThingSpeak or mock fallback.
        2. Preprocess and aggregate vitals.
        3. Evaluate deterministic clinical rules.
        4. Perform FAISS semantic retrieval on patient cohort.
        5. Generate Gemini initial & RAG explainable syntheses.
        """
        self.ensure_retriever()

        profile = {
            "Age": int(age),
            "Gender": str(gender),
            "Weight": float(weight),
            "Height": float(height),
        }

        # 1. Fetch sensor feeds
        feeds, source = self.iot_client.fetch_feeds(
            channel_id=channel_id,
            read_api_key=read_key,
            force_mock=force_mock,
        )

        # 2. Clean & summarize vitals
        df_cleaned = clean_sensor_data(feeds)
        summary = compute_vital_summary(df_cleaned)

        # 3. Rule-based evaluation
        diagnosis = self.rules_engine.evaluate_summary(summary)

        # 4. Vector similarity search
        query = (
            f"Patient Age {age} {gender}, Temp {summary.avg_temp:.2f} C, "
            f"Pulse {summary.avg_hr:.0f} bpm, SpO2 {summary.avg_spo2:.1f}%, "
            f"Conditions: {', '.join(diagnosis.conditions) or 'Normal'}"
        )
        retrieved_cases = self.retriever.search(query, top_k=self.settings.top_k_retrieval)

        # 5. Gemini syntheses
        initial_summary = self.agent.generate_initial_summary(summary)
        rag_analysis = self.agent.generate_rag_analysis(
            profile=profile,
            summary=summary,
            diagnosis=diagnosis,
            retrieved_cases=retrieved_cases,
        )

        # Format display components
        telemetry_display = (
            f"**Telemetry Source**: `{source}`\n\n"
            f"{summary.formatted_text}"
        )

        severity_badge = {
            SeverityLevel.NORMAL: "🟢 NORMAL",
            SeverityLevel.WARNING: "🟡 WARNING",
            SeverityLevel.CRITICAL: "🔴 CRITICAL",
        }.get(diagnosis.severity, "⚪ UNKNOWN")

        rules_display = (
            f"### Triage Classification: {severity_badge}\n\n"
            f"**Risk Flags Triggered:**\n"
            f"{diagnosis.explanation}"
        )

        retrieval_display = PatientRAGRetriever.format_for_prompt(retrieved_cases)

        return (
            telemetry_display,
            rules_display,
            retrieval_display,
            initial_summary,
            rag_analysis,
        )

    def handle_chat(
        self,
        message: str,
        chat_history: list[tuple[str, str]],
        age: int,
        gender: str,
        weight: float,
        height: float,
    ) -> tuple[list[tuple[str, str]], str]:
        """Handle conversational interactions grounded in current session context."""
        if not message.strip():
            return chat_history, ""

        self.ensure_retriever()
        profile = {"Age": age, "Gender": gender, "Weight": weight, "Height": height}

        # Use current fallback vitals for context
        feeds, _ = self.iot_client.fetch_feeds(force_mock=True)
        df_cleaned = clean_sensor_data(feeds)
        summary = compute_vital_summary(df_cleaned)
        diagnosis = self.rules_engine.evaluate_summary(summary)
        retrieved = self.retriever.search(f"Age {age} {gender} vitals query", top_k=2)

        bot_reply = self.agent.chat_response(
            user_message=message,
            chat_history=chat_history or [],
            profile=profile,
            summary=summary,
            diagnosis=diagnosis,
            retrieved_cases=retrieved,
        )

        updated_history = (chat_history or []) + [(message, bot_reply)]
        return updated_history, ""


def create_gradio_ui(app: HealthMonitoringApp):
    """Construct modern Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="RAG-Enhanced IoT Health Monitoring") as demo:
        gr.Markdown(
            "# 🩺 RAG-Enhanced IoT Health Monitoring & Explainable Feedback System\n"
            "**Continuous IoT Sensor Telemetry • Deterministic Clinical Rules • FAISS RAG • Gemini 3.8 Flash**"
        )
        gr.Markdown(
            "> ⚠️ **Disclaimer**: Research and educational demonstration only. "
            "This application is not a medical device and is not intended to diagnose, "
            "treat, or replace professional medical advice."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Patient Demographics")
                age_input = gr.Slider(minimum=1, maximum=110, value=53, step=1, label="Age (years)")
                gender_input = gr.Radio(["Male", "Female", "Other"], value="Female", label="Gender")
                weight_input = gr.Number(value=65.0, label="Weight (kg)")
                height_input = gr.Number(value=1.55, label="Height (m)")

                gr.Markdown("### 📡 ThingSpeak IoT Connection")
                channel_input = gr.Textbox(
                    value=app.settings.thingspeak_channel_id,
                    label="Channel ID (leave empty for mock)",
                    placeholder="e.g. 2684921",
                )
                key_input = gr.Textbox(
                    value="",
                    label="Read API Key (leave empty for mock)",
                    type="password",
                )
                force_mock_chk = gr.Checkbox(
                    value=not app.settings.has_thingspeak_creds,
                    label="Use Local Telemetry Simulator (Offline Mode)",
                )

                analyze_btn = gr.Button("⚡ Run Telemetry & RAG Analysis", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("📊 Telemetry & Rules"):
                    telemetry_box = gr.Markdown("Click **'Run Telemetry & RAG Analysis'** to ingest data.")
                    rules_box = gr.Markdown("")

                with gr.Tab("🔍 Vector Retrieval (FAISS)"):
                    gr.Markdown("#### Semantically Similar Synthetic/Sample Records")
                    retrieval_box = gr.Markdown("Historical cases will appear after analysis.")

                with gr.Tab("🧠 Gemini Explainable AI"):
                    gr.Markdown("#### Automated Health Summary")
                    ai_summary_box = gr.Markdown("")
                    gr.Markdown("#### Grounded RAG Decision Support")
                    rag_output_box = gr.Markdown("")

        gr.Markdown("---")
        gr.Markdown("### 💬 Conversational Health Assistant")
        chatbot = gr.Chatbot(height=300)
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about current vitals, symptoms, or risks...",
                scale=8,
                show_label=False,
            )
            send_btn = gr.Button("Send", scale=1, variant="secondary")

        # Event handlers
        analyze_btn.click(
            fn=app.process_patient_pipeline,
            inputs=[
                age_input, gender_input, weight_input, height_input,
                channel_input, key_input, force_mock_chk,
            ],
            outputs=[telemetry_box, rules_box, retrieval_box, ai_summary_box, rag_output_box],
        )

        send_btn.click(
            fn=app.handle_chat,
            inputs=[msg_input, chatbot, age_input, gender_input, weight_input, height_input],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=app.handle_chat,
            inputs=[msg_input, chatbot, age_input, gender_input, weight_input, height_input],
            outputs=[chatbot, msg_input],
        )

    return demo


def main():
    """Application entry point."""
    app = HealthMonitoringApp()
    try:
        demo = create_gradio_ui(app)
        demo.launch(
            server_name=app.settings.server_host,
            server_port=app.settings.server_port,
            share=False,
        )
    except ImportError:
        logger.warning("Gradio is not installed. Running CLI demonstration mode...")
        telemetry, rules, retrieval, summary, rag = app.process_patient_pipeline(
            age=53, gender="Female", weight=65.0, height=1.55,
            channel_id="", read_key="", force_mock=True
        )
        print("\n" + "=" * 60)
        print(telemetry)
        print("\n" + "=" * 60)
        print(rules)
        print("\n" + "=" * 60)
        print("RETRIEVAL CONTEXT:")
        print(retrieval)
        print("\n" + "=" * 60)
        print(summary)
        print("\n" + "=" * 60)
        print(rag)
        print("=" * 60)


if __name__ == "__main__":
    main()

"""
Hugging Face Spaces entry point for RAG-Enhanced IoT Health Monitoring.
"""

from src.app import HealthMonitoringApp, create_gradio_ui

health_app = HealthMonitoringApp()
demo = create_gradio_ui(health_app)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

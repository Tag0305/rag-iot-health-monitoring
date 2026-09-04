"""
Smoke test for root app and Gradio UI initialization.
"""

from src.app import HealthMonitoringApp, create_gradio_ui


def test_app_and_ui_initialization():
    """Verify that HealthMonitoringApp and Gradio UI initialize without errors."""
    health_app = HealthMonitoringApp()
    demo = create_gradio_ui(health_app)

    assert demo is not None
    # Verify Gradio Blocks instance
    assert hasattr(demo, "launch")
    assert hasattr(demo, "title")
    assert "RAG-Enhanced IoT Health Monitoring" in demo.title

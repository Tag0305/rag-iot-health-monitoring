"""
Unit tests for ThingSpeakClient live query and mock/offline fallback.
"""

from src.config import Settings
from src.iot_client import ThingSpeakClient


def test_iot_client_mock_mode():
    """Verify that force_mock returns valid feeds with required telemetry keys."""
    client = ThingSpeakClient()
    feeds, source = client.fetch_feeds(force_mock=True, results=10)

    assert len(feeds) == 10
    assert "mock" in source.lower()

    for item in feeds:
        assert "created_at" in item
        assert "entry_id" in item
        assert "field1" in item  # SpO2
        assert "field2" in item  # Pulse
        assert "field3" in item  # Temperature


def test_iot_client_missing_creds_fallback():
    """Verify that omitting credentials automatically triggers fallback."""
    empty_settings = Settings(thingspeak_channel_id="", thingspeak_read_api_key="")
    client = ThingSpeakClient(settings=empty_settings)

    feeds, source = client.fetch_feeds(results=5)
    assert len(feeds) == 5
    assert "mock" in source.lower()


def test_iot_client_synthetic_generation():
    """Verify standalone synthetic feed generation."""
    synthetic = ThingSpeakClient._generate_synthetic_feeds(count=15)
    assert len(synthetic) == 15
    first = synthetic[0]
    assert float(first["field1"]) > 0
    assert float(first["field2"]) > 0
    assert float(first["field3"]) > 0

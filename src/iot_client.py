"""
ThingSpeak IoT Client for fetching live and fallback sensor telemetry streams.
"""

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from src.config import Settings, default_settings

logger = logging.getLogger(__name__)


class ThingSpeakClient:
    """
    Client for retrieving sensor telemetry from ThingSpeak IoT channels,
    with an integrated deterministic fallback/mock engine for offline evaluation.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings

    def fetch_feeds(
        self,
        channel_id: str | None = None,
        read_api_key: str | None = None,
        results: int | None = None,
        force_mock: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Fetch feeds from ThingSpeak. If credentials are missing, force_mock is True,
        or the HTTP request fails, returns deterministic mock telemetry.

        Returns:
            Tuple of (feeds_list, source_description)
        """
        channel = (channel_id or self.settings.thingspeak_channel_id or "").strip()
        api_key = (read_api_key or self.settings.thingspeak_read_api_key or "").strip()
        count = results or self.settings.thingspeak_results_count

        if not force_mock and channel and api_key:
            url = f"https://api.thingspeak.com/channels/{channel}/feeds.json?api_key={api_key}&results={count}"
            try:
                response = requests.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    feeds = data.get("feeds", [])
                    if feeds:
                        return feeds, f"live (ThingSpeak channel {channel})"
                logger.warning(
                    "ThingSpeak request returned HTTP %s. Falling back to mock telemetry.",
                    response.status_code
                )
            except requests.RequestException as e:
                logger.warning(
                    "Failed to connect to ThingSpeak (%s). Falling back to mock telemetry.",
                    str(e)
                )

        # Fallback to local sample or synthetic generation
        return self._load_fallback_feeds(count), "mock (local telemetry simulation)"

    def _load_fallback_feeds(self, count: int) -> list[dict[str, Any]]:
        """Load feeds from local sample JSON if available, else generate synthetic vitals."""
        sample_path: Path = self.settings.sample_feeds_path
        if sample_path.exists():
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    feeds = data.get("feeds", [])
                    if feeds:
                        return feeds[-count:] if len(feeds) > count else feeds
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as err:
                logger.error("Error reading sample_iot_feeds.json: %s", err)

        # Generate on-the-fly synthetic data if file is missing
        return self._generate_synthetic_feeds(count)

    @staticmethod
    def _generate_synthetic_feeds(count: int = 50) -> list[dict[str, Any]]:
        """Generate deterministic, realistic IoT vital signs."""
        random_gen = random.Random(42)
        start_time = datetime.now(timezone.utc) - timedelta(seconds=count * 20)
        feeds = []
        for i in range(1, count + 1):
            ts = (start_time + timedelta(seconds=i * 20)).strftime("%Y-%m-%dT%H:%M:%SZ")
            spo2 = round(random_gen.uniform(95.0, 99.0), 1)
            hr = int(random_gen.uniform(68, 88))
            temp = round(random_gen.uniform(36.5, 37.3), 1)
            feeds.append({
                "created_at": ts,
                "entry_id": i,
                "field1": str(spo2),
                "field2": str(hr),
                "field3": str(temp)
            })
        return feeds

"""
Configuration management and clinical thresholds for IoT Health Monitoring.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class VitalThresholds:
    """Configurable clinical thresholds for vital sign anomaly detection."""
    # Temperature (°C)
    fever_temp_c: float = 37.5
    high_fever_temp_c: float = 38.0
    hypothermia_temp_c: float = 35.0
    
    # Heart Rate (BPM)
    tachycardia_hr: int = 100
    bradycardia_hr: int = 60
    
    # Oxygen Saturation (%)
    hypoxemia_spo2: float = 95.0
    severe_hypoxemia_spo2: float = 90.0


@dataclass
class Settings:
    """Application-wide settings and credentials."""
    # ThingSpeak IoT Configuration
    thingspeak_channel_id: str = field(
        default_factory=lambda: os.getenv("THINGSPEAK_CHANNEL_ID", "")
    )
    thingspeak_read_api_key: str = field(
        default_factory=lambda: os.getenv("THINGSPEAK_READ_API_KEY", "")
    )
    thingspeak_results_count: int = 100
    
    # Gemini AI Configuration
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )
    
    # RAG & Embedding Configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k_retrieval: int = 3
    
    # File Paths
    data_dir: Path = DATA_DIR
    sample_feeds_path: Path = DATA_DIR / "sample_iot_feeds.json"
    patient_records_path: Path = DATA_DIR / "patient_vital_records.csv"
    
    # Clinical Thresholds
    thresholds: VitalThresholds = field(default_factory=VitalThresholds)
    
    # Web App Server Settings
    server_host: str = field(
        default_factory=lambda: os.getenv("HOST", "127.0.0.1")
    )
    server_port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "7860"))
    )

    @property
    def has_thingspeak_creds(self) -> bool:
        """Check if active ThingSpeak credentials are provided."""
        return bool(self.thingspeak_channel_id.strip() and self.thingspeak_read_api_key.strip())

    @property
    def has_gemini_api_key(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.gemini_api_key.strip() and self.gemini_api_key != "your_gemini_api_key_here")


# Global default settings instance
default_settings = Settings()

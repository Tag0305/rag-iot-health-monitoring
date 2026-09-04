"""
FAISS Vector Store and Semantic Retrieval Engine for historical patient cohorts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, default_settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCase:
    """Structure for a retrieved historical patient case."""
    patient_id: str
    age: int
    gender: str
    heart_rate: float
    body_temperature: float
    spo2: float
    blood_pressure: str
    risk_category: str
    textual_data: str
    similarity_distance: float


class PatientRAGRetriever:
    """
    RAG Retriever utilizing HuggingFace sentence-transformers and FAISS
    IndexFlatL2 to retrieve clinically relevant historical patient cohorts.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.model = None
        self.index = None
        self.df_records: pd.DataFrame | None = None
        self.dimension: int | None = None
        self._is_initialized = False

    def initialize(self, records_path: str | Path | None = None) -> None:
        """Load records, load sentence-transformer model, and build FAISS index."""
        if self._is_initialized:
            return

        target_path = Path(records_path or self.settings.patient_records_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Patient records file not found at: {target_path}")

        self.df_records = pd.read_csv(target_path)
        if "Textual Data" not in self.df_records.columns:
            # Generate textual representation if not explicitly present
            self.df_records["Textual Data"] = [
                f"Patient ID: {row.get('Patient ID', 'N/A')}, Age: {row.get('Age', 'N/A')}, "
                f"Gender: {row.get('Gender', 'N/A')}, Heart Rate: {row.get('Heart Rate', 'N/A')} bpm, "
                f"Temp: {float(row.get('Body Temperature', 0)):.2f} C, "
                f"SpO2: {float(row.get('Oxygen Saturation', 0)):.2f}%, "
                f"BP: {row.get('Systolic Blood Pressure', 'N/A')}/{row.get('Diastolic Blood Pressure', 'N/A')}, "
                f"Risk: {row.get('Risk Category', 'N/A')}"
                for _, row in self.df_records.iterrows()
            ]

        # Lazy import of heavy ML libraries to keep lightweight operations fast
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.settings.embedding_model_name)
        self.model = SentenceTransformer(self.settings.embedding_model_name)

        documents = self.df_records["Textual Data"].tolist()
        logger.info("Generating embeddings for %d records...", len(documents))
        embeddings = self.model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        self._is_initialized = True
        logger.info("FAISS vector store successfully initialized with %d entries.", self.index.ntotal)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedCase]:
        """
        Perform vector similarity search against the historical patient cohort.
        """
        if not self._is_initialized:
            self.initialize()

        k = top_k or self.settings.top_k_retrieval
        k = min(k, len(self.df_records))

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results: list[RetrievedCase] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.df_records):
                continue
            row = self.df_records.iloc[idx]
            bp = f"{row.get('Systolic Blood Pressure', 'N/A')}/{row.get('Diastolic Blood Pressure', 'N/A')}"
            results.append(RetrievedCase(
                patient_id=str(row.get("Patient ID", f"PID-{idx}")),
                age=int(row.get("Age", 0)),
                gender=str(row.get("Gender", "Unknown")),
                heart_rate=float(row.get("Heart Rate", 0.0)),
                body_temperature=float(row.get("Body Temperature", 0.0)),
                spo2=float(row.get("Oxygen Saturation", 0.0)),
                blood_pressure=bp,
                risk_category=str(row.get("Risk Category", "Unknown")),
                textual_data=str(row.get("Textual Data", "")),
                similarity_distance=float(dist),
            ))

        return results

    @staticmethod
    def format_for_prompt(cases: list[RetrievedCase]) -> str:
        """Format retrieved cases for LLM grounding context."""
        if not cases:
            return "No historical reference records retrieved."

        formatted_cases = []
        for i, case in enumerate(cases, 1):
            formatted_cases.append(
                f"Case #{i} [ID: {case.patient_id}] (Distance: {case.similarity_distance:.3f}):\n"
                f"  • Demographics: Age {case.age}, Gender: {case.gender}\n"
                f"  • Vitals: Temp {case.body_temperature:.2f} °C, Pulse {case.heart_rate:.0f} BPM, SpO₂ {case.spo2:.1f}%\n"
                f"  • Blood Pressure: {case.blood_pressure} mmHg | Clinical Risk: {case.risk_category}"
            )
        return "\n\n".join(formatted_cases)

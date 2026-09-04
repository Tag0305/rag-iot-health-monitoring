"""
Unit tests for the deterministic clinical rules engine and risk classification.
"""

from src.config import VitalThresholds
from src.rules_engine import RulesEngine, SeverityLevel


def test_rules_engine_normal_vitals():
    """Verify that normal vitals produce NORMAL severity and no conditions."""
    engine = RulesEngine()
    diagnosis = engine.evaluate(temperature=36.8, heart_rate=72.0, spo2=98.5)

    assert diagnosis.severity == SeverityLevel.NORMAL
    assert not diagnosis.is_abnormal
    assert len(diagnosis.conditions) == 0
    assert "within normal clinical thresholds" in diagnosis.explanation


def test_rules_engine_fever():
    """Verify fever detection for temp > 37.5 C."""
    engine = RulesEngine()
    diagnosis = engine.evaluate(temperature=38.2, heart_rate=75.0, spo2=98.0)

    assert diagnosis.is_abnormal
    assert diagnosis.severity in (SeverityLevel.WARNING, SeverityLevel.CRITICAL)
    assert any("Fever detected" in c for c in diagnosis.conditions)


def test_rules_engine_hypothermia():
    """Verify hypothermia warning for temp < 35.0 C."""
    engine = RulesEngine()
    diagnosis = engine.evaluate(temperature=34.5, heart_rate=65.0, spo2=97.0)

    assert diagnosis.is_abnormal
    assert diagnosis.severity == SeverityLevel.CRITICAL
    assert any("Hypothermia warning" in c for c in diagnosis.conditions)


def test_rules_engine_hypoxemia():
    """Verify mild and severe hypoxemia conditions."""
    engine = RulesEngine()

    # Mild hypoxemia (90 <= spo2 < 95)
    diag_mild = engine.evaluate(temperature=36.7, heart_rate=75.0, spo2=93.0)
    assert diag_mild.is_abnormal
    assert any("Low SpO₂" in c for c in diag_mild.conditions)

    # Severe hypoxemia (spo2 < 90)
    diag_severe = engine.evaluate(temperature=36.7, heart_rate=75.0, spo2=88.0)
    assert diag_severe.is_abnormal
    assert diag_severe.severity == SeverityLevel.CRITICAL
    assert any("Severe hypoxemia" in c for c in diag_severe.conditions)


def test_rules_engine_tachycardia_and_bradycardia():
    """Verify heart rate boundary checks."""
    engine = RulesEngine()

    # Tachycardia (>100 bpm)
    diag_tachy = engine.evaluate(temperature=36.7, heart_rate=115.0, spo2=98.0)
    assert any("Tachycardia" in c for c in diag_tachy.conditions)

    # Bradycardia (<60 bpm)
    diag_brady = engine.evaluate(temperature=36.7, heart_rate=52.0, spo2=98.0)
    assert any("Bradycardia" in c for c in diag_brady.conditions)


def test_rules_engine_compound_rules():
    """Verify multi-vital interaction detection."""
    engine = RulesEngine()

    # Fever + Low SpO2 -> Respiratory infection warning
    diag_resp = engine.evaluate(temperature=38.1, heart_rate=80.0, spo2=93.0)
    assert any("respiratory infection" in c.lower() for c in diag_resp.conditions)

    # High fever + Tachycardia -> Sepsis / systemic infection risk
    diag_sepsis = engine.evaluate(temperature=38.7, heart_rate=110.0, spo2=97.0)
    assert diag_sepsis.severity == SeverityLevel.CRITICAL
    assert any("sepsis" in c.lower() for c in diag_sepsis.conditions)


def test_rules_engine_custom_thresholds():
    """Verify custom clinical thresholds can be injected."""
    custom_thresholds = VitalThresholds(
        fever_temp_c=37.0,  # Lower fever threshold
        tachycardia_hr=90,  # Lower tachycardia threshold
    )
    engine = RulesEngine(thresholds=custom_thresholds)

    diagnosis = engine.evaluate(temperature=37.2, heart_rate=95.0, spo2=98.0)
    assert diagnosis.is_abnormal
    assert any("Fever detected" in c for c in diagnosis.conditions)
    assert any("Tachycardia" in c for c in diagnosis.conditions)

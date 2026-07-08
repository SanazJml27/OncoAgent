import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncoagent.agents.redaction_agent import RedactionAgent
from oncoagent.agents.safety_agent import SafetyAgent
from oncoagent.orchestrator import OncoAgentOrchestrator
from oncoagent.schemas import PatientIntake, UrgencyLevel


def test_redaction_removes_email_and_phone():
    agent = RedactionAgent()
    text = "Contact me at anna.virtanen@example.com or 0401234567."
    redacted, spans = agent.run(text)
    assert "anna.virtanen@example.com" not in redacted
    assert "0401234567" not in redacted
    assert len(spans) >= 2


def test_safety_agent_flags_emergency():
    agent = SafetyAgent()
    intake = PatientIntake(
        patient_id="t1",
        free_text_notes="Sudden severe chest pain and difficulty breathing since this morning.",
    )
    flags = agent.run(intake)
    urgency = agent.overall_urgency(flags)
    assert urgency == UrgencyLevel.EMERGENCY


def test_safety_agent_routine_when_no_flags():
    agent = SafetyAgent()
    intake = PatientIntake(patient_id="t2", free_text_notes="Feeling generally well, no complaints.")
    flags = agent.run(intake)
    assert agent.overall_urgency(flags) == UrgencyLevel.ROUTINE


def test_full_pipeline_runs_end_to_end():
    orchestrator = OncoAgentOrchestrator()
    result = orchestrator.run(
        patient_id="demo-patient-001",
        cancer_type="breast",
        free_text="My name is Anna Virtanen, phone 0401234567. Fatigue and joint aching. On letrozole.",
    )
    assert result.intake.patient_id == "demo-patient-001"
    assert "letrozole" in result.intake.medications
    assert "0401234567" not in result.intake.redacted_notes
    assert len(result.retrieved_snippets) > 0
    assert result.summary.summary_text  # non-empty
    assert len(result.trace) > 0


def test_pipeline_emergency_case_flags_correctly():
    orchestrator = OncoAgentOrchestrator()
    result = orchestrator.run(
        patient_id="demo-patient-004",
        cancer_type="lung",
        free_text="Sudden severe chest pain and difficulty breathing since this morning, feels like pressure.",
    )
    assert result.summary.urgency == UrgencyLevel.EMERGENCY
    assert any(f.urgency == UrgencyLevel.EMERGENCY for f in result.safety_flags)

#!/usr/bin/env python3
"""
Command-line entry point for OncoAgent.

Usage:
    python cli.py --demo                      # run all synthetic demo patients
    python cli.py --demo-id demo-patient-002   # run a single demo patient
    python cli.py --text "..." --cancer-type lung --patient-id p001 --trace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oncoagent.orchestrator import OncoAgentOrchestrator
from oncoagent.schemas import PipelineResult

DEMO_PATH = Path(__file__).resolve().parent / "data" / "synthetic_patients.json"


def print_result(result: PipelineResult, show_trace: bool) -> None:
    print("=" * 72)
    print(f"Patient: {result.intake.patient_id}  |  Cancer type: {result.intake.cancer_type}")
    print(f"Urgency: {result.summary.urgency.value.upper()}")
    print("-" * 72)
    print("Redacted notes:")
    print(f"  {result.intake.redacted_notes}")
    print(f"  ({len(result.intake.redaction_spans)} PII span(s) redacted)")
    print("-" * 72)
    print("Structured intake:")
    print(f"  Symptoms   : {', '.join(result.intake.reported_symptoms) or '(none)'}")
    print(f"  Medications: {', '.join(result.intake.medications) or '(none)'}")
    print("-" * 72)
    if result.safety_flags:
        print("Safety flags:")
        for f in result.safety_flags:
            print(f"  [{f.urgency.value.upper()}] {f.reason} (matched: '{f.matched_phrase}')")
        print("-" * 72)
    print("Retrieved guidance:")
    for s in result.retrieved_snippets:
        print(f"  ({s.score:.3f}) [{s.topic}] {s.text}")
    print("-" * 72)
    print("Visit summary:")
    print(result.summary.summary_text)
    if result.summary.critic_notes:
        print("\nUnresolved critic notes (surfaced for human review):")
        for n in result.summary.critic_notes:
            print(f"  - {n}")
    print(f"\n(revisions requested by critic: {result.summary.revision_count})")

    if show_trace:
        print("-" * 72)
        print("Agent trace:")
        for event in result.trace:
            print(f"  [{event.timestamp}] {event.agent} :: {event.action} -> {event.detail}")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OncoAgent multi-agent pipeline.")
    parser.add_argument("--demo", action="store_true", help="Run all synthetic demo patients.")
    parser.add_argument("--demo-id", type=str, help="Run a single synthetic demo patient by ID.")
    parser.add_argument("--text", type=str, help="Free-text patient notes to process.")
    parser.add_argument("--cancer-type", type=str, default=None)
    parser.add_argument("--patient-id", type=str, default="cli-patient")
    parser.add_argument("--trace", action="store_true", help="Print the full agent trace.")
    parser.add_argument("--json-out", type=str, help="Also write the full result as JSON to this path.")
    args = parser.parse_args()

    orchestrator = OncoAgentOrchestrator()
    results = []

    if args.demo or args.demo_id:
        with open(DEMO_PATH, "r", encoding="utf-8") as f:
            patients = json.load(f)
        if args.demo_id:
            patients = [p for p in patients if p["patient_id"] == args.demo_id]
            if not patients:
                raise SystemExit(f"No demo patient with id {args.demo_id}")
        for p in patients:
            result = orchestrator.run(p["patient_id"], p.get("cancer_type"), p["free_text_notes"])
            print_result(result, args.trace)
            results.append(result)
    elif args.text:
        result = orchestrator.run(args.patient_id, args.cancer_type, args.text)
        print_result(result, args.trace)
        results.append(result)
    else:
        parser.print_help()
        return

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)
        print(f"Wrote {len(results)} result(s) to {args.json_out}")


if __name__ == "__main__":
    main()

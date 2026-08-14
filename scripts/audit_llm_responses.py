from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.data import parse_llm_response
from demand_sim.research import load_probability_rows_strict, probability_monotonicity_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw LLM simulation outputs for parse failures and price monotonicity."
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.responses)
    parse_ok = raw["response"].map(lambda text: parse_llm_response(text) is not None)
    probabilities, strict_report = load_probability_rows_strict(args.responses)
    monotonicity = probability_monotonicity_audit(probabilities)

    report = {
        "response_rows": int(len(raw)),
        "loosely_parsed_rows": int(parse_ok.sum()),
        "loose_parse_success_rate": float(parse_ok.mean()) if len(raw) else 0.0,
        "strict_validation": strict_report,
        "monotonicity": monotonicity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

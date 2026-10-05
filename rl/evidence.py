"""Evidence-string helpers for the Old Bailey annotated templates.

An evidence item is a `[EVIDENCE evN]` header, a `Description:` line and the annotated ground truth
(`Prosecution/Defense Strength:` + `... Reasoning:`). `parse_strength` reads one side's strength,
the rule for which side an item favors (`rl/game_rows.py`, `rl/monitors.py`). `clean_evidence`
strips the annotations, leaving what the sender is shown; it is the judge's leak guard in
`rl.strategy_audit.audit.classify` and is applied to every template baked into a prompt, so its
output is part of the prompts.
"""


def parse_strength(evidence_str: str, side: str = "prosecution") -> float:
    """Parse the strength value for prosecution or defense from an evidence string."""
    lines = evidence_str.split("\n")
    for line in lines:
        if side.lower() == "prosecution" and line.startswith("Prosecution Strength:"):
            return float(line.split(":")[1].strip())
        elif side.lower() == "defense" and line.startswith("Defense Strength:"):
            return float(line.split(":")[1].strip())
    return 0.0


def clean_evidence(evidence_str: str) -> str:
    """Remove strength and reasoning lines from evidence string."""
    lines = evidence_str.split("\n")
    cleaned_lines = []
    for line in lines:
        if not any(x in line for x in ["Strength:", "Reasoning:"]):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

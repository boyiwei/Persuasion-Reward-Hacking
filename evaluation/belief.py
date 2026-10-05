"""Receiver-belief extraction.

`parse_belief` is the regex P(guilty) extractor shared by the GRPO reward path
(rl/reward_function.py, rl/persuasion_interaction.py), evaluation/rl_rollout.py,
evaluation/audit_fabrications.py and the Old Bailey evaluator; `is_valid_belief` is the shared
validity rule. Stdlib-only; import as `evaluation.belief` with the repo root on sys.path.
"""
import re
from typing import Optional

# --- P(guilty) belief parsing ---
# The receiver states its <belief> in model-specific prose, not a bare float, so float() fails on
# almost every game. Seen in gpt-oss-120B / llama3.3-70B / gemma4-31B:
#   gpt-oss : "P(guilty | evidence) ≈ 0.75", "Guilty: **0.60**", LaTeX "$P(\\text{guilty})$"
#   gemma   : "P(guilty) = 0.7", "Probability of Guilty: 0.3", "Guilty: 0.5"
#   llama   : almost always numberless prose ("more likely to be guilty than innocent")
# Keep the harmony final channel, normalize LaTeX/markdown/unicode, then take the guilty-side number,
# else 1 - P(innocent), else a lone number. Numberless prose returns None.
_FINAL_CHANNEL_RE = re.compile(r"<\|channel\|>final<\|message\|>")
_BELIEF_TAG_RE = re.compile(r"<belief>(.*?)</belief>", re.IGNORECASE | re.DOTALL)
# A probability token: a decimal (0.75, .75) or a percentage (75%, 12.5%). Bare integers
# are excluded on purpose — they are usually noise like "Rounds 1-2", not the posterior.
_NUM = r"(\d{1,3}(?:\.\d+)?\s*%|\d*\.\d+)"
_GUILTY_NUM_RE = re.compile(r"guilty.*?" + _NUM, re.DOTALL)
_INNOCENT_NUM_RE = re.compile(r"innocent.*?" + _NUM, re.DOTALL)
_ANY_NUM_RE = re.compile(_NUM)


def _normalize_belief_text(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    for tok in ("\\text{", "}", "\\(", "\\)", "$", "**", "\\approx", "\\,", "\\;", "\\:", "≈", "~"):
        s = s.replace(tok, " ")
    return s.lower()


def _to_prob(tok: str) -> Optional[float]:
    tok = tok.strip()
    is_pct = tok.endswith("%")
    try:
        v = float(tok.rstrip("%").strip())
    except ValueError:
        return None
    if is_pct:
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


def parse_belief(message: Optional[str]) -> Optional[float]:
    """Extract the receiver's P(guilty) in [0,1] from a raw response, or None. Regex only."""
    if not message:
        return None
    text = message
    if "<|channel|>" in text:  # gpt-oss harmony: keep only the final channel
        matches = list(_FINAL_CHANNEL_RE.finditer(text))
        if matches:
            text = text[matches[-1].end():]
    blocks = _BELIEF_TAG_RE.findall(text)
    if not blocks:
        return None
    block = _normalize_belief_text(blocks[-1])  # the round's final <belief> block
    # 1. P(guilty) stated before the innocent side (the common, unambiguous case).
    head = re.split("innocent", block)[0]
    m = _GUILTY_NUM_RE.search(head)
    if m and (v := _to_prob(m.group(1))) is not None:
        return v
    # 2. Only the innocent side is numeric: P(guilty) = 1 - P(innocent).
    m = _INNOCENT_NUM_RE.search(block)
    if m and (v := _to_prob(m.group(1))) is not None:
        return 1.0 - v
    # 3. A single lone probability in the block (treat as P(guilty), legacy behavior).
    nums = _ANY_NUM_RE.findall(block)
    if len(nums) == 1 and (v := _to_prob(nums[0])) is not None:
        return v
    return None  # numberless prose / ambiguous — not regex-extractable


def is_valid_belief(b) -> bool:
    """True iff `b` is a real number in [0,1]; bools are rejected (`isinstance(True, int)` is True)."""
    return isinstance(b, (int, float)) and not isinstance(b, bool) and 0.0 <= float(b) <= 1.0

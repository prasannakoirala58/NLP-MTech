"""Check every decimal number in DRAFT.html against the values the notebook computed.

Run from anywhere:
    uv run --no-sync python assignments/a5-transformers/check_draft_numbers.py

The notebook writes outputs/verified_values.json (one list of values per question).
Each <section data-q="Qn"> of the draft is checked against that question's values:

  * numbers with 1-3 decimals (final answers) must equal a notebook value rounded to the
    same number of decimals, exactly. This catches a wrong last digit such as 0.504 for 0.503.
  * numbers with 4 or more decimals (working steps, done the paper way with 4-decimal
    inputs) may drift by a hair from full precision, e.g. 8.1694 vs 8.16947.

Whole numbers (shapes, indices, given inputs) and FIGURE box sizes are not checked.
Exit code 0 = every number matched.
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).parent

html = (HERE / "DRAFT.html").read_text(encoding="utf-8")
verified = json.loads((HERE / "outputs" / "verified_values.json").read_text(encoding="utf-8"))

# Drop the script and style blocks: they are not part of what gets copied
html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL)
html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL)
# Drop the FIGURE placeholder boxes: their numbers are box sizes in cm, not maths
html = re.sub(r'<div class="figbox".*?</div></div>', "", html, flags=re.DOTALL)

# A decimal number, optionally with a minus sign glued to it (the draft uses the "−" character)
NUMBER = re.compile(r"(?<![\w.])[−-]?\d+\.\d+")


def round_half_up(value, decimals):
    """Round the way a calculator does (0.5 always goes up), not Python's banker's rounding."""
    step = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(value)).quantize(step, rounding=ROUND_HALF_UP))


def is_verified(text, allowed):
    number = float(text.replace("−", "-"))
    decimals = len(text.split(".")[1])
    for value in allowed:
        if decimals <= 3:
            if round_half_up(value, decimals) == number:
                return True
        else:
            if abs(number - value) <= max(1.5e-4, 6e-5 * abs(value)):
                return True
    return False


sections = re.findall(r'<section data-q="(Q\d+)">(.*?)</section>', html, flags=re.DOTALL)
total_checked = 0
problems = []
for question, body in sections:
    text = re.sub(r"<[^>]+>", " ", body).replace("&nbsp;", " ")
    allowed = verified[question] + verified["constants"]
    numbers = NUMBER.findall(text)
    bad = [n for n in numbers if not is_verified(n, allowed)]
    total_checked += len(numbers)
    status = "ok" if not bad else "MISMATCH"
    print(f"{question:<4} {len(numbers):>3} numbers checked   {status}")
    for n in bad:
        problems.append((question, n))
        print(f"       not found in the notebook: {n}")

print()
if problems:
    print(f"{len(problems)} of {total_checked} numbers did not match. Fix the draft before copying.")
    sys.exit(1)
print(f"All {total_checked} numbers in the draft match the notebook. Safe to copy.")

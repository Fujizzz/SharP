from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
readme = (ROOT / "README.md").read_text(encoding="utf-8")

references = re.findall(r'(?:src="|\]\()([^"\)]+)', readme)
local_references = [
    reference.split("#")[0]
    for reference in references
    if not reference.startswith(("http://", "https://", "#"))
]
missing = [
    reference
    for reference in local_references
    if reference and not (ROOT / reference).exists()
]

fence_count = readme.count("```")
if fence_count % 2:
    raise SystemExit(f"[FAIL] unclosed Markdown code fence: {fence_count}")
if missing:
    raise SystemExit("[FAIL] missing local README references: " + ", ".join(missing))

figures = [
    reference
    for reference in local_references
    if reference.removeprefix("./").startswith("assets/figures/")
]
if len(figures) != 3:
    raise SystemExit(f"[FAIL] expected three architecture figures, found {len(figures)}")

print(f"[OK] README: {len(readme.splitlines())} lines, {len(local_references)} local references")
print(f"[OK] Markdown code fences: {fence_count}")
print("[OK] Architecture figures: " + ", ".join(figures))

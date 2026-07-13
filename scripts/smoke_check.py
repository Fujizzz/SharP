from __future__ import annotations

import ast
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CODE = [
    "DAAL/amodel.py",
    "DAAL/model_adjust.py",
    "DAAL/process_data.py",
    "DAAL/Focal_loss.py",
    "DAAL/Samper.py",
    "DAAL/vocab.txt",
    "data/README.md",
]

REQUIRED_DATA = [
    "data/pheme/processed/source_data.pkl",
    "data/pheme/processed/source_extend_data.pkl",
    "data/pheme/prompt_seed/clippool_16.xlsx",
    "data/pheme/prompt_seed/clippool_50.xlsx",
    "data/liar/raw/train.tsv",
    "data/liar/raw/valid.tsv",
    "data/liar/raw/test.tsv",
    "data/liar/processed/source_data.pkl",
    "data/liar/processed/destination_data.pkl",
    "data/liar/prompt_seed/clippool_16_release.xlsx",
    "data/twitter15_16/raw/twitter15/source_tweets.txt",
    "data/twitter15_16/raw/twitter16/source_tweets.txt",
    "data/twitter15_16/processed/twitter_source_data.pkl",
    "data/twitter15_16/processed/twitter_destination_data.pkl",
    "data/twitter15_16/prompt_seed/clippool_16_release.xlsx",
    "data/resources/lexicons/emotional/nrc.txt",
    "data/resources/lexicons/imageability/imageability.predictions",
]

PHEME_FILES = [
    (0, "Charlie Hebdo", "charliehebdo.csv", 2079),
    (1, "Ferguson", "ferguson.csv", 1143),
    (2, "Germanwings", "germanwings-crash.csv", 469),
    (3, "Ottawa", "ottawashooting.csv", 888),
    (4, "Sydney", "sydneysiege.csv", 1209),
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


missing_code = [path for path in REQUIRED_CODE if not (ROOT / path).is_file()]
if missing_code:
    fail("missing required code-release files: " + ", ".join(missing_code))
print(f"[OK] required code-release files: {len(REQUIRED_CODE)}")

python_files = sorted(
    path for path in ROOT.rglob("*.py") if ".git" not in path.parts and "__pycache__" not in path.parts
)
for path in python_files:
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:
        fail(f"syntax error in {path.relative_to(ROOT)}: {exc}")
print(f"[OK] Python syntax: {len(python_files)} files")

missing_data = [path for path in REQUIRED_DATA if not (ROOT / path).is_file()]
if missing_data:
    main_model = (ROOT / "DAAL" / "model_adjust.py").read_text(encoding="utf-8-sig")
    for marker in ("--less-frac", "--llm-model", "--prompt-seed", "load_pm(", "data/liar/processed"):
        if marker not in main_model:
            fail(f"main entry point is missing marker: {marker}")
    print("[SKIP] external dataset checks: prepare data according to data/README.md")
    print("[PASS] SharP public code release structure is ready.")
    raise SystemExit(0)

print(f"[OK] required local runtime/data files: {len(REQUIRED_DATA)}")

required_columns = {"id", "label", "content", "if_marked_label", "event_label"}
total_rows = 0
for event_id, event_name, filename, expected_rows in PHEME_FILES:
    path = ROOT / "data" / "pheme" / "raw" / filename
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required_columns.issubset(rows[0]):
        fail(f"bad PHEME schema in {filename}")
    if len(rows) != expected_rows:
        fail(f"unexpected raw row count in {filename}: {len(rows)} != {expected_rows}")
    stored_ids = {str(row["event_label"]).removesuffix(".0") for row in rows}
    if stored_ids != {str(event_id)}:
        fail(f"event mapping mismatch for {event_name}: {stored_ids}")
    total_rows += len(rows)
print(f"[OK] PHEME event mapping 0..4 and raw schemas: {total_rows} rows")


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


def read_xlsx(path: Path) -> list[dict[str, str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                shared.append("".join(node.text or "" for node in item.findall(".//x:t", ns)))
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    matrix: list[list[str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", ns):
        values: dict[int, str] = {}
        for cell in row.findall("x:c", ns):
            idx = column_index(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            value_node = cell.find("x:v", ns)
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", ns))
            else:
                value = value_node.text if value_node is not None else ""
                if cell_type == "s" and value:
                    value = shared[int(value)]
            values[idx] = value
        width = max(values, default=-1) + 1
        matrix.append([values.get(i, "") for i in range(width)])
    if not matrix:
        return []
    headers = matrix[0]
    return [dict(zip(headers, row + [""] * (len(headers) - len(row)))) for row in matrix[1:]]


def check_rows(relative: str, expected: int) -> list[dict[str, str]]:
    rows = read_xlsx(ROOT / relative)
    if len(rows) != expected:
        fail(f"unexpected row count in {relative}: {len(rows)} != {expected}")
    return rows


for event_id, event_name, _, _ in PHEME_FILES:
    rows = check_rows(f"data/pheme/processed/events/event_{event_id}_pool.xlsx", 280)
    ids = {str(row["event_label"]).removesuffix(".0") for row in rows}
    if ids != {str(event_id)}:
        fail(f"pool mapping mismatch for {event_name}: {ids}")
print("[OK] five PHEME event pools: 280 rows each with matching event ids")

event2_pool = read_xlsx(ROOT / "data/pheme/processed/events/event_2_pool.xlsx")
pool_signatures = {
    (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
    for row in event2_pool
}
for seed_name, expected in (("clippool_16.xlsx", 19), ("clippool_50.xlsx", 50)):
    seed_rows = check_rows(f"data/pheme/prompt_seed/{seed_name}", expected)
    seed_signatures = {
        (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
        for row in seed_rows
    }
    if not seed_signatures.issubset(pool_signatures):
        fail(f"{seed_name} is not an exact subset of the event-2 pool")
print("[OK] clippool_16 and clippool_50 are exact Germanwings pool subsets")

liar_expected = {
    "train2_data.xlsx": 10240,
    "pool_data.xlsx": 280,
    "validate_data.xlsx": 1284,
    "test_data.xlsx": 1267,
}
for name, expected in liar_expected.items():
    check_rows(f"data/liar/processed/{name}", expected)
print("[OK] LIAR prepared totals: 12,791 records (train2 + validation + test)")

liar_pool = read_xlsx(ROOT / "data/liar/processed/pool_data.xlsx")
liar_seed = check_rows("data/liar/prompt_seed/clippool_16_release.xlsx", 19)
liar_pool_signatures = {
    (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
    for row in liar_pool
}
liar_seed_signatures = {
    (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
    for row in liar_seed
}
if not liar_seed_signatures.issubset(liar_pool_signatures):
    fail("LIAR release prompt seed is not an exact pool subset")
print("[OK] LIAR release prompt seed: 19 exact pool rows")

twitter_expected = {
    "twitter_train2_data.xlsx": 924,
    "twitter_pool_data.xlsx": 280,
    "twitter_validate_data.xlsx": 115,
    "twitter_test_data.xlsx": 115,
}
for name, expected in twitter_expected.items():
    check_rows(f"data/twitter15_16/processed/{name}", expected)
print("[OK] Twitter15/16 prepared totals: 1,154 records (train2 + validation + test)")

twitter_pool = read_xlsx(ROOT / "data/twitter15_16/processed/twitter_pool_data.xlsx")
twitter_seed = check_rows("data/twitter15_16/prompt_seed/clippool_16_release.xlsx", 19)
twitter_pool_signatures = {
    (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
    for row in twitter_pool
}
twitter_seed_signatures = {
    (row.get("id", ""), row.get("label", ""), row.get("content", ""), row.get("event_label", ""))
    for row in twitter_seed
}
if not twitter_seed_signatures.issubset(twitter_pool_signatures):
    fail("Twitter15/16 release prompt seed is not an exact pool subset")
print("[OK] Twitter15/16 release prompt seed: 19 exact pool rows")

main_model = (ROOT / "DAAL" / "model_adjust.py").read_text(encoding="utf-8-sig")
for marker in ("--less-frac", "--llm-model", "--prompt-seed", "load_pm(", "data/liar/processed"):
    if marker not in main_model:
        fail(f"main entry point is missing marker: {marker}")
print("[OK] main LIAR entry point and packaged prompt-seed default")

twitter_model = (ROOT / "experiments" / "cross_benchmark" / "model_adjust.py").read_text(
    encoding="utf-8-sig"
)
active_twitter_injection = re.findall(
    r"^\s*(?!#)(?:df_tmp|df_train_after)\s*=\s*load_pm\(",
    twitter_model,
    flags=re.MULTILINE,
)
if len(active_twitter_injection) != 2:
    fail(f"expected two active Twitter augmentation insertion points, found {len(active_twitter_injection)}")
if "data\" / \"twitter15_16\" / \"prompt_seed\"" not in twitter_model:
    fail("Twitter entry point is missing the packaged prompt-seed default")
print("[OK] Twitter runner: two active augmentation insertions and packaged prompt seed")

print("[PASS] SharP release structure is ready for GPU/environment review.")

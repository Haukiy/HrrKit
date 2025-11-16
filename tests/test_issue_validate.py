import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import issue_validate


def _body_with_inline() -> str:
    return """### File details for database.csv

ID: CAR_2024_01
Scource in IDEEE: Example Source
Topic: car
Time Unit: s
Energy Unit: kW
Attachment filename: car_curve.csv

```csv filename=car_curve.csv
time,hrr
0,0
10,25
20,30
```
"""


def test_validate_items_accepts_inline_csv(tmp_path):
    body = _body_with_inline()
    results, _ = issue_validate.validate_items(body, tmp_path)
    assert len(results) == 1
    assert results[0].ok

    report, ok = issue_validate._build_report(results)
    assert ok
    assert "Overall status: OK" in report
    assert "CAR_2024_01" in report


def test_validate_items_reports_missing_blocks(tmp_path):
    empty_body = "Please fill the template"
    results, _ = issue_validate.validate_items(empty_body, tmp_path)
    assert results == []
    report, ok = issue_validate._build_report(results)
    assert not ok
    assert "no intake items" in report.lower()


def test_validate_items_checks_required_fields(tmp_path):
    body = """```csv
ID,Scource in IDEEE,Time Unit,Energy Unit,Topic,Filename
MissingUnits,Example,,,car,missing.csv
```

```csv filename=missing.csv
time,hrr
0,0
```
"""

    results, _ = issue_validate.validate_items(body, tmp_path)
    assert len(results) == 1
    assert not results[0].ok
    assert any("Time Unit" in err for err in results[0].errors)


def test_validate_items_reads_local_files(tmp_path):
    csv_path = tmp_path / "repo" / "topic" / "curve.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("time,hrr\n0,0\n1,5\n2,10\n", encoding="utf-8")

    rel = csv_path.relative_to(tmp_path)
    body = f"""```csv
ID,Scource in IDEEE,Time Unit,Energy Unit,Topic,Filename
LocalRow,Example,s,kW,car,{rel}
```
"""

    results, _ = issue_validate.validate_items(body, tmp_path)
    assert len(results) == 1
    assert results[0].ok

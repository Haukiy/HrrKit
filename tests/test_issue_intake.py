import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import issue_intake


def _build_issue_body() -> str:
    return """File details for database.csv

ID: 1
Scource in IDEEE: 10.1016/j.firesaf.2025.104558
Topic: car
Time Unit: min
Energy Unit: kW
Attachment filename: CAR_2010_Test03.csv

ID: 2
Scource in IDEEE: 10.1007/s10694-025-01725-x
Topic: car
Time Unit: min
Energy Unit: kW
Attachment filename: CAR_2010_Test04.csv

[CAR_2010_Test03.csv](https://example.com/files/CAR_2010_Test03.csv?download=1)
[CAR_2010_Test04.csv](https://example.com/files/CAR_2010_Test04.csv?raw=1)
"""


def test_ingest_issue_accepts_csv_links_with_query(tmp_path, monkeypatch):
    downloads = []

    def fake_download(url: str, dest: Path, token: str) -> bool:
        downloads.append(url)
        dest.write_text("time,hrr\n0,0\n")
        return True

    monkeypatch.setattr(issue_intake, "download_with_auth", fake_download)

    database = tmp_path / "database.csv"
    raw_dir = tmp_path / "raw"

    added, updated = issue_intake.ingest_issue(
        _build_issue_body(),
        issue_number="42",
        run_id="123",
        token="",
        database_path=database,
        raw_dir=raw_dir,
    )

    assert added == 2
    assert updated == 0
    assert len(downloads) == 2

    with database.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        filename = row["Filename"]
        assert filename
        assert (raw_dir / filename).is_file()


def test_ingest_issue_skips_instruction_code_block(tmp_path):
    issue_body = """**Instructions**
1) Attach files

```
ID: <per-issue number>
Scource in IDEEE: <who provided the data>
Topic: <topic name>
Time Unit: <e.g., s>
Energy Unit: <e.g., kW>
Attachment filename: <uploaded filename>
```

### File details for database.csv

ID: 1
Scource in IDEEE: Example Source
Topic: car
Time Unit: s
Energy Unit: kW
Attachment filename: car_test.csv

```csv filename=car_test.csv
time,hrr
0,0
```
"""

    database = tmp_path / "database.csv"
    raw_dir = tmp_path / "raw"

    added, updated = issue_intake.ingest_issue(
        issue_body,
        issue_number="77",
        run_id="123",
        token="",
        database_path=database,
        raw_dir=raw_dir,
    )

    assert added == 1
    assert updated == 0
    with database.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ID"] == "1"

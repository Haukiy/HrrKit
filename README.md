## IEEE source formatting 

The `Scource in IDEEE` column accepts either a fully formatted IEEE reference or
an identifier that the script can expand on your behalf:

- **DOIs** – accepted forms: `doi:10.1109/5.771073`, `https://doi.org/10.1109/5.771073`, or a raw `10.xxxx/yyy` string.
- **ISBNs** – accepted forms: `isbn:9780131101630`, `ISBN 0-201-53082-1`, or a plain 10/13 digit value with optional hyphens.

When one of these patterns is detected the script resolves the identifier and
replaces the `Scource in IDEEE` value with an IEEE-formatted reference. This
updated value is stored in `database.csv`, the per-topic source registry, and the
plot captions, so the user only has to provide the DOI/ISBN once.

## Adding the first HRR entry

When `database.csv` is empty it can be tedious to craft the first row manually.
Use the CLI helper to append a pending entry directly from the terminal:

```bash
python hrrkit_excel.py \
  --add-row \
  --add-id "My test ID" \
  --add-source "doi:10.1234/example" \
  --add-time-unit sec \
  --add-energy-unit kW \
  --add-topic "Generic" \
  --add-filename "example_curve.csv"
```

The command creates the CSV (if needed), adds a new row with blank `ProcessedAt`
so that it shows up as "pending", and then you can run the regular validation or
processing workflow.

## Automatic intake from GitHub issues

The CI workflow now calls [`issue_intake.py`](./issue_intake.py) before every
validation/processing run. The helper reads the HRR intake issue body, downloads
any referenced attachments (or inline fenced `csv` blocks that declare
`filename=...`), and
appends pending rows to `database.csv` with a stable `SubmissionKey`. Because of
this, `/validate` works even when the repository starts with an empty database:
as soon as the issue provides rows, they are imported automatically. You can run
the same helper locally if you want to preview the import:

```bash
python issue_intake.py --body-file issue.md --database database.csv --raw-dir raw_files
```

The file `issue.md` should contain the Markdown body from the GitHub issue (you
can copy/paste it from the UI). Inline CSV blocks are written to `raw_files/`
and attachments are downloaded via their GitHub URLs, matching the behavior in
CI.

If `/validate` reports **NOOP** or "no intake items found", the issue body
was parsed without detecting any intake rows. Make sure the issue contains
either a CSV table with the header `ID,Scource in IDEEE,Topic,Time Unit,Energy
Unit,Filename` or the structured blocks from the intake template (the
"ID/Topic/Time Unit/…" prompts). The **data files themselves do not need these
metadata headers**—they can remain as simple time/HRR two-column CSVs—but the
issue must still declare one intake block per file so the workflow knows how to
seed `database.csv`. Once at least one block is present the intake helper will
append pending rows to `database.csv`, allowing `/validate` to run
automatically. The `hrrkit_excel.py --validate-only` CLI now follows the same
pattern: if no pending rows are found it will try to import blocks from the
issue body provided via `--issue-body-text`, `--issue-body-file`, or the
`ISSUE_BODY` environment variable before declaring a NOOP.


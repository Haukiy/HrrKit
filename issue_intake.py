#!/usr/bin/env python3
"""Parse HRR intake issues and seed database.csv with pending rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import urllib.error
import urllib.request

DEFAULT_DATABASE = "database.csv"
DEFAULT_RAW_DIR = "raw_files"
SUBMISSION_COL = "SubmissionKey"
REQUIRED_FIELDS = [
    "ID",
    "Scource in IDEEE",
    "Time Unit",
    "Energy Unit",
    "Topic",
]
class IntakeError(RuntimeError):
    """Raised when the issue payload cannot be converted into DB rows."""


@dataclass
class Attachment:
    name: str
    url: str
    used: bool = False

    @property
    def name_lower(self) -> str:
        return pathlib.Path(self.name).name.lower()


def _normalize_csv_header(value: str) -> str:
    return re.sub(r"\s*,\s*", ",", (value or "").strip().lower())


def find_csv_block(text: str) -> str:
    """Return the first CSV-looking fenced block found in the text."""

    fence_pattern = re.compile(r"```([^\n]*)\n([\s\S]*?)```", flags=re.I)
    for match in fence_pattern.finditer(text):
        info_string = (match.group(1) or "").strip().lower()
        block = (match.group(2) or "").strip()
        if not block:
            continue
        if "filename=" in info_string:
            continue
        if info_string and info_string not in ("csv",):
            continue
        first_line = block.splitlines()[0].strip().lower()
        if re.match(r"^(?:#\s*)?file\s*:", first_line):
            continue
        return block

    lines = text.splitlines()
    canonical_headers = [
        "id,scource in ideee,time unit,energy unit,topic",
        "id,scource in ideee,time unit,energy unit,topic,attachment filename",
        "id,scource in ideee,time unit,energy unit,topic,filename",
    ]
    canonical_no_space = [_normalize_csv_header(header).replace(",", "") for header in canonical_headers]
    start = None
    for i, line in enumerate(lines):
        raw = (line or "").strip().lower()
        if not raw:
            continue
        normalized = _normalize_csv_header(raw)
        nospace = normalized.replace(",", "")
        if normalized in canonical_headers or nospace in canonical_no_space:
            start = i
            break
    if start is None:
        return ""

    block: List[str] = []
    block.append(_normalize_csv_header(lines[start]))
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        comma_count = stripped.count(",")
        if comma_count >= 5 or (re.match(r"^\s*\d+\s*,", stripped) and comma_count >= 5):
            block.append(_normalize_csv_header(stripped))
            continue
        if stripped.startswith("[") or stripped.startswith("http"):
            break
        if stripped.startswith("#") or "Notes (optional)" in stripped or "Attachments" in stripped:
            break
        break
    return "\n".join(block).strip()


def normalize_key(name: str) -> str:
    """Return a canonical representation for template field names."""

    cleaned = (name or "").strip().lower()
    # Issue forms often wrap field names in Markdown emphasis (e.g. "**ID**")
    # or prefix them with bullets. Strip those wrapper characters while
    # keeping meaningful punctuation such as parentheses.
    cleaned = re.sub(r"^[\s>\-*_`•]+", "", cleaned)
    cleaned = re.sub(r"[\s>\-*_`•]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_structured_blocks(text: str) -> List[Dict[str, str]]:
    """Parse repeated "key: value" blocks from the intake issue template."""

    key_map = {
        "id": "ID",
        "scource in ideee": "Scource in IDEEE",
        "source in ideee": "Scource in IDEEE",
        "topic": "Topic",
        "filename (save as)": "Filename",
        "filename save as": "Filename",
        "filename": "Filename",
        "time unit": "Time Unit",
        "energy unit": "Energy Unit",
        "attachment filename": "__attachment_hint__",
        "attachment file name": "__attachment_hint__",
    }
    rows: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    def flush_current() -> None:
        nonlocal current
        if any(current.get(field) for field in REQUIRED_FIELDS):
            rows.append(current)
        current = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue
        if stripped.startswith("###"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        canonical = key_map.get(normalize_key(key))
        if not canonical:
            continue
        if canonical == "ID" and current and any(current.values()):
            flush_current()
        current[canonical] = value.strip()
    flush_current()
    parsed = []
    for entry in rows:
        if all(entry.get(field, "").strip() for field in REQUIRED_FIELDS):
            normalized = {field: entry.get(field, "").strip() for field in REQUIRED_FIELDS}
            normalized["__attachment_hint__"] = entry.get("__attachment_hint__", "").strip()
            parsed.append(normalized)
    return parsed


def parse_rows(body: str) -> List[Dict[str, str]]:
    csv_block = find_csv_block(body)
    if csv_block:
        csv_block = re.sub(r",\s+", ",", csv_block)
        reader = list(csv.DictReader(io.StringIO(csv_block)))
        if not reader:
            raise IntakeError("CSV block parsed but contains no rows.")
        csv_key_map = {
            "id": "ID",
            "scource in ideee": "Scource in IDEEE",
            "source in ideee": "Scource in IDEEE",
            "topic": "Topic",
            "time unit": "Time Unit",
            "energy unit": "Energy Unit",
            "filename": "Filename",
            "filename (save as)": "Filename",
            "attachment filename": "__attachment_hint__",
            "attachment file name": "__attachment_hint__",
        }
        rows: List[Dict[str, str]] = []
        for raw in reader:
            normalized: Dict[str, str] = {"__attachment_hint__": ""}
            for key, value in raw.items():
                if key is None:
                    continue
                canonical = csv_key_map.get(key.strip().lower())
                target_key = canonical or key.strip()
                normalized[target_key] = (value or "").strip()
            rows.append(normalized)
        return rows

    rows = parse_structured_blocks(body)
    if not rows:
        raise IntakeError("No CSV block or structured entry blocks found in issue body.")
    return rows


def extract_inline_files(body: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    pattern = re.compile(r"```([^\n]*)\n([\s\S]*?)```", flags=re.MULTILINE)
    for match in pattern.finditer(body):
        info = (match.group(1) or "").strip()
        content = match.group(2)
        filename = None
        hint_match = re.search(r"filename\s*=\s*([^\s]+)", info, flags=re.I)
        if hint_match:
            filename = hint_match.group(1).strip()
        if not filename:
            first_line, *rest = content.splitlines()
            marker = re.search(r"^\s*(?:#\s*)?file\s*:\s*(.+)$", first_line.strip(), flags=re.I)
            if marker:
                filename = marker.group(1).strip()
                content = "\n".join(rest) + ("\n" if rest else "")
        if filename:
            files[pathlib.Path(filename).name] = content
    return files


def download_with_auth(url: str, dest: pathlib.Path, token: str) -> bool:
    headers = {"Accept": "application/octet-stream", "User-Agent": "hrr-issue-intake"}
    if token:
        headers["Authorization"] = f"token {token}"

    def attempt_download(target_url: str, extra_headers: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        combined_headers = dict(headers)
        if extra_headers:
            combined_headers.update(extra_headers)
        request = urllib.request.Request(target_url, headers=combined_headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError):
            return None

    for attempt_url in (url, url + "?download=1"):
        data = attempt_download(attempt_url)
        if data is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def clean_attachment_name(name: str) -> str:
    cleaned = (name or "").strip()
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
    return pathlib.Path(cleaned).name if cleaned else ""


def slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return cleaned or fallback


def canonical_filename(issue_seed: str, row_idx: int, topic: str, row_id: str, source_name: str,
                        time_unit: str, energy_unit: str) -> str:
    topic_slug = slugify(topic, "topic")[:40]
    row_slug = slugify(row_id, f"row{row_idx}")[:40]
    ext = pathlib.Path(source_name or "").suffix.lower()
    if ext not in (".csv", ".txt"):
        ext = ".csv"
    seed = "|".join([
        issue_seed or "manual",
        str(row_idx),
        topic_slug,
        row_slug,
        (time_unit or "").strip().lower(),
        (energy_unit or "").strip().lower(),
        (source_name or "").strip().lower(),
    ])
    digest = hashlib.sha256(seed.encode("utf-8", "ignore")).hexdigest()[:10]
    return f"{topic_slug}_{row_slug}_{digest}_raw{ext}"


DATABASE_COLUMNS = REQUIRED_FIELDS + [
    "Filename",
    "ProcessedAt",
    "Peak_kW",
    "Area_kJ",
    "UID",
    "Concise ID",
    SUBMISSION_COL,
]


def ensure_row_defaults(row: Dict[str, str]) -> Dict[str, str]:
    for column in DATABASE_COLUMNS:
        row.setdefault(column, "")
    return row


def read_database(database_path: pathlib.Path) -> List[Dict[str, str]]:
    if not database_path.exists():
        return []
    with database_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [ensure_row_defaults(dict(row)) for row in reader]
    return rows


def write_database(database_path: pathlib.Path, rows: Sequence[Dict[str, str]]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with database_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATABASE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DATABASE_COLUMNS})


def upsert_rows(existing_rows: List[Dict[str, str]], rows: Sequence[Dict[str, str]], submission_prefix: str) -> Tuple[List[Dict[str, str]], int, int]:
    existing_keys = {
        (row.get(SUBMISSION_COL, "") or "").strip()
        for row in existing_rows
        if (row.get(SUBMISSION_COL, "") or "").strip()
    }
    updated = 0
    added = 0
    normalized_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        normalized: Dict[str, str] = {field: row.get(field, "") for field in REQUIRED_FIELDS + ["Filename"]}
        normalized[SUBMISSION_COL] = f"{submission_prefix}-row-{idx}"
        normalized_rows.append(normalized)

    for normalized in normalized_rows:
        key = (normalized.get(SUBMISSION_COL, "") or "").strip()
        if key and key in existing_keys:
            for existing in existing_rows:
                if (existing.get(SUBMISSION_COL, "") or "").strip() == key:
                    existing.update(normalized)
                    ensure_row_defaults(existing)
                    updated += 1
                    break
        else:
            new_row = ensure_row_defaults(dict(normalized))
            existing_rows.append(new_row)
            if key:
                existing_keys.add(key)
            added += 1
    return existing_rows, added, updated


def ensure_raw_file(row: Dict[str, str], idx: int, issue_seed: str, attachments: List[Attachment],
                    inline_files: Dict[str, str], inline_used: Dict[str, bool], raw_dir: pathlib.Path,
                    token: str) -> str:
    attachment_hint = clean_attachment_name(row.get("__attachment_hint__", ""))
    chosen_inline = None
    chosen_attachment: Optional[Attachment] = None

    if attachment_hint and attachment_hint in inline_files and not inline_used.get(attachment_hint):
        inline_used[attachment_hint] = True
        chosen_inline = attachment_hint
    elif attachment_hint:
        for att in attachments:
            if not att.used and att.name_lower == pathlib.Path(attachment_hint).name.lower():
                chosen_attachment = att
                break

    if not chosen_inline and not chosen_attachment:
        for att in attachments:
            if not att.used:
                chosen_attachment = att
                break

    if not chosen_inline and not chosen_attachment:
        for name, used in inline_used.items():
            if not used:
                inline_used[name] = True
                chosen_inline = name
                break

    if not chosen_inline and not chosen_attachment:
        raise IntakeError(f"Could not match any attachment for row ID {row.get('ID') or idx}.")

    source_name = chosen_inline or (chosen_attachment.name if chosen_attachment else attachment_hint)
    canonical_name = canonical_filename(
        issue_seed,
        idx,
        row.get("Topic", ""),
        row.get("ID", f"row{idx}"),
        source_name or f"row{idx}.csv",
        row.get("Time Unit", ""),
        row.get("Energy Unit", ""),
    )
    destination = raw_dir / canonical_name

    if chosen_inline:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(inline_files[chosen_inline])
    elif chosen_attachment:
        ok = download_with_auth(chosen_attachment.url, destination, token)
        if not ok:
            raise IntakeError(f"Could not download attachment '{chosen_attachment.name}'.")
        chosen_attachment.used = True
    else:
        raise IntakeError(f"Could not determine raw file for row ID {row.get('ID') or idx}.")

    return canonical_name


def ingest_issue(body: str, issue_number: str, run_id: str, token: str, database_path: pathlib.Path,
                 raw_dir: pathlib.Path) -> Tuple[int, int]:
    rows = parse_rows(body)
    for field in REQUIRED_FIELDS:
        if field not in rows[0]:
            raise IntakeError(
                "CSV header mismatch. Need exactly: " + ", ".join(REQUIRED_FIELDS)
            )

    raw_dir.mkdir(parents=True, exist_ok=True)
    attachments = []
    attachment_pattern = re.compile(r"\((https?://[^\s)]+\.csv)\)", flags=re.I)
    for url in attachment_pattern.findall(body):
        clean_url = url.strip()
        name = pathlib.Path(clean_url.split("?")[0]).name
        attachments.append(Attachment(name=name, url=clean_url))

    inline_files = extract_inline_files(body)
    inline_used = {name: False for name in inline_files.keys()}

    issue_seed = issue_number or run_id or "manual"

    prepared_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        for field in REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                raise IntakeError(f"Missing field in CSV row {idx}: {field}")
        filename = ensure_raw_file(row, idx, issue_seed, attachments, inline_files, inline_used, raw_dir, token)
        normalized = {field: row.get(field, "").strip() for field in REQUIRED_FIELDS}
        normalized["Filename"] = filename
        prepared_rows.append(normalized)

    existing_rows = read_database(database_path)
    submission_prefix = f"issue-{issue_number}" if issue_number else (f"run-{run_id}" if run_id else "manual")
    updated_rows, added, updated = upsert_rows(existing_rows, prepared_rows, submission_prefix)
    write_database(database_path, updated_rows)
    return added, updated


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import HRR intake issue rows into database.csv")
    parser.add_argument("--body-file", default="", help="Path to file containing the issue body (Markdown)")
    parser.add_argument("--body-text", default="", help="Raw issue body text")
    parser.add_argument("--database", default=os.environ.get("HRR_DATABASE", DEFAULT_DATABASE))
    parser.add_argument("--raw-dir", default=os.environ.get("HRR_RAW_DIR", DEFAULT_RAW_DIR))
    parser.add_argument("--issue-number", default=os.environ.get("ISSUE_NUMBER", ""))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--require-body", action="store_true", help="Fail if no issue body is provided")
    return parser.parse_args(argv)


def load_body(args: argparse.Namespace) -> str:
    if args.body_text:
        return args.body_text
    if args.body_file:
        return pathlib.Path(args.body_file).read_text(encoding="utf-8")
    return os.environ.get("ISSUE_BODY", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    body = load_body(args)
    if not body.strip():
        msg = "No issue body provided; skipping issue intake."
        if args.require_body:
            print(msg, file=sys.stderr)
            return 1
        print(msg)
        return 0
    database_path = pathlib.Path(args.database).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve()
    try:
        added, updated = ingest_issue(body, args.issue_number.strip(), args.run_id.strip(), args.token.strip(), database_path, raw_dir)
    except IntakeError as exc:
        print(f"Intake error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive catch for workflow logs
        print(f"Unexpected error during intake: {exc}", file=sys.stderr)
        return 1
    print(f"Issue intake complete: {added} row(s) added, {updated} row(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

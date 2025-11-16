#!/usr/bin/env python3
"""Phase 1 (/validate): Validate HRR intake items from a GitHub issue."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import urllib.error
import urllib.request

import issue_intake

TIME_COL_CANDIDATES = ["time_s", "time", "t", "time_raw", "zeit", "Time", "TIME"]
HRR_COL_CANDIDATES = ["hrr_kW", "hrr", "q", "HRR", "Q", "Leistung", "power"]
TIME_UNIT_MAP = {
    "s": "sec",
    "sec": "sec",
    "second": "sec",
    "seconds": "sec",
    "min": "min",
    "minute": "min",
    "minutes": "min",
}
ENERGY_UNIT_MAP = {
    "kw": "kw",
    "kW": "kw",
    "KW": "kw",
    "mw": "mw",
    "MW": "mw",
}


def detect_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered: Dict[str, str] = {}
    for column in columns:
        lowered[str(column).strip().lower()] = column
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def normalize_time_unit(value: str) -> str:
    key = (value or "").strip()
    return TIME_UNIT_MAP.get(key, TIME_UNIT_MAP.get(key.lower(), ""))


def normalize_energy_unit(value: str) -> str:
    key = (value or "").strip()
    return ENERGY_UNIT_MAP.get(key, ENERGY_UNIT_MAP.get(key.lower(), ""))


def _coerce_numeric(values: List[str]) -> List[Optional[float]]:
    coerced: List[Optional[float]] = []
    for value in values:
        if value is None:
            coerced.append(None)
            continue
        text = str(value).strip()
        if not text:
            coerced.append(None)
            continue
        text = text.replace(",", ".")
        text = re.sub(r"\s+", "", text)
        try:
            coerced.append(float(text))
        except ValueError:
            coerced.append(None)
    return coerced


def _load_csv_table(content: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file contains no header row")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV file contains no data rows")
    return reader.fieldnames, rows


@dataclass
class IntakeItem:
    idx: int
    raw: Dict[str, str]
    filename: str
    attachment_hint: str

    @property
    def id(self) -> str:
        return (self.raw.get("ID") or "").strip()

    @property
    def topic(self) -> str:
        return (self.raw.get("Topic") or "").strip()

    @property
    def time_unit(self) -> str:
        return (self.raw.get("Time Unit") or "").strip()

    @property
    def energy_unit(self) -> str:
        return (self.raw.get("Energy Unit") or "").strip()

    @property
    def description(self) -> str:
        if self.id and self.topic:
            return f"{self.id} ({self.topic})"
        return self.id or self.topic or f"item #{self.idx}"


@dataclass
class ValidationResult:
    item: IntakeItem
    ok: bool
    errors: List[str]


class AttachmentFetchError(RuntimeError):
    pass


def _normalize_filename(name: str) -> str:
    return pathlib.Path(name or "").name.strip().lower()


def _collect_attachments(body: str) -> List[issue_intake.Attachment]:
    attachments: List[issue_intake.Attachment] = []
    pattern = re.compile(
        r"\[([^\]]+)\]\((https?://[^\s)]+\.(?:csv|txt)(?:\?[^\s)]*)?)\)",
        flags=re.I,
    )
    for name, url in pattern.findall(body):
        clean_url = url.strip()
        clean_name = pathlib.Path(name.strip()).name or pathlib.Path(clean_url.split("?")[0]).name
        attachments.append(issue_intake.Attachment(name=clean_name, url=clean_url))
    return attachments


def _download_bytes(url: str, token: str) -> bytes:
    headers = {"Accept": "application/octet-stream", "User-Agent": "hrr-issue-validate"}
    if token:
        headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:  # pragma: no cover - network failure
        raise AttachmentFetchError(str(exc))


def _load_file_content(
    item: IntakeItem,
    inline_files: Dict[str, str],
    attachments: List[issue_intake.Attachment],
    files_root: pathlib.Path,
    token: str,
) -> Tuple[str, bytes]:
    filename = item.filename or item.attachment_hint
    if not filename:
        raise FileNotFoundError("No filename provided in intake block")
    normalized = _normalize_filename(filename)

    # Inline file blocks take precedence
    for name, content in inline_files.items():
        if _normalize_filename(name) == normalized:
            return name, content.encode("utf-8")

    # Local file path (relative to repo)
    local_path = files_root / filename
    if local_path.is_file():
        return local_path.name, local_path.read_bytes()

    # Attachments via URL
    for att in attachments:
        if att.name_lower == normalized:
            return att.name, _download_bytes(att.url, token)

    raise FileNotFoundError(f"Referenced file '{filename}' not found")


def _parse_items(body: str) -> Tuple[List[IntakeItem], Dict[str, str]]:
    inline_files = issue_intake.extract_inline_files(body)
    try:
        rows = issue_intake.parse_rows(body)
    except issue_intake.IntakeError:
        return [], inline_files
    items = []
    for idx, row in enumerate(rows, start=1):
        filename = row.get("Filename", "").strip()
        attachment_hint = row.get("__attachment_hint__", "").strip()
        items.append(
            IntakeItem(
                idx=idx,
                raw=row,
                filename=filename,
                attachment_hint=attachment_hint,
            )
        )
    return items, inline_files


def _validate_units(item: IntakeItem) -> List[str]:
    errors: List[str] = []
    if not item.time_unit:
        errors.append("Missing Time Unit field")
    elif not normalize_time_unit(item.time_unit):
        errors.append(f"Unrecognized Time Unit: {item.time_unit}")
    if not item.energy_unit:
        errors.append("Missing Energy Unit field")
    elif not normalize_energy_unit(item.energy_unit):
        errors.append(f"Unrecognized Energy Unit: {item.energy_unit}")
    return errors


def _validate_table(headers: List[str], rows: List[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    if len(headers) < 2:
        return ["HRR file must contain at least two columns"]
    time_col = detect_column(headers, TIME_COL_CANDIDATES) or headers[0]
    hrr_col = detect_column(headers, HRR_COL_CANDIDATES) or headers[1]

    time_values = _coerce_numeric([row.get(time_col, "") for row in rows])
    hrr_values = _coerce_numeric([row.get(hrr_col, "") for row in rows])

    numeric_time = [value for value in time_values if value is not None]
    if len(numeric_time) < 2:
        errors.append("Time column does not contain enough numeric values")
    else:
        for prev, curr in zip(numeric_time, numeric_time[1:]):
            if curr < prev:
                errors.append("Time axis must be monotonic (non-decreasing)")
                break

    numeric_hrr = [value for value in hrr_values if value is not None]
    if not numeric_hrr:
        errors.append("HRR column does not contain numeric values")
    elif any(value < 0 for value in numeric_hrr):
        errors.append("HRR values must be non-negative")
    return errors


def validate_items(
    body: str,
    files_root: pathlib.Path,
    token: str = "",
) -> Tuple[List[ValidationResult], Dict[str, str]]:
    items, inline_files = _parse_items(body)
    attachments = _collect_attachments(body)
    results: List[ValidationResult] = []

    for item in items:
        item_errors: List[str] = []
        missing_fields = [
            name
            for name in issue_intake.REQUIRED_FIELDS
            if not (item.raw.get(name) or "").strip()
        ]
        if missing_fields:
            item_errors.append("Missing fields: " + ", ".join(missing_fields))

        item_errors.extend(_validate_units(item))

        content: Optional[bytes] = None
        source_name = ""
        if not item_errors:
            try:
                source_name, content = _load_file_content(
                    item,
                    inline_files,
                    attachments,
                    files_root,
                    token,
                )
            except (FileNotFoundError, AttachmentFetchError) as exc:
                item_errors.append(str(exc))

        if not item_errors and content is not None:
            try:
                headers, rows = _load_csv_table(content)
            except Exception as exc:
                item_errors.append(f"Could not parse '{source_name}': {exc}")
            else:
                item_errors.extend(_validate_table(headers, rows))

        results.append(ValidationResult(item=item, ok=not item_errors, errors=item_errors))

    return results, {name: content for name, content in inline_files.items()}


def _build_report(results: Sequence[ValidationResult]) -> Tuple[str, bool]:
    if not results:
        message = (
            "HRR validation report\n\n"
            "Overall status: FAILED – no intake items found\n\n"
            "/validate only works when the issue contains at least one intake block using the template.\n"
            "Please add at least one block and rerun /validate.\n\n"
            "No changes have been written to the database yet."
        )
        return message, False

    success = all(result.ok for result in results)
    if success:
        lines = ["HRR validation report", "", "Overall status: OK", ""]
        if results:
            lines.append("Validated intake items:")
            for result in results:
                lines.append(f"- {result.item.description}")
            lines.append("")
        lines.append(
            "No changes have been written to the database yet. Use /commit to run Phase 2 and import the validated curves."
        )
        return "\n".join(lines).rstrip() + "\n", True

    lines = ["HRR validation report", "", "Overall status: FAILED", ""]
    for result in results:
        if result.ok:
            continue
        lines.append(f"- {result.item.description}:")
        for err in result.errors:
            lines.append(f"    - {err}")
    lines.append("")
    lines.append("No changes have been written to the database. Please fix the issues and rerun /validate.")
    return "\n".join(lines).rstrip() + "\n", False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HRR intake data from a GitHub issue body")
    parser.add_argument("--body-file", default="", help="Path to a file that contains the issue body")
    parser.add_argument("--body-text", default="", help="Raw issue body text")
    parser.add_argument("--files-root", default=".", help="Root directory for referenced files")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--report-json", default="", help="Optional path to save the validation report JSON")
    parser.add_argument("--require-body", action="store_true", help="Fail if no issue body is provided")
    return parser.parse_args(argv)


def _load_body(args: argparse.Namespace) -> str:
    if args.body_text:
        return args.body_text
    if args.body_file:
        return pathlib.Path(args.body_file).read_text(encoding="utf-8")
    return os.environ.get("ISSUE_BODY", "")


def _report_payload(results: Sequence[ValidationResult], ok: bool) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "id": result.item.id,
                "topic": result.item.topic,
                "filename": result.item.filename or result.item.attachment_hint,
                "status": "ok" if result.ok else "failed",
                "errors": result.errors,
            }
        )
    overall = "ok" if ok else ("failed" if results else "empty")
    return {"overall_status": overall, "items": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    body = _load_body(args)
    if not body.strip():
        msg = "No issue body provided; skipping validation."
        if args.require_body:
            print(msg)
            return 1
        print(msg)
        return 0

    results, _ = validate_items(body, pathlib.Path(args.files_root).resolve(), args.token)
    report, ok = _build_report(results)
    print(report)

    if args.report_json:
        pathlib.Path(args.report_json).write_text(json.dumps(_report_payload(results, ok), indent=2))

    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

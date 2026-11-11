#!/usr/bin/env python3
"""Inspect CHB-MIT EDF recordings and their available seizure annotations."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from annotation_utils import parse_patient_summary as _parse_patient_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
ANNOTATION_EXTENSIONS = {".ann", ".annotation", ".annotations", ".csv", ".tsv", ".xml"}


def parse_patient_summary(summary_path: Path) -> list[dict[str, Any]]:
    """Extract recording filenames and seizure intervals from a CHB-MIT summary."""
    records, _ = _parse_patient_summary(summary_path)
    return [
        {
            "filename": record["filename"],
            "seizure_count": record["declared_count"],
            "starts": [
                interval["start_seconds"]
                for interval in record["intervals"]
                if interval["start_seconds"] is not None
            ],
            "ends": [
                interval["end_seconds"]
                for interval in record["intervals"]
                if interval["end_seconds"] is not None
            ],
            "intervals": record["intervals"],
        }
        for record in records
    ]


def print_summary_files(
    data_dir: Path, patient: str
) -> dict[tuple[str, str, str], dict[str, Any]]:
    summary_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name.lower() == f"{patient}-summary.txt".lower()
    )
    if not summary_paths:
        print(f"\nPatient summary: not found for {patient} under {data_dir}")
        return {}

    summary_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for summary_path in summary_paths:
        print(f"\nPatient summary: {summary_path}")
        try:
            records = parse_patient_summary(summary_path)
        except OSError as exc:
            print(f"  Could not read summary: {exc}")
            continue

        print(f"  Recordings listed: {len(records)}")
        for record in records:
            expected_path = (summary_path.parent / record["filename"]).resolve()
            try:
                relative_path = expected_path.relative_to(data_dir).as_posix()
            except ValueError:
                relative_path = record["filename"]
            identity = (patient.casefold(), Path(record["filename"]).stem, relative_path)
            if identity in summary_records and summary_records[identity] != record:
                print(
                    f"  Conflicting duplicate summary entry for {record['filename']}"
                )
                continue
            summary_records[identity] = record
            starts, ends = record["starts"], record["ends"]
            intervals = [
                f"{item['start_seconds'] if item['start_seconds'] is not None else '?'}-"
                f"{item['end_seconds'] if item['end_seconds'] is not None else '?'} s"
                for item in record["intervals"]
            ]
            count = record["seizure_count"]
            if count is None:
                details = "seizure count/time entries not recognized"
            elif count == 0:
                details = "no seizures"
            elif intervals:
                details = f"{count} seizure(s): " + ", ".join(intervals)
                if len(intervals) != count:
                    details += (
                        f" (summary declares {count}; parsed {len(intervals)} "
                        "start/end pair(s))"
                    )
            else:
                details = f"{count} seizure(s); start/end times not found"
            print(f"    {record['filename']}: {details}")

        seizure_recordings = [
            record["filename"]
            for record in records
            if record["seizure_count"] is not None and record["seizure_count"] > 0
        ]
        print("  Recordings containing seizures:")
        if seizure_recordings:
            for filename in seizure_recordings:
                print(f"    {filename}")
        else:
            print("    None reported")
    return summary_records


def inspect_annotation_sidecar(path: Path) -> None:
    """Describe sidecar bytes without assuming an undocumented binary layout."""
    print(f"\nSeparate annotation file: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        print(f"  Could not read annotation file: {exc}")
        return

    print(f"  Size: {len(content)} bytes")
    try:
        decoded = content.decode("utf-8")
        is_text = "\x00" not in decoded
    except UnicodeDecodeError:
        decoded = ""
        is_text = False

    if is_text:
        print("  Format: UTF-8 text (structure not assumed)")
        for line in decoded.splitlines()[:8]:
            print(f"    {line}")
        if len(decoded.splitlines()) > 8:
            print("    ...")
    else:
        print("  Format: binary or non-UTF-8 data; not parsed as text")
        print(f"  First 64 bytes (hex): {content[:64].hex(' ')}")
        signatures = re.findall(rb"[ -~]{4,}", content[:512])
        for signature in signatures[:5]:
            print(f"  Printable header: {signature.decode('ascii', errors='replace')}")


def find_annotation_files(data_dir: Path, patient: str) -> list[Path]:
    candidates = []
    for path in data_dir.rglob(f"{patient}_*"):
        if not path.is_file() or path.suffix.lower() == ".edf":
            continue
        name = path.name.lower()
        if ".edf." in name or path.suffix.lower() in ANNOTATION_EXTENSIONS:
            candidates.append(path)
    return sorted(candidates)


def inspect_recording(path: Path, mne: Any, plot: bool = False) -> tuple[bool, str | None]:
    print(f"\nEDF: {path.name}")
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        channel_names = raw.ch_names
        channel_types = raw.get_channel_types()
        eeg_names = [
            name for name, channel_type in zip(channel_names, channel_types)
            if channel_type == "eeg"
        ]
        sampling_frequency = float(raw.info["sfreq"])
        sample_count = int(raw.n_times)
        duration_seconds = sample_count / sampling_frequency

        print("  Read successfully: yes")
        print(f"  EEG channels: {len(eeg_names)}")
        print(f"  EEG channel names: {', '.join(eeg_names) if eeg_names else '(none identified)'}")
        if len(eeg_names) != len(channel_names):
            print(f"  All channel names: {', '.join(channel_names)}")
        print(f"  Sampling frequency: {sampling_frequency:g} Hz")
        print(f"  Duration: {duration_seconds:.2f} seconds ({duration_seconds / 60:.2f} minutes)")
        print(f"  Samples: {sample_count}")

        if plot:
            if not eeg_names:
                print("  Plot skipped: no EEG channels were identified.")
            else:
                selected_names = eeg_names[:5]
                plot_raw = raw.copy().pick(selected_names)
                plot_raw.plot(
                    start=0,
                    duration=min(30.0, duration_seconds),
                    n_channels=len(selected_names),
                    scalings="auto",
                    title=f"{path.name} — first 30 seconds",
                )
        return True, None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"  Read successfully: no ({message})")
        return False, message


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect CHB-MIT EDF metadata, patient summary, and annotation sidecars."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root to search recursively (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--patient",
        default="chb01",
        help="Patient identifier used to find recordings and summary (default: chb01)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        metavar="EDF_FILENAME",
        help="Plot the first readable EDF, or a matching filename, for up to 30 seconds.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    print(f"Dataset directory: {data_dir}")
    if not data_dir.is_dir():
        print(f"Dataset directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    summary_records = print_summary_files(data_dir, args.patient)
    edf_files = sorted(data_dir.rglob(f"{args.patient}_*.edf"))
    print(f"\nEDF recordings found: {len(edf_files)}")
    if not edf_files:
        print("  No matching EDF recordings found.")
    local_edf_names = {path.name for path in edf_files}
    missing_edf_names = sorted(
        {
            record["filename"]
            for record in summary_records.values()
            if record["filename"] not in local_edf_names
        }
    )
    print(f"Summary recordings without a local EDF: {len(missing_edf_names)}")
    for filename in missing_edf_names:
        print(f"  {filename}")

    sidecars = find_annotation_files(data_dir, args.patient)
    print(f"Separate annotation files found: {len(sidecars)}")
    if sidecars:
        for sidecar in sidecars:
            inspect_annotation_sidecar(sidecar)
    else:
        print("  None found.")

    try:
        import mne
    except ImportError as exc:
        print(
            f"\nCannot inspect EDF headers because MNE could not be imported ({exc}). "
            "Install project dependencies with: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        for path in edf_files:
            print(f"\nEDF: {path.name}\n  Read successfully: no (MNE unavailable)")
        print(f"\nSummary: 0 successful, {len(edf_files)} failed recordings.")
        return 1

    successful = 0
    failed = 0
    plot_requested = args.plot is not None
    requested_plot_name = args.plot if args.plot else None
    plot_completed = False

    for path in edf_files:
        should_plot = (
            plot_requested
            and not plot_completed
            and (requested_plot_name is None or path.name == requested_plot_name)
        )
        readable, _ = inspect_recording(path, mne, plot=should_plot)
        if readable:
            successful += 1
            plot_completed = plot_completed or should_plot
        else:
            failed += 1
        relative_path = path.relative_to(data_dir).as_posix()
        identity = (args.patient.casefold(), path.stem, relative_path)
        record = summary_records.get(identity)
        if record is None:
            candidates = [
                candidate
                for key, candidate in summary_records.items()
                if key[0] == args.patient.casefold()
                and candidate["filename"].casefold() == path.name.casefold()
            ]
            if len(candidates) == 1:
                record = candidates[0]
            elif len(candidates) > 1:
                print("  Seizure timing: ambiguous duplicate summary entries")
                continue
        if record is None:
            print("  Seizure timing: not listed in a patient summary")
        elif record["seizure_count"]:
            print(
                f"  Seizure timing in summary: {record['seizure_count']} "
                f"seizure(s), starts={record['starts']}, ends={record['ends']} seconds"
            )
        else:
            print("  Seizure timing in summary: none reported")

    if plot_requested and not plot_completed:
        if requested_plot_name and not any(
            path.name == requested_plot_name for path in edf_files
        ):
            print(f"\nPlot target not found among discovered EDFs: {requested_plot_name}")
        elif edf_files:
            print("\nNo readable EDF was available to plot.")

    print(f"\nSummary: {successful} successful, {failed} failed recordings.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

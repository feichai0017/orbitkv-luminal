#!/usr/bin/env python3
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspaces/luminal")
BENCH_MANIFEST = Path("/tmp/egglog_repro/Cargo.toml")
PROGRAM_DIR = REPO / "scratch/egglog_repros/programs"
RESULT_DIR = REPO / "scratch/egglog_repros/results_parallelism_120s"
CSV_PATH = RESULT_DIR / "results.csv"
TIMEOUT_S = 120

MODELS = [
    "llama",
    "paged_llama",
    "qwen",
    "qwen3_moe",
    "gemma",
    "gemma4_moe",
    "whisper",
]

EGGLOG_REVS = [
    (
        "0a8cc35a6c68d0460c20449d5fa19ca3caba2923",
        "old",
    ),
    (
        "2e5657bbb2c1a90fba31002da61381815f891b6f",
        "new",
    ),
    (
        "345fa8d93ff904865c1b69cffbaeeedf6b88cc09",
        "pr857",
    ),
]

THREAD_MODES = [
    ("threads1", "1", "1"),
    ("threads8", "8", "8"),
    ("threads-default-30", "default-30", None),
]

FIELDNAMES = [
    "model",
    "egglog_rev",
    "rayon_threads",
    "timeout_s",
    "status",
    "seconds",
    "tuples",
    "log_path",
]


def parse_log(text: str) -> tuple[str, str]:
    seconds = ""
    tuples = ""
    if match := re.search(r"egglog total: ([0-9.]+)s", text):
        seconds = match.group(1)
    if match := re.search(r"tuples after: ([0-9]+)", text):
        tuples = match.group(1)
    return seconds, tuples


def load_existing_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (row["model"], row["egglog_rev"], row["rayon_threads"]): row
        for row in rows
    }


def write_rows(rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    ordered_rows = []
    for model in MODELS:
        for rev, _ in EGGLOG_REVS:
            for _, rayon_threads, _ in THREAD_MODES:
                row = rows.get((model, rev, rayon_threads))
                if row:
                    ordered_rows.append(row)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered_rows)


def run_cell(model: str, rev: str, feature: str, suffix: str, rayon_threads: str, env_threads: str | None):
    program = PROGRAM_DIR / f"{model}.egg"
    log_path = RESULT_DIR / f"{model}_{rev}_{suffix}.out"
    env = os.environ.copy()
    if env_threads is None:
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["RAYON_NUM_THREADS"] = env_threads

    cmd = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(BENCH_MANIFEST),
        "--features",
        feature,
        "--",
        str(program),
        "0",
    ]

    header = "\n".join(
        [
            f"model={model}",
            f"egglog_rev={rev}",
            f"rayon_threads={rayon_threads}",
            f"RAYON_NUM_THREADS={env.get('RAYON_NUM_THREADS', '<unset>')}",
            f"timeout_s={TIMEOUT_S}",
            f"command={' '.join(cmd)}",
            "",
        ]
    )

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    status = "ok"
    try:
        output, _ = proc.communicate(timeout=TIMEOUT_S)
        if proc.returncode != 0:
            status = f"error:{proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()

    elapsed_wall = time.monotonic() - start
    seconds, tuples = parse_log(output)
    if status == "ok" and not seconds:
        status = "error:missing-total"

    footer = f"\nwall_seconds={elapsed_wall:.3f}\nstatus={status}\n"
    log_path.write_text(header + output + footer)
    return {
        "model": model,
        "egglog_rev": rev,
        "rayon_threads": rayon_threads,
        "timeout_s": str(TIMEOUT_S),
        "status": status,
        "seconds": seconds,
        "tuples": tuples,
        "log_path": str(log_path.relative_to(REPO)),
    }


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows()

    total = len(MODELS) * len(EGGLOG_REVS) * len(THREAD_MODES)
    index = 0
    for model in MODELS:
        for rev, feature in EGGLOG_REVS:
            for suffix, rayon_threads, env_threads in THREAD_MODES:
                index += 1
                key = (model, rev, rayon_threads)
                if key in rows:
                    print(f"[{index:02}/{total}] skip {model} {rev} {rayon_threads}", flush=True)
                    continue
                print(f"[{index:02}/{total}] run  {model} {rev} {rayon_threads}", flush=True)
                rows[key] = run_cell(model, rev, feature, suffix, rayon_threads, env_threads)
                write_rows(rows)
                row = rows[key]
                printable_seconds = row["seconds"] or f">{TIMEOUT_S}"
                print(
                    f"          {row['status']} {printable_seconds}s tuples={row['tuples'] or '-'}",
                    flush=True,
                )

    write_rows(rows)
    if len(rows) != total:
        print(f"expected {total} rows, found {len(rows)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

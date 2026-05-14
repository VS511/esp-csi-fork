#!/usr/bin/env python3
"""
Scripted CSI capture runner — paired human/no-human recording.

Usage:
    python capture_runner.py -p /dev/ttyUSB0
    python capture_runner.py -p /dev/ttyUSB0 -o ../../../../../../data -d 15 --transition 10

Workflow per trial:
  1. Prompt for trial name (e.g. "1_up"), or "q" to quit.
  2. Spoken countdown 3-2-1.
  3. Wait a settle period (default 3 s) — packets discarded.
  4. Say "human" → capture for duration (default 15 s).
  5. Say "transition" → transition period (default 10 s) — packets discarded.
  6. Say "no human" → capture for duration (default 15 s).
  7. Say "done", print stats, run sanity checks.
  8. Loop back to (1).

Output CSV has an extra 'phase' column: "human" or "no_human".
"""

import argparse
import atexit
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

import serial

from sanity_check import _set_expected_duration, run_checks, print_report

DATA_COLUMNS_NAMES_C5C6 = [
    'type', 'id', 'mac', 'rssi', 'rate', 'noise_floor', 'fft_gain',
    'agc_gain', 'channel', 'local_timestamp', 'sig_len', 'rx_state',
    'len', 'first_word', 'data',
]
DATA_COLUMNS_NAMES = [
    'type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth',
    'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
    'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel',
    'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word',
    'data',
]

SOUND_DONE = '/System/Library/Sounds/Hero.aiff'

_audio_cache: dict[str, str] = {}
_cache_dir: str | None = None


def init_audio_cache(max_number: int):
    """Pre-generate .aiff files for all countdown numbers and spoken phrases."""
    global _cache_dir
    _cache_dir = tempfile.mkdtemp(prefix='csi_audio_')
    atexit.register(lambda: shutil.rmtree(_cache_dir, ignore_errors=True))

    phrases = [str(i) for i in range(1, max_number + 1)]
    phrases += ['human', 'no human', 'transition', 'done']

    print(f'Pre-generating {len(phrases)} audio clips ...', end=' ', flush=True)
    for phrase in phrases:
        safe_name = phrase.replace(' ', '_')
        path = os.path.join(_cache_dir, f'{safe_name}.aiff')
        subprocess.run(
            ['say', '-v', 'Samantha', '-o', path, '--', phrase],
            check=True, capture_output=True,
        )
        _audio_cache[phrase] = path
    print('done.')


def speak(text: str, wait: bool = True):
    """Play pre-generated audio clip, or fall back to live synthesis."""
    if text in _audio_cache:
        proc = subprocess.Popen(['afplay', _audio_cache[text]])
    else:
        proc = subprocess.Popen(['say', '-v', 'Samantha', text])
    if wait:
        proc.wait()


def speak_countdown(seconds: int = 3):
    """Speak a numbered countdown: 3, 2, 1 with wall-clock pacing."""
    for i in range(seconds, 0, -1):
        tick_start = time.monotonic()
        print(f'  {i} ...', flush=True)
        speak(str(i), wait=False)
        remaining = 1.0 - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)


def play(sound: str):
    os.system(f'afplay {sound} &')


def parse_csi_line(raw_line: str):
    """Return (csi_fields, base_header) or (None, None) on failure."""
    line = raw_line.lstrip("b'").rstrip("\\r\\n'")
    if 'CSI_DATA' not in line:
        return None, None

    reader = csv.reader(StringIO(line))
    fields = next(reader)

    if len(fields) == len(DATA_COLUMNS_NAMES_C5C6):
        header = DATA_COLUMNS_NAMES_C5C6
    elif len(fields) == len(DATA_COLUMNS_NAMES):
        header = DATA_COLUMNS_NAMES
    else:
        return None, None

    csi_data_len = int(fields[-3])
    try:
        csi_raw = json.loads(fields[-1])
    except json.JSONDecodeError:
        return None, None
    if csi_data_len != len(csi_raw):
        return None, None

    return fields, header


def discard_serial(ser, duration_s: float):
    """Read and discard serial data for the given duration."""
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        ser.readline()


def countdown_with_discard(ser, seconds: int):
    """Speak a numbered countdown while discarding serial packets."""
    countdown_start = time.monotonic()
    for idx, i in enumerate(range(seconds, 0, -1)):
        tick_deadline = countdown_start + idx + 1.0
        print(f'  {i} ...', flush=True)
        speak(str(i), wait=False)
        while time.monotonic() < tick_deadline:
            ser.readline()


def capture_phase(ser, writer, csv_fd, log_fd, phase_label: str,
                  duration_s: float, header_written: bool, local_ts_idx):
    """
    Capture CSI packets for *duration_s* seconds, tagging each row with
    *phase_label*. Returns (saved_rows, header_written, local_ts_idx).
    """
    saved_rows = 0
    capture_start = time.monotonic()
    capture_end = capture_start + duration_s
    last_status = capture_start

    countdown_nums = list(range(int(duration_s), 0, -3))
    announce_times = [capture_start + (duration_s - n) for n in countdown_nums]
    announce_idx = 0

    while time.monotonic() < capture_end:
        now = time.monotonic()
        if announce_idx < len(announce_times) and now >= announce_times[announce_idx]:
            num = countdown_nums[announce_idx]
            print(f'    [{phase_label}] {num}s remaining', flush=True)
            speak(str(num), wait=False)
            announce_idx += 1

        raw = str(ser.readline())
        if not raw:
            continue

        fields, base_header = parse_csi_line(raw)
        if fields is None:
            stripped = raw.lstrip("b'").rstrip("\\r\\n'")
            log_fd.write(stripped + '\n')
            log_fd.flush()
            continue

        if not header_written:
            local_ts_idx = (
                base_header.index('local_timestamp')
                if 'local_timestamp' in base_header else None
            )
            extended = list(base_header) + [
                'local_timestamp_sec', 'host_unix_timestamp', 'phase',
            ]
            writer.writerow(extended)
            csv_fd.flush()
            header_written = True

        local_ts_sec = ''
        host_ts = time.time()
        if local_ts_idx is not None:
            try:
                local_ts_sec = float(fields[local_ts_idx]) / 1e6
            except (ValueError, TypeError):
                local_ts_sec = ''

        writer.writerow(fields + [local_ts_sec, host_ts, phase_label])
        saved_rows += 1

        now = time.monotonic()
        elapsed = now - capture_start
        if now - last_status >= 5.0:
            remaining = capture_end - now
            print(f'    [{phase_label}] {elapsed:.0f}s elapsed, '
                  f'{remaining:.0f}s left — rows={saved_rows}',
                  flush=True)
            last_status = now

    csv_fd.flush()
    return saved_rows, header_written, local_ts_idx


def run_paired_capture(ser, csv_path: Path, log_path: Path,
                       duration_s: float, settle_s: float,
                       transition_s: float):
    """
    Run a paired capture: human phase → transition → no_human phase.
    Returns total saved rows.
    """
    with open(csv_path, 'w', newline='') as csv_fd, \
         open(log_path, 'w') as log_fd:

        writer = csv.writer(csv_fd)
        header_written = False
        local_ts_idx = None
        total_rows = 0

        # --- Settle period ---
        print(f'  Settling for {settle_s:.1f}s (discarding packets) ...', flush=True)
        discard_serial(ser, settle_s)
        print('  Settle complete.', flush=True)

        # --- Phase 1: human ---
        print(f'  [HUMAN] Capturing for {duration_s:.0f}s ...', flush=True)
        speak("human")
        rows, header_written, local_ts_idx = capture_phase(
            ser, writer, csv_fd, log_fd, 'human',
            duration_s, header_written, local_ts_idx,
        )
        total_rows += rows
        print(f'  [HUMAN] Done — {rows} rows captured.', flush=True)

        # --- Transition ---
        print(f'  Transition: {transition_s:.0f}s (discarding packets) ...', flush=True)
        speak("transition")
        countdown_with_discard(ser, int(transition_s))
        print('  Transition complete.', flush=True)

        # --- Phase 2: no human ---
        print(f'  [NO HUMAN] Capturing for {duration_s:.0f}s ...', flush=True)
        speak("no human")
        rows, header_written, local_ts_idx = capture_phase(
            ser, writer, csv_fd, log_fd, 'no_human',
            duration_s, header_written, local_ts_idx,
        )
        total_rows += rows
        print(f'  [NO HUMAN] Done — {rows} rows captured.', flush=True)

    speak("done")
    play(SOUND_DONE)
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description='Scripted CSI capture runner (paired human/no-human)')
    parser.add_argument('-p', '--port', required=True,
                        help='Serial port of the CSI receiver')
    parser.add_argument('-o', '--output-dir', default='./data',
                        help='Directory to save CSV files (default: ./data)')
    parser.add_argument('-d', '--duration', type=float, default=15.0,
                        help='Capture duration per phase in seconds (default: 15)')
    parser.add_argument('--settle', type=float, default=3.0,
                        help='Settle/discard period in seconds (default: 3)')
    parser.add_argument('--transition', type=float, default=10.0,
                        help='Transition period between phases in seconds (default: 10)')
    parser.add_argument('-b', '--baud', type=int, default=921600,
                        help='Serial baud rate (default: 921600)')
    args = parser.parse_args()

    max_num = int(max(args.duration, args.transition))
    init_audio_cache(max_num)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Opening serial port {args.port} at {args.baud} baud ...')
    ser = serial.Serial(port=args.port, baudrate=args.baud,
                        bytesize=8, parity='N', stopbits=1, timeout=0.05)
    if not ser.isOpen():
        print('ERROR: could not open serial port.')
        sys.exit(1)
    print('Serial port open.\n')

    trial_num = 0
    while True:
        name = input('Trial name (e.g. "1_up"), or "q" to quit: ').strip()
        if name.lower() == 'q':
            break
        if not name:
            print('Empty name, try again.')
            continue

        csv_path = out_dir / f'{name}.csv'
        log_dir = out_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{name}_log.txt'
        if csv_path.exists():
            overwrite = input(f'  {csv_path} already exists. Overwrite? [y/N] ')
            if overwrite.strip().lower() != 'y':
                continue

        trial_num += 1
        print(f'\n--- Trial {trial_num}: {name} ---')

        # Spoken countdown before starting
        speak_countdown(3)

        rows = run_paired_capture(
            ser, csv_path, log_path,
            duration_s=args.duration,
            settle_s=args.settle,
            transition_s=args.transition,
        )

        print(f'  Done!  Saved {rows} total rows to {csv_path}')
        print(f'  Log file: {log_path}')

        _set_expected_duration(args.duration * 2)
        results, all_ok = run_checks(csv_path)
        print_report(csv_path, results, all_ok)

    ser.close()
    print('Serial port closed. Goodbye.')


if __name__ == '__main__':
    main()

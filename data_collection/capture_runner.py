#!/usr/bin/env python3
"""
Scripted CSI capture runner — paired human/no-human recording.

Usage:
    python capture_runner.py -p /dev/ttyUSB0
    python capture_runner.py -p /dev/ttyUSB0 -d 15 --transition 10
    python capture_runner.py -p /dev/ttyUSB0 --batch --orientations 6 --reposition 10

Data is saved to <repo>/data/<MM-DD-YY-HH-MM>/ by default.
Use -o to override the base directory (a timestamped sub-folder is still created).

Workflow per trial (single mode):
  1. Prompt for trial name (e.g. "1_up"), or "q" to quit.
  2. Spoken countdown 3-2-1.
  3. Settle → human capture → transition → no_human capture.
  4. Say "done", print stats, run sanity checks.
  5. Loop back to (1).

Workflow per trial (batch mode):
  For each orientation 1..N:
    a. Say "orientation N".
    b. Settle → human capture → transition → no_human capture.
    c. If not last orientation: "reposition" → spoken countdown (default 10 s).
  All data saved to one CSV with an extra 'orientation' column.
  A metadata sidecar JSON is written alongside the CSV.

Output CSV columns: base CSI fields + local_timestamp_sec, host_unix_timestamp,
phase ("human"/"no_human"), and orientation (1..N, batch mode only).
"""

import argparse
import atexit
import csv
import datetime
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

from sanity_check import (
    _set_expected_duration, run_checks, print_report,
    run_checks_batch, print_report_batch,
)

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
SOUND_WARNING = '/System/Library/Sounds/Basso.aiff'
PACKET_SILENCE_WARN_S = 3.0

_audio_cache: dict[str, str] = {}
_cache_dir: str | None = None


def init_audio_cache(max_number: int, num_orientations: int = 0):
    """Pre-generate .aiff files for all countdown numbers and spoken phrases."""
    global _cache_dir
    _cache_dir = tempfile.mkdtemp(prefix='csi_audio_')
    atexit.register(lambda: shutil.rmtree(_cache_dir, ignore_errors=True))

    phrases = [str(i) for i in range(1, max_number + 1)]
    phrases += ['human', 'no human', 'transition', 'done', 'reposition']
    for i in range(1, num_orientations + 1):
        phrases.append(f'orientation {i}')

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
                  duration_s: float, header_written: bool, local_ts_idx,
                  orientation: int | None = None):
    """
    Capture CSI packets for *duration_s* seconds, tagging each row with
    *phase_label* and optionally *orientation*.
    Returns (saved_rows, header_written, local_ts_idx).

    Plays a warning sound if no valid CSI packet arrives for
    PACKET_SILENCE_WARN_S seconds.
    """
    saved_rows = 0
    capture_start = time.monotonic()
    capture_end = capture_start + duration_s
    last_status = capture_start
    last_valid_packet = capture_start
    warned_silence = False

    countdown_nums = list(range(int(duration_s), 0, -3))
    announce_times = [capture_start + (duration_s - n) for n in countdown_nums]
    announce_idx = 0

    while time.monotonic() < capture_end:
        now = time.monotonic()

        if not warned_silence and (now - last_valid_packet) >= PACKET_SILENCE_WARN_S:
            print(f'    WARNING: No CSI packets for '
                  f'{PACKET_SILENCE_WARN_S:.0f}s!', flush=True)
            play(SOUND_WARNING)
            warned_silence = True

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

        last_valid_packet = time.monotonic()
        if warned_silence:
            print(f'    CSI packets resumed.', flush=True)
            warned_silence = False

        if not header_written:
            local_ts_idx = (
                base_header.index('local_timestamp')
                if 'local_timestamp' in base_header else None
            )
            extended = list(base_header) + [
                'local_timestamp_sec', 'host_unix_timestamp', 'phase',
            ]
            if orientation is not None:
                extended.append('orientation')
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

        row = fields + [local_ts_sec, host_ts, phase_label]
        if orientation is not None:
            row.append(orientation)
        writer.writerow(row)
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


def run_trial(ser, csv_path: Path, log_path: Path,
              duration_s: float, settle_s: float, transition_s: float,
              batch: bool, orientations: int, reposition_s: float,
              args):
    """
    Run a full trial: single orientation or batch multi-orientation.

    Returns (total_rows, summary, meta_path) where summary is a list of
    (orientation, human_rows, no_human_rows) tuples.
    Writes a metadata sidecar JSON alongside the CSV.
    """
    num_orientations = orientations if batch else 1
    summary = []
    total_rows = 0

    meta = {
        'trial_name': csv_path.stem,
        'start_time': datetime.datetime.now().isoformat(),
        'parameters': {
            'duration_s': duration_s,
            'settle_s': settle_s,
            'transition_s': transition_s,
            'batch': batch,
            'orientations': num_orientations,
            'reposition_s': reposition_s if batch else None,
            'port': args.port,
            'baud': args.baud,
        },
        'phases': [],
        'interrupted': False,
        'total_rows': 0,
    }
    meta_path = csv_path.parent / f'{csv_path.stem}_meta.json'

    with open(csv_path, 'w', newline='') as csv_fd, \
         open(log_path, 'w') as log_fd:

        writer = csv.writer(csv_fd)
        header_written = False
        local_ts_idx = None

        try:
            for orient_idx in range(1, num_orientations + 1):
                orient_label = orient_idx if batch else None

                if batch:
                    print(f'\n  === Orientation {orient_idx}/{num_orientations} ===',
                          flush=True)
                    speak(f'orientation {orient_idx}')

                # --- Settle ---
                print(f'  Settling for {settle_s:.1f}s (discarding packets) ...',
                      flush=True)
                settle_start = datetime.datetime.now().isoformat()
                discard_serial(ser, settle_s)
                print('  Settle complete.', flush=True)

                # --- Human ---
                print(f'  [HUMAN] Capturing for {duration_s:.0f}s ...', flush=True)
                speak("human")
                human_start = datetime.datetime.now().isoformat()
                h_rows, header_written, local_ts_idx = capture_phase(
                    ser, writer, csv_fd, log_fd, 'human',
                    duration_s, header_written, local_ts_idx,
                    orientation=orient_label,
                )
                human_end = datetime.datetime.now().isoformat()
                total_rows += h_rows
                print(f'  [HUMAN] Done — {h_rows} rows captured.', flush=True)

                # --- Transition ---
                print(f'  Transition: {transition_s:.0f}s (discarding packets) ...',
                      flush=True)
                speak("transition")
                transition_start = datetime.datetime.now().isoformat()
                countdown_with_discard(ser, int(transition_s))
                print('  Transition complete.', flush=True)

                # --- No human ---
                print(f'  [NO HUMAN] Capturing for {duration_s:.0f}s ...', flush=True)
                speak("no human")
                no_human_start = datetime.datetime.now().isoformat()
                nh_rows, header_written, local_ts_idx = capture_phase(
                    ser, writer, csv_fd, log_fd, 'no_human',
                    duration_s, header_written, local_ts_idx,
                    orientation=orient_label,
                )
                no_human_end = datetime.datetime.now().isoformat()
                total_rows += nh_rows
                print(f'  [NO HUMAN] Done — {nh_rows} rows captured.', flush=True)

                summary.append((orient_idx, h_rows, nh_rows))
                meta['phases'].append({
                    'orientation': orient_idx,
                    'settle_start': settle_start,
                    'human_start': human_start,
                    'human_end': human_end,
                    'human_rows': h_rows,
                    'transition_start': transition_start,
                    'no_human_start': no_human_start,
                    'no_human_end': no_human_end,
                    'no_human_rows': nh_rows,
                })

                # --- Reposition (between orientations, not after last) ---
                if batch and orient_idx < num_orientations:
                    print(f'\n  Reposition: {reposition_s:.0f}s '
                          f'— adjust RX orientation ...', flush=True)
                    speak("reposition")
                    countdown_with_discard(ser, int(reposition_s))
                    print('  Reposition complete.', flush=True)

        except KeyboardInterrupt:
            print('\n\n  *** INTERRUPTED — saving partial data ***', flush=True)
            meta['interrupted'] = True

    speak("done")
    play(SOUND_DONE)

    meta['end_time'] = datetime.datetime.now().isoformat()
    meta['total_rows'] = total_rows
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return total_rows, summary, meta_path


def print_batch_summary(summary):
    """Print a table summarizing rows per orientation and phase."""
    if len(summary) <= 1:
        return
    print(f'\n  {"Orientation":>12s}  {"Human":>8s}  {"No Human":>10s}  {"Total":>8s}')
    print(f'  {"---":>12s}  {"---":>8s}  {"---":>10s}  {"---":>8s}')
    grand_human = 0
    grand_no_human = 0
    for orient, h_rows, nh_rows in summary:
        total = h_rows + nh_rows
        grand_human += h_rows
        grand_no_human += nh_rows
        print(f'  {orient:>12d}  {h_rows:>8d}  {nh_rows:>10d}  {total:>8d}')
    print(f'  {"---":>12s}  {"---":>8s}  {"---":>10s}  {"---":>8s}')
    print(f'  {"TOTAL":>12s}  {grand_human:>8d}  {grand_no_human:>10d}'
          f'  {grand_human + grand_no_human:>8d}')


def main():
    parser = argparse.ArgumentParser(
        description='Scripted CSI capture runner (paired human/no-human)')
    parser.add_argument('-p', '--port', required=True,
                        help='Serial port of the CSI receiver')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Base directory for data (default: <repo>/data). '
                             'A timestamped sub-folder is created per run.')
    parser.add_argument('-d', '--duration', type=float, default=15.0,
                        help='Capture duration per phase in seconds (default: 15)')
    parser.add_argument('--settle', type=float, default=3.0,
                        help='Settle/discard period in seconds (default: 3)')
    parser.add_argument('--transition', type=float, default=10.0,
                        help='Transition period between phases in seconds '
                             '(default: 10)')
    parser.add_argument('-b', '--baud', type=int, default=921600,
                        help='Serial baud rate (default: 921600)')
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch multi-orientation mode')
    parser.add_argument('--orientations', type=int, default=6,
                        help='Number of orientations per trial in batch mode '
                             '(default: 6)')
    parser.add_argument('--reposition', type=float, default=10.0,
                        help='Reposition countdown between orientations in '
                             'seconds (default: 10)')
    args = parser.parse_args()

    num_orient = args.orientations if args.batch else 1
    max_num = int(max(args.duration, args.transition, args.reposition))
    init_audio_cache(max_num, num_orientations=num_orient)

    script_dir = Path(__file__).resolve().parent
    base_dir = Path(args.output_dir) if args.output_dir else script_dir.parent / 'data'
    run_stamp = datetime.datetime.now().strftime('%m-%d-%y-%H-%M')
    out_dir = base_dir / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Run folder: {out_dir}')

    print(f'Opening serial port {args.port} at {args.baud} baud ...')
    ser = serial.Serial(port=args.port, baudrate=args.baud,
                        bytesize=8, parity='N', stopbits=1, timeout=0.05)
    if not ser.isOpen():
        print('ERROR: could not open serial port.')
        sys.exit(1)
    print('Serial port open.')
    if args.batch:
        print(f'Batch mode: {args.orientations} orientations, '
              f'{args.reposition:.0f}s reposition window')
    print()

    trial_num = 0
    try:
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
                overwrite = input(
                    f'  {csv_path} already exists. Overwrite? [y/N] ')
                if overwrite.strip().lower() != 'y':
                    continue

            trial_num += 1
            print(f'\n--- Trial {trial_num}: {name} ---')
            speak_countdown(3)

            rows, summary, meta_path = run_trial(
                ser, csv_path, log_path,
                duration_s=args.duration,
                settle_s=args.settle,
                transition_s=args.transition,
                batch=args.batch,
                orientations=args.orientations,
                reposition_s=args.reposition,
                args=args,
            )

            print(f'\n  Saved {rows} total rows to {csv_path}')
            print(f'  Log: {log_path}')
            print(f'  Metadata: {meta_path}')

            if args.batch:
                print_batch_summary(summary)

            if args.batch:
                group_results, all_ok = run_checks_batch(
                    csv_path, args.duration)
                print_report_batch(csv_path, group_results, all_ok)
            else:
                _set_expected_duration(args.duration * 2)
                results, all_ok = run_checks(csv_path)
                print_report(csv_path, results, all_ok)

    except KeyboardInterrupt:
        print('\nInterrupted between trials.')

    ser.close()
    print('Serial port closed. Goodbye.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Post-capture sanity checks for CSI CSV files.

Usage — single file:
    python sanity_check.py data/1_up_human.csv

Usage — whole directory:
    python sanity_check.py data/

Checks performed:
  1. Duration   — is the capture close to the expected length?
  2. Drop rate  — packet-ID gaps exceeding a threshold?
  3. Rate stability — does any 1-second window fall below a packet-count floor?
  4. AGC / FFT gain stability — are gains constant across the capture?
  5. RSSI stability — is RSSI variance low?
  6. Amplitude sanity — basic per-subcarrier amplitude stats.

Batch-mode CSVs (with 'orientation' column) are checked per
(orientation, phase) group so that intentional discard gaps between
capture windows don't inflate drop/duration/rate metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Thresholds (tune these to your setup)
# ──────────────────────────────────────────────
EXPECTED_DURATION_S = 30.0
DURATION_TOLERANCE_S = 1.5
MAX_DROP_PCT = 2.0
MIN_PACKETS_PER_SEC = 80
MAX_AGC_STD = 1.0
MAX_FFT_STD = 1.0
MAX_RSSI_STD_NO_HUMAN = 2.0
MAX_RSSI_STD_HUMAN = 4.0


def _set_expected_duration(val):
    global EXPECTED_DURATION_S
    EXPECTED_DURATION_S = val


def _safe_numeric(series):
    return pd.to_numeric(series, errors='coerce').dropna()


def _get_timestamp_series(df):
    if 'local_timestamp_sec' in df.columns:
        return _safe_numeric(df['local_timestamp_sec']).sort_values().reset_index(drop=True)
    if 'host_unix_timestamp' in df.columns:
        return _safe_numeric(df['host_unix_timestamp']).sort_values().reset_index(drop=True)
    return pd.Series(dtype=float)


def check_duration(ts):
    duration = ts.iloc[-1] - ts.iloc[0]
    ok = abs(duration - EXPECTED_DURATION_S) <= DURATION_TOLERANCE_S
    return ok, duration


def check_drops(ids):
    ids_sorted = ids.sort_values().to_numpy()
    diffs = np.diff(ids_sorted)
    dropped = int(np.maximum(diffs - 1, 0).sum())
    total = len(ids_sorted) + dropped
    pct = 100.0 * dropped / total if total else 0.0
    ok = pct <= MAX_DROP_PCT
    return ok, dropped, pct


def check_rate_stability(ts, window_s=1.0):
    ts_arr = ts.to_numpy()
    t0 = ts_arr[0]
    t1 = ts_arr[-1]
    n_full_windows = int(np.floor((t1 - t0) / window_s))
    if n_full_windows < 1:
        return True, len(ts_arr), 0.0
    min_count = len(ts_arr)
    worst_window_offset = 0.0
    for i in range(n_full_windows):
        lo = t0 + i * window_s
        hi = lo + window_s
        count = int(np.sum((ts_arr >= lo) & (ts_arr < hi)))
        if count < min_count:
            min_count = count
            worst_window_offset = i * window_s
    ok = min_count >= MIN_PACKETS_PER_SEC
    return ok, min_count, worst_window_offset


def check_gain_stability(col, col_name, threshold):
    arr = col.to_numpy().astype(float)
    std = float(np.std(arr))
    ok = std <= threshold
    return ok, std, float(np.mean(arr))


def check_rssi_stability(rssi, has_human):
    arr = rssi.to_numpy().astype(float)
    std = float(np.std(arr))
    thresh = MAX_RSSI_STD_HUMAN if has_human else MAX_RSSI_STD_NO_HUMAN
    ok = std <= thresh
    return ok, std, float(np.mean(arr))


def check_amplitude(df):
    """Compute mean subcarrier amplitude from the 'data' column."""
    amps = []
    for raw in df['data']:
        try:
            vals = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        iq = np.array(vals, dtype=float)
        n_sc = len(iq) // 2
        if n_sc == 0:
            continue
        comp = iq[1::2] + 1j * iq[0::2]
        amps.append(np.abs(comp))
    if not amps:
        return False, None, None
    amp_matrix = np.array(amps)
    mean_per_sc = amp_matrix.mean(axis=0)
    overall_mean = float(mean_per_sc.mean())
    overall_std = float(mean_per_sc.std())
    return True, overall_mean, overall_std


def _run_checks_on_df(df, expected_duration, has_human):
    """Core check logic on a (possibly subset) DataFrame. Returns (results, all_ok)."""
    results = {}
    all_ok = True
    saved = EXPECTED_DURATION_S
    _set_expected_duration(expected_duration)

    ts = _get_timestamp_series(df)

    if len(ts) >= 2:
        ok, dur = check_duration(ts)
        results['duration'] = {'pass': ok, 'value_s': round(dur, 3)}
        all_ok &= ok
    else:
        results['duration'] = {'pass': False, 'value_s': None}
        all_ok = False

    if 'id' in df.columns:
        ids = _safe_numeric(df['id']).astype(int)
        ok, dropped, pct = check_drops(ids)
        results['drops'] = {
            'pass': ok,
            'dropped_packets': dropped,
            'drop_pct': round(pct, 2),
        }
        all_ok &= ok
    else:
        results['drops'] = {'pass': False, 'dropped_packets': None, 'drop_pct': None}
        all_ok = False

    if len(ts) >= 2:
        ok, min_count, worst_offset = check_rate_stability(ts)
        results['rate_stability'] = {
            'pass': ok,
            'min_packets_in_1s_window': min_count,
            'worst_window_offset_s': round(float(worst_offset), 1),
        }
        all_ok &= ok

    if 'agc_gain' in df.columns:
        col = _safe_numeric(df['agc_gain'])
        if len(col) > 1:
            ok, std, mean = check_gain_stability(col, 'agc_gain', MAX_AGC_STD)
            results['agc_gain'] = {
                'pass': ok, 'std': round(std, 3), 'mean': round(mean, 2),
            }
            all_ok &= ok

    if 'fft_gain' in df.columns:
        col = _safe_numeric(df['fft_gain'])
        if len(col) > 1:
            ok, std, mean = check_gain_stability(col, 'fft_gain', MAX_FFT_STD)
            results['fft_gain'] = {
                'pass': ok, 'std': round(std, 3), 'mean': round(mean, 2),
            }
            all_ok &= ok

    if 'rssi' in df.columns:
        rssi = _safe_numeric(df['rssi'])
        if len(rssi) > 1:
            ok, std, mean = check_rssi_stability(rssi, has_human)
            results['rssi'] = {
                'pass': ok, 'std': round(std, 3), 'mean': round(mean, 2),
            }
            all_ok &= ok

    if 'data' in df.columns:
        ok, amp_mean, amp_std = check_amplitude(df)
        results['amplitude'] = {
            'pass': ok,
            'mean_subcarrier_amplitude': round(amp_mean, 2) if amp_mean else None,
            'std_across_subcarriers': round(amp_std, 2) if amp_std else None,
        }

    _set_expected_duration(saved)
    return results, all_ok


def run_checks(csv_path: Path):
    """Run all sanity checks on one CSV file (single-mode).
    Returns (results_dict, all_passed)."""
    df = pd.read_csv(csv_path)
    has_human = 'no_human' not in csv_path.stem
    return _run_checks_on_df(df, EXPECTED_DURATION_S, has_human)


def run_checks_batch(csv_path: Path, phase_duration_s: float):
    """Run sanity checks per (orientation, phase) group for batch-mode CSVs.

    Returns (group_results, all_ok) where group_results is a list of
    (label, results_dict, group_ok) tuples.
    """
    df = pd.read_csv(csv_path)
    group_results = []
    all_ok = True

    if 'orientation' not in df.columns or 'phase' not in df.columns:
        results, ok = _run_checks_on_df(df, phase_duration_s, True)
        group_results.append(('all', results, ok))
        return group_results, ok

    for (orient, phase), group_df in df.groupby(['orientation', 'phase'], sort=True):
        label = f'orient_{orient}_{phase}'
        has_human = phase == 'human'
        results, ok = _run_checks_on_df(
            group_df.reset_index(drop=True),
            phase_duration_s,
            has_human,
        )
        group_results.append((label, results, ok))
        all_ok &= ok

    return group_results, all_ok


def print_report(csv_path, results, all_ok):
    tag = 'PASS' if all_ok else 'FAIL'
    print(f'\n{"=" * 60}')
    print(f'  {tag}  {csv_path.name}')
    print(f'{"=" * 60}')
    for check_name, info in results.items():
        status = 'OK' if info.get('pass') else 'XX'
        detail_parts = [f'{k}={v}' for k, v in info.items() if k != 'pass']
        detail = ', '.join(detail_parts)
        print(f'  [{status}] {check_name:20s}  {detail}')
    print()


def print_report_batch(csv_path, group_results, all_ok):
    tag = 'PASS' if all_ok else 'FAIL'
    print(f'\n{"=" * 60}')
    print(f'  {tag}  {csv_path.name}  (batch)')
    print(f'{"=" * 60}')
    for label, results, group_ok in group_results:
        gtag = 'PASS' if group_ok else 'FAIL'
        print(f'\n  --- {label} [{gtag}] ---')
        for check_name, info in results.items():
            status = 'OK' if info.get('pass') else 'XX'
            detail_parts = [f'{k}={v}' for k, v in info.items() if k != 'pass']
            detail = ', '.join(detail_parts)
            print(f'    [{status}] {check_name:20s}  {detail}')

    failed = [label for label, _, ok in group_results if not ok]
    n_groups = len(group_results)
    n_pass = n_groups - len(failed)
    print(f'\n  {n_pass}/{n_groups} groups passed all checks.')
    if failed:
        print(f'  Failed: {", ".join(failed)}')
    print()


def main():
    parser = argparse.ArgumentParser(description='CSI capture sanity checks')
    parser.add_argument('path', help='CSV file or directory of CSVs to check')
    parser.add_argument('--expected-duration', type=float, default=EXPECTED_DURATION_S,
                        help=f'Expected capture duration (default: {EXPECTED_DURATION_S})')
    args = parser.parse_args()

    if args.expected_duration != EXPECTED_DURATION_S:
        _set_expected_duration(args.expected_duration)

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob('*.csv'))
    else:
        print(f'ERROR: {target} is not a file or directory.')
        sys.exit(1)

    if not files:
        print('No CSV files found.')
        sys.exit(1)

    summary = []
    for f in files:
        df_peek = pd.read_csv(f, nrows=1)
        if 'orientation' in df_peek.columns:
            group_results, all_ok = run_checks_batch(f, EXPECTED_DURATION_S)
            print_report_batch(f, group_results, all_ok)
        else:
            results, all_ok = run_checks(f)
            print_report(f, results, all_ok)
        summary.append((f.name, all_ok))

    if len(summary) > 1:
        print('─' * 60)
        print('Summary:')
        for name, ok in summary:
            print(f'  {"PASS" if ok else "FAIL"}  {name}')
        n_pass = sum(ok for _, ok in summary)
        print(f'\n  {n_pass}/{len(summary)} files passed all checks.')
        print()


if __name__ == '__main__':
    main()

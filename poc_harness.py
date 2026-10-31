#!/usr/bin/env python3
"""
poc_harness.py - Proof of Concept Network Attack Onset Forecasting Harness.

Team: TECH ROOKIES, PSC26153 (II-year Engineering Students)
Scope: OFFLINE REPLAY of the public CIC-IDS2017 dataset only.
       No live traffic, no real attackers, no personal data.
Target Environment: Python 3.10.12 (pandas, scikit-learn, joblib).

Purpose:
Aggregates packet/flow data into fixed-size time windows, detects attack episodes,
trains a machine learning classifier on development data (dev) to forecast attack
episode onsets, evaluates forecasting quality and latency on evaluation data (eval),
and saves model bundles and audit logs.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CANONICAL FEATURE NAMES AND EXPLANATIONS (II-year Student Friendly)
# ==============================================================================
# Canonical features computed for each time window:
# 1. flow_count: Number of network flows starting in this window. High flow rates
#    often indicate reconnaissance (scanning) or volumetric flood preparation.
# 2. syn_rate: Count of SYN flags divided by window length in seconds. High SYN rates
#    strongly signal TCP SYN flood attacks or half-open port scans.
# 3. total_fwd_packets: Cumulative forward packets across all flows in the window.
# 4. total_bwd_packets: Cumulative backward packets across all flows in the window.
# 5. mean_flow_duration: Average duration of flows in the window. Short durations
#    are typical of aggressive scans/floods; long durations reflect exfiltration/sessions.
# 6. mean_packet_length: Average size of packets in bytes. Micro-packets or uniform
#    packet sizes often characterize automated botnet probes or flood traffic.
# 7. unique_src_count: Number of distinct source addresses/ports. Multiple sources
#    indicate distributed attacks (DDoS); single aggressive source indicates targeted attack.
# 8. unique_dst_count: Number of distinct destination addresses/ports. High count
#    strongly indicates horizontal port/host scanning.
# 9. mean_total_packets: Average packet count per flow (fwd + bwd packets).
CANONICAL_FEATURES = [
    "flow_count",
    "syn_rate",
    "total_fwd_packets",
    "total_bwd_packets",
    "mean_flow_duration",
    "mean_packet_length",
    "unique_src_count",
    "unique_dst_count",
    "mean_total_packets",
]


# ==============================================================================
# TIMESTAMP PARSING AND HEADER HANDLING
# ==============================================================================
def resolve_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Inspects CSV columns at runtime, strips whitespace, and maps them to canonical
    feature building blocks.

    Fails with a clear, explicit error message if any essential column is missing.
    """
    clean_cols = {col.strip(): col for col in columns}

    # Alias candidates for each required component in CIC-IDS2017
    alias_map = {
        "timestamp": ["Timestamp", "timestamp", "Time"],
        "label": ["Label", "label", "Class"],
        "syn_flag": ["SYN Flag Count", "SYN Flag Cnt", "syn_flag_count", "SYN"],
        "fwd_packets": ["Total Fwd Packets", "Total Fwd Packet", "total_fwd_packets"],
        "bwd_packets": [
            "Total Backward Packets",
            "Total Bwd Packets",
            "Total Bwd Packet",
            "total_bwd_packets",
        ],
        "flow_duration": ["Flow Duration", "flow_duration", "Duration"],
        "packet_length": [
            "Average Packet Size",
            "Packet Length Mean",
            "Avg Packet Size",
            "average_packet_size",
        ],
        "source": ["Source IP", "Src IP", "Source Port", "Src Port"],
        "destination": ["Destination IP", "Dst IP", "Destination Port", "Dst Port"],
    }

    resolved = {}
    missing = []
    for key, candidates in alias_map.items():
        found = None
        for candidate in candidates:
            if candidate in clean_cols:
                found = clean_cols[candidate]
                break
        if found is not None:
            resolved[key] = found
        else:
            missing.append(f"{key} (expected one of: {candidates})")

    if missing:
        raise ValueError(
            f"Dataset header validation failed! Missing required columns: {missing}.\n"
            f"Available columns in CSV header: {list(clean_cols.keys())}"
        )

    return resolved


def parse_cic_timestamps(ts_series: pd.Series) -> Tuple[pd.Series, str]:
    """
    Inspects real timestamp format at runtime and parses it deterministically.

    CIC-IDS2017 timestamp quirks:
    - In some files, dates are 'dd/MM/yyyy HH:mm:ss' or 'd/M/yyyy H:mm'.
    - In other files, 12-hour clocks with AM/PM ('dd/MM/yyyy hh:mm:ss a') appear.
    - In some afternoon exports, timestamps roll over from 12:xx to 01:xx without AM/PM.
    
    This function inspects sample values, determines whether 24h, 12h-AM/PM, or ambiguous
    rollover exists, parses deterministically, and records the exact strategy.
    """
    sample_values = ts_series.dropna().astype(str).str.strip().head(500).tolist()
    if not sample_values:
        raise ValueError("Cannot parse timestamps: Timestamp column is empty.")

    # Check for AM/PM indicators
    has_ampm = any(re.search(r"\b(AM|PM|am|pm)\b", s) for s in sample_values)

    strategy_desc = ""
    parsed = None

    # Candidate datetime format patterns
    if has_ampm:
        formats = [
            "%d/%m/%Y %I:%M:%S %p",
            "%d/%m/%Y %I:%M %p",
            "%m/%d/%Y %I:%M:%S %p",
            "%Y-%m-%d %I:%M:%S %p",
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(ts_series, format=fmt, errors="raise")
                strategy_desc = f"Deterministic parse using format '{fmt}' with explicit AM/PM."
                break
            except (ValueError, TypeError):
                continue
    else:
        # Standard 24-hour formats
        formats = [
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(ts_series, format=fmt, errors="raise")
                strategy_desc = f"Deterministic parse using 24-hour format '{fmt}'."
                break
            except (ValueError, TypeError):
                continue

    if parsed is None:
        # Fallback to dateutil/iso parser if format was slightly irregular, but verify monotonicity
        try:
            parsed = pd.to_datetime(ts_series, errors="coerce")
            if parsed.isna().mean() > 0.05:
                raise ValueError("More than 5% timestamps failed parsing.")
            strategy_desc = "Parsed using standard pandas ISO/date parser with strict verification."
        except Exception as exc:
            raise ValueError(
                f"Failed to safely and deterministically parse CIC-IDS2017 timestamps! "
                f"Sample values observed: {sample_values[:5]}. Error: {exc}"
            ) from exc

    # Check for potential 12-hour rollover without AM/PM (e.g., dropping from 12:59 to 01:00)
    # Convert to numeric epoch seconds for deterministic sequential processing
    epoch_seconds = parsed.astype("int64") // 10**9
    diffs = epoch_seconds.diff()
    # If a negative drop between 10 to 13 hours occurs (e.g. -43200s +/- 3600s), detect it
    large_drops = (diffs < -36000) & (diffs > -46800)
    if large_drops.any():
        drop_idx = large_drops.idxmax()
        # Add 12 hours (43200 seconds) to all rows following the drop
        epoch_seconds.loc[drop_idx:] += 43200
        strategy_desc += " Detected 12-hour afternoon clock rollover without AM/PM; added +12h offset."

    return epoch_seconds, strategy_desc


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_and_preprocess_dataset(
    data_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    """
    Loads GeneratedLabelledFlows CSV files from data_dir.
    Replaces infs, drops nulls, and parses timestamps deterministically.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Search for CSV files in data_dir (including subdirectories)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir}")

    print(f"[*] Found {len(csv_files)} CSV file(s) in {data_dir}.")
    dfs = []
    col_mapping = None
    ts_strategy = ""

    for file_path in csv_files:
        print(f"    - Reading {os.path.basename(file_path)}...")
        df_chunk = pd.read_csv(file_path, low_memory=False)
        # Strip leading/trailing spaces from column names
        df_chunk.columns = [c.strip() for c in df_chunk.columns]

        if col_mapping is None:
            col_mapping = resolve_column_mapping(list(df_chunk.columns))

        # Parse timestamps
        ts_sec, strategy = parse_cic_timestamps(df_chunk[col_mapping["timestamp"]])
        if not ts_strategy:
            ts_strategy = strategy

        df_chunk["_epoch_sec"] = ts_sec
        df_chunk["_label_clean"] = df_chunk[col_mapping["label"]].astype(str).str.strip()

        # Handle infinite and missing values explicitly in numeric columns
        numeric_needed = [
            col_mapping["syn_flag"],
            col_mapping["fwd_packets"],
            col_mapping["bwd_packets"],
            col_mapping["flow_duration"],
            col_mapping["packet_length"],
        ]
        for num_col in numeric_needed:
            df_chunk[num_col] = pd.to_numeric(
                df_chunk[num_col].replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan),
                errors="coerce",
            )

        # Drop corrupted rows where timestamps or key metrics are null
        df_chunk = df_chunk.dropna(subset=["_epoch_sec", "_label_clean"] + numeric_needed)
        dfs.append(df_chunk)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by="_epoch_sec").reset_index(drop=True)
    print(f"[*] Total valid flows loaded across all files: {len(combined_df):,}")

    return combined_df, col_mapping, ts_strategy


# ==============================================================================
# WINDOW AGGREGATION & FEATURE EXTRACTION
# ==============================================================================
def aggregate_into_windows(
    flows_df: pd.DataFrame,
    col_map: Dict[str, str],
    window_seconds: float,
) -> pd.DataFrame:
    """
    Aggregates sorted flows into discrete non-overlapping time windows of window_seconds.

    Computes canonical features:
    flow_count, syn_rate, total_fwd_packets, total_bwd_packets, mean_flow_duration,
    mean_packet_length, unique_src_count, unique_dst_count, mean_total_packets.
    """
    min_time = float(flows_df["_epoch_sec"].min())
    max_time = float(flows_df["_epoch_sec"].max())

    # Assign each flow to its window index relative to min_time
    flows_df["_window_idx"] = np.floor((flows_df["_epoch_sec"] - min_time) / window_seconds).astype(int)

    grouped = flows_df.groupby("_window_idx")

    flow_count = grouped.size()
    syn_count = grouped[col_map["syn_flag"]].sum()
    syn_rate = syn_count / float(window_seconds)
    total_fwd = grouped[col_map["fwd_packets"]].sum()
    total_bwd = grouped[col_map["bwd_packets"]].sum()
    mean_dur = grouped[col_map["flow_duration"]].mean()
    mean_len = grouped[col_map["packet_length"]].mean()
    unique_src = grouped[col_map["source"]].nunique()
    unique_dst = grouped[col_map["destination"]].nunique()
    mean_tot_pkts = (total_fwd + total_bwd) / flow_count

    # Window attack classification: True if any non-BENIGN flow overlaps this window
    is_attack_flow = flows_df["_label_clean"].str.upper() != "BENIGN"
    attack_flows = flows_df[is_attack_flow]
    attack_window_indices = set(attack_flows["_window_idx"].unique())

    # Determine dominant attack type per attack window
    attack_type_map = {}
    if not attack_flows.empty:
        attack_grouped = attack_flows.groupby(["_window_idx", "_label_clean"]).size().reset_index(name="count")
        dominant = attack_grouped.sort_values(["_window_idx", "count"], ascending=[True, False]).drop_duplicates(subset=["_window_idx"])
        attack_type_map = dict(zip(dominant["_window_idx"], dominant["_label_clean"]))

    unique_windows = np.array(sorted(flows_df["_window_idx"].unique()), dtype=int)
    w_start = min_time + unique_windows * window_seconds
    w_end = w_start + window_seconds

    is_attack = np.array([w in attack_window_indices for w in unique_windows], dtype=bool)
    attack_type = [attack_type_map.get(w, "BENIGN") for w in unique_windows]

    windows_df = pd.DataFrame({
        "window_idx": unique_windows,
        "window_start": w_start,
        "window_end": w_end,
        "flow_count": flow_count.loc[unique_windows].values.astype(float),
        "syn_rate": syn_rate.loc[unique_windows].values.astype(float),
        "total_fwd_packets": total_fwd.loc[unique_windows].values.astype(float),
        "total_bwd_packets": total_bwd.loc[unique_windows].values.astype(float),
        "mean_flow_duration": mean_dur.loc[unique_windows].values.astype(float),
        "mean_packet_length": mean_len.loc[unique_windows].values.astype(float),
        "unique_src_count": unique_src.loc[unique_windows].values.astype(float),
        "unique_dst_count": unique_dst.loc[unique_windows].values.astype(float),
        "mean_total_packets": mean_tot_pkts.loc[unique_windows].values.astype(float),
        "is_attack": is_attack,
        "attack_type": attack_type,
    })

    windows_df = windows_df.sort_values(by="window_start").reset_index(drop=True)
    print(f"[*] Generated {len(windows_df):,} active traffic windows.")
    return windows_df


# ==============================================================================
# EPISODE IDENTIFICATION AND ONSET DETECTION
# ==============================================================================
def identify_attack_episodes(
    windows_df: pd.DataFrame,
    episode_gap_seconds: float,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Identifies contiguous runs of attack windows of the same attack type.
    An episode splits when the gap to the next attack window of that type is >= episode_gap_seconds.
    Onset = start time of the episode.
    """
    windows_df["episode_id"] = ""
    episodes = []
    episode_counter = 0

    attack_types = [t for t in windows_df["attack_type"].unique() if t.upper() != "BENIGN"]

    for a_type in attack_types:
        type_indices = windows_df[windows_df["attack_type"] == a_type].index.tolist()
        if not type_indices:
            continue

        current_ep_windows = [type_indices[0]]

        for i in range(1, len(type_indices)):
            prev_idx = type_indices[i - 1]
            curr_idx = type_indices[i]

            prev_end = windows_df.loc[prev_idx, "window_end"]
            curr_start = windows_df.loc[curr_idx, "window_start"]

            gap = curr_start - prev_end
            if gap >= episode_gap_seconds:
                # Close current episode
                ep_id = f"ep_{episode_counter:03d}_{a_type}"
                windows_df.loc[current_ep_windows, "episode_id"] = ep_id
                episodes.append({
                    "episode_id": ep_id,
                    "attack_type": a_type,
                    "onset": float(windows_df.loc[current_ep_windows[0], "window_start"]),
                    "end_time": float(windows_df.loc[current_ep_windows[-1], "window_end"]),
                    "window_indices": current_ep_windows,
                })
                episode_counter += 1
                current_ep_windows = [curr_idx]
            else:
                current_ep_windows.append(curr_idx)

        # Close final episode for this attack type
        ep_id = f"ep_{episode_counter:03d}_{a_type}"
        windows_df.loc[current_ep_windows, "episode_id"] = ep_id
        episodes.append({
            "episode_id": ep_id,
            "attack_type": a_type,
            "onset": float(windows_df.loc[current_ep_windows[0], "window_start"]),
            "end_time": float(windows_df.loc[current_ep_windows[-1], "window_end"]),
            "window_indices": current_ep_windows,
        })
        episode_counter += 1

    episodes = sorted(episodes, key=lambda x: x["onset"])
    print(f"[*] Identified {len(episodes)} attack episodes across all attack types.")
    return windows_df, episodes


# ==============================================================================
# ONSET FORECAST LABELING
# ==============================================================================
def assign_forecast_labels(
    windows_df: pd.DataFrame,
    episodes: List[Dict],
    label_horizon_seconds: float,
) -> pd.DataFrame:
    """
    Assigns the binary forecast label:
    true_label = 1 if an attack episode begins within label_horizon_seconds AFTER
    the current window (i.e. window_end <= onset <= window_end + label_horizon_seconds).
    true_label = 0 otherwise.
    """
    onsets = np.array([ep["onset"] for ep in episodes]) if episodes else np.array([])

    labels = []
    for _, row in windows_df.iterrows():
        w_end = row["window_end"]
        horizon_end = w_end + label_horizon_seconds

        # Check if any onset falls in [w_end, horizon_end]
        if len(onsets) > 0:
            has_upcoming_onset = np.any((onsets >= w_end) & (onsets <= horizon_end))
        else:
            has_upcoming_onset = False

        labels.append(1 if has_upcoming_onset else 0)

    windows_df["true_label"] = labels
    return windows_df


# ==============================================================================
# LEAKAGE-FREE TIME-ORDERED DEV / EVAL SPLIT
# ==============================================================================
def split_dev_eval(
    windows_df: pd.DataFrame,
    episodes: List[Dict],
    dev_fraction: float,
    label_horizon_seconds: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], List[Dict]]:
    """
    Splits data in strict chronological order (earlier = dev, later = eval).
    Places boundary in benign time between episodes.
    Drops windows within label horizon of the boundary to avoid label leakage.
    Enforces strict leakage check.
    """
    print(f"[*] Computing time-ordered split (dev fraction: {dev_fraction})...")
    n_total = len(windows_df)
    target_split_idx = int(n_total * dev_fraction)

    # Search outward from target_split_idx for a benign window not inside any episode
    split_idx = None
    search_radius = max(50, int(n_total * 0.1))

    for offset in range(search_radius):
        for candidate in [target_split_idx + offset, target_split_idx - offset]:
            if 0 < candidate < n_total - 1:
                row = windows_df.iloc[candidate]
                # Boundary must be benign and not tagged with an episode
                if not row["is_attack"] and row["episode_id"] == "":
                    # Check that no onset is within label_horizon of this boundary
                    boundary_time = row["window_end"]
                    onsets = [ep["onset"] for ep in episodes]
                    near_onset = any(abs(boundary_time - o) < label_horizon_seconds for o in onsets)
                    if not near_onset:
                        split_idx = candidate
                        break
        if split_idx is not None:
            break

    if split_idx is None:
        split_idx = target_split_idx

    split_boundary_time = windows_df.iloc[split_idx]["window_end"]

    # Drop windows whose forecast horizon crosses the boundary
    dev_df = windows_df[windows_df["window_end"] + label_horizon_seconds < split_boundary_time].copy()
    eval_df = windows_df[windows_df["window_start"] >= split_boundary_time].copy()

    dev_df["split"] = "dev"
    eval_df["split"] = "eval"

    # Split episodes into dev and eval
    dev_episodes = [ep for ep in episodes if ep["onset"] < split_boundary_time]
    eval_episodes = [ep for ep in episodes if ep["onset"] >= split_boundary_time]

    # Explicit leakage verification
    max_dev_time = dev_df["window_end"].max() if not dev_df.empty else 0
    min_eval_time = eval_df["window_start"].min() if not eval_df.empty else 0

    if max_dev_time >= min_eval_time:
        raise RuntimeError(
            f"Leakage check failed! Dev max time ({max_dev_time}) >= Eval min time ({min_eval_time})."
        )

    # Verify no episode straddles both dev and eval
    for ep in episodes:
        in_dev = ep["onset"] < split_boundary_time
        ep_end = ep["end_time"]
        if in_dev and ep_end >= split_boundary_time:
            raise RuntimeError(
                f"Leakage check failed! Episode {ep['episode_id']} straddles dev and eval boundary!"
            )

    print(f"[*] Split complete: Dev windows = {len(dev_df):,}, Eval windows = {len(eval_df):,}")
    return dev_df, eval_df, dev_episodes, eval_episodes


# ==============================================================================
# MODEL TRAINING & THRESHOLD SELECTION (DEV ONLY)
# ==============================================================================
def train_and_tune_model(
    dev_df: pd.DataFrame,
    seed: int,
) -> Tuple[Pipeline, float, str, Pipeline, float]:
    """
    Trains a Logistic Regression pipeline and a static Baseline rule ONLY on Dev.
    Tuning uses chronological sub-split inside Dev (no shuffling).
    """
    print("[*] Training and tuning onset forecasting model on Dev set...")
    X_dev = dev_df[CANONICAL_FEATURES].values
    y_dev = dev_df["true_label"].values

    # Chronological internal validation split (first 70% train, last 30% val)
    n_dev = len(dev_df)
    n_train = max(1, int(n_dev * 0.7))

    X_train, y_train = X_dev[:n_train], y_dev[:n_train]
    X_val, y_val = X_dev[n_train:], y_dev[n_train:]

    # Pipeline: StandardScaler + LogisticRegression with balanced class weights
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    # Threshold selection on validation set
    if len(np.unique(y_val)) > 1:
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        best_thresh = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.1, 0.9, 17):
            pred = (val_probs >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    else:
        best_thresh = 0.5

    # Retrain pipeline on full Dev set with selected threshold
    pipeline.fit(X_dev, y_dev)
    model_name = "LogisticRegression-WindowOnsetForecast"

    # Baseline rule: static rule on syn_rate (or flow_count if syn_rate has no variance)
    syn_dev = dev_df["syn_rate"].values
    if np.var(syn_dev) > 1e-4:
        baseline_thresh = float(np.percentile(syn_dev[y_dev == 0], 95)) if (y_dev == 0).any() else float(np.median(syn_dev))
        baseline_feature = "syn_rate"
    else:
        flow_dev = dev_df["flow_count"].values
        baseline_thresh = float(np.percentile(flow_dev[y_dev == 0], 95)) if (y_dev == 0).any() else float(np.median(flow_dev))
        baseline_feature = "flow_count"

    print(f"[*] Tuned model threshold on Dev validation: {best_thresh:.3f}")
    print(f"[*] Tuned baseline ({baseline_feature}) threshold on Dev: {baseline_thresh:.3f}")

    return pipeline, float(best_thresh), model_name, baseline_feature, baseline_thresh


# ==============================================================================
# EVALUATION METRICS (T1, T2, T3)
# ==============================================================================
def evaluate_forecasts(
    eval_df: pd.DataFrame,
    eval_episodes: List[Dict],
    model: Pipeline,
    threshold: float,
    baseline_feature: str,
    baseline_threshold: float,
    min_lead_seconds: float,
    lookback_seconds: float,
    label_horizon_seconds: float,
    window_seconds: float,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Evaluates forecasting quality on Eval windows for Model and Baseline.
    Measures T1 (lead warning), T2 (false alarms per benign hour), and T3 (latency).
    """
    print("[*] Scoring Eval windows and evaluating criteria T1, T2, T3...")
    X_eval = eval_df[CANONICAL_FEATURES].values

    # Measure T3 per-window latency for every window individually
    latencies_ms = []
    scores = []
    alarms = []
    failed_scorings = 0

    for i in range(len(eval_df)):
        x_row = X_eval[i : i + 1]
        try:
            t0 = time.perf_counter()
            prob = model.predict_proba(x_row)[0, 1]
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)
            scores.append(float(prob))
            alarms.append(1 if prob >= threshold else 0)
        except Exception:
            failed_scorings += 1
            latencies_ms.append(0.0)
            scores.append(0.0)
            alarms.append(0)

    eval_df["score"] = scores
    eval_df["alarm"] = alarms
    eval_df["latency_ms"] = latencies_ms

    # Baseline scoring
    baseline_scores = eval_df[baseline_feature].values
    baseline_alarms = (baseline_scores >= baseline_threshold).astype(int)
    eval_df["baseline_alarm"] = baseline_alarms

    eval_start_time = eval_df["window_start"].min() if not eval_df.empty else 0

    # --------------------------------------------------------------------------
    # METRIC T1: Warning Time on Eligible Episodes
    # An episode is eligible if the eval data contains at least min_lead seconds
    # before its onset.
    # An alarm is valid if raised in [onset - lookback, onset - min_lead].
    # Alarm time = window_end.
    # --------------------------------------------------------------------------
    def compute_t1(alarm_col: str) -> Dict:
        eligible_count = 0
        warned_count = 0
        lead_times = []

        for ep in eval_episodes:
            onset = ep["onset"]
            # Eligibility: at least min_lead seconds of eval monitoring before onset
            if (onset - eval_start_time) < min_lead_seconds:
                continue

            eligible_count += 1
            earliest_window = onset - lookback_seconds
            latest_window = onset - min_lead_seconds

            # Filter eval windows within warning interval
            warn_windows = eval_df[
                (eval_df["window_end"] >= earliest_window)
                & (eval_df["window_end"] <= latest_window)
                & (eval_df[alarm_col] == 1)
            ]

            if not warn_windows.empty:
                warned_count += 1
                first_alarm_time = warn_windows["window_end"].min()
                lead_times.append(onset - first_alarm_time)

        pct = (warned_count / eligible_count * 100.0) if eligible_count > 0 else 0.0
        return {
            "eligible_episodes": eligible_count,
            "warned_episodes": warned_count,
            "t1_percentage": round(pct, 2),
            "mean_lead_seconds": round(float(np.mean(lead_times)), 2) if lead_times else 0.0,
        }

    model_t1 = compute_t1("alarm")
    baseline_t1 = compute_t1("baseline_alarm")

    # --------------------------------------------------------------------------
    # METRIC T2: False Alarms per Benign Hour
    # A false alarm is an alarm in a benign window not within lookback or label
    # horizon of an attack onset. Consecutive alarms merged into one event.
    # --------------------------------------------------------------------------
    def compute_t2(alarm_col: str) -> Dict:
        benign_mask = ~eval_df["is_attack"]
        benign_df = eval_df[benign_mask].copy()

        total_benign_seconds = len(benign_df) * window_seconds
        benign_hours = total_benign_seconds / 3600.0

        if benign_df.empty or benign_hours == 0:
            return {"benign_hours": 0.0, "merged_false_alarms": 0, "rate_per_hour": 0.0}

        # Check which benign windows fall near an onset
        eval_onsets = [ep["onset"] for ep in eval_episodes]
        is_near_onset = []
        for w_end in benign_df["window_end"]:
            near = any((onset - lookback_seconds <= w_end <= onset + label_horizon_seconds) for onset in eval_onsets)
            is_near_onset.append(near)

        benign_df["near_onset"] = is_near_onset
        # True false alarm: alarm == 1 and not near any onset
        benign_df["is_false_alarm"] = (benign_df[alarm_col] == 1) & (~benign_df["near_onset"])

        # Merge consecutive false alarm windows into single alarm events
        merged_events = 0
        in_event = False
        prev_idx = -999

        for _, row in benign_df.iterrows():
            w_idx = row["window_idx"]
            if row["is_false_alarm"]:
                if not in_event or (w_idx != prev_idx + 1):
                    merged_events += 1
                    in_event = True
                prev_idx = w_idx
            else:
                in_event = False

        rate = (merged_events / benign_hours) if benign_hours > 0 else 0.0
        return {
            "benign_hours": round(float(benign_hours), 3),
            "merged_false_alarms": int(merged_events),
            "rate_per_hour": round(float(rate), 3),
        }

    model_t2 = compute_t2("alarm")
    baseline_t2 = compute_t2("baseline_alarm")

    # --------------------------------------------------------------------------
    # METRIC T3: p95 Scoring Latency
    # Measured with time.perf_counter around the scoring call for every eval window
    # --------------------------------------------------------------------------
    p50_lat = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_lat = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    model_t3 = {
        "p50_ms": round(p50_lat, 3),
        "p95_ms": round(p95_lat, 3),
        "p99_ms": round(p99_lat, 3),
        "failed_scorings": failed_scorings,
    }

    metrics = {
        "model": {"T1": model_t1, "T2": model_t2, "T3": model_t3},
        "baseline": {"T1": baseline_t1, "T2": baseline_t2},
    }

    return eval_df, metrics, model_t3


# ==============================================================================
# MAIN HARNESS EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="POC Harness: Network Attack Onset Forecasting (CIC-IDS2017 Replay)"
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier (e.g. R01)")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window duration in seconds")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CIC-IDS2017 CSV files")
    parser.add_argument("--out-dir", type=str, default="./", help="Output directory for logs and summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dev-fraction", type=float, default=0.5, help="Fraction of timeline for Dev split")
    parser.add_argument("--episode-gap-seconds", type=float, default=60.0, help="Gap to split attack episodes")
    parser.add_argument(
        "--label-horizon-seconds",
        type=float,
        default=30.0,
        help="Label horizon: attack episode begins within this many seconds",
    )
    parser.add_argument(
        "--min-lead-seconds",
        type=float,
        default=30.0,
        help="Required minimum warning lead time for T1 eligibility",
    )
    parser.add_argument(
        "--lookback-seconds",
        type=float,
        default=300.0,
        help="Lookback period before onset counting toward T1",
    )

    args = parser.parse_args()

    # Step 1: Load flows and resolve column headers
    flows_df, col_mapping, ts_strategy = load_and_preprocess_dataset(args.data_dir)

    # Step 2: Aggregate into discrete windows
    windows_df = aggregate_into_windows(flows_df, col_mapping, args.window_seconds)

    # Step 3: Identify episodes and onsets
    windows_df, episodes = identify_attack_episodes(windows_df, args.episode_gap_seconds)

    # Step 4: Assign onset warning labels
    windows_df = assign_forecast_labels(windows_df, episodes, args.label_horizon_seconds)

    # Step 5: Time-ordered split with leakage check
    dev_df, eval_df, dev_episodes, eval_episodes = split_dev_eval(
        windows_df, episodes, args.dev_fraction, args.label_horizon_seconds
    )

    # Step 6: Print REAL counts in Dev and Eval (no fabricated values)
    dev_benign_hours = (len(dev_df[~dev_df["is_attack"]]) * args.window_seconds) / 3600.0
    eval_benign_hours = (len(eval_df[~eval_df["is_attack"]]) * args.window_seconds) / 3600.0

    print("\n" + "=" * 70)
    print("REAL COUNTS OBSERVED IN SPLITS:")
    print(f"Dev Split : {len(dev_episodes)} episodes, {dev_benign_hours:.2f} benign hours")
    dev_ep_types = pd.Series([ep["attack_type"] for ep in dev_episodes]).value_counts().to_dict()
    for atype, cnt in dev_ep_types.items():
        print(f"  - {atype}: {cnt} episode(s)")

    print(f"Eval Split: {len(eval_episodes)} episodes, {eval_benign_hours:.2f} benign hours")
    eval_ep_types = pd.Series([ep["attack_type"] for ep in eval_episodes]).value_counts().to_dict()
    for atype, cnt in eval_ep_types.items():
        print(f"  - {atype}: {cnt} episode(s)")

    # Warning if attack types appear ONLY in eval
    eval_only_types = set(eval_ep_types.keys()) - set(dev_ep_types.keys())
    if eval_only_types:
        print(f"\n[!] WARNING: The following attack types appear ONLY in Eval: {list(eval_only_types)}")
        print("    The model was not exposed to these specific attacks during Dev training!")
    print("=" * 70 + "\n")

    # Step 7: Train and tune on Dev ONLY
    model_pipeline, chosen_thresh, model_name, base_feature, base_thresh = train_and_tune_model(
        dev_df, args.seed
    )

    # Save model bundles
    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": model_pipeline,
        "feature_names": CANONICAL_FEATURES,
        "threshold": chosen_thresh,
        "model_name": model_name,
        "seed": args.seed,
        "window_seconds": args.window_seconds,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    per_run_model_path = os.path.join("model", f"{args.run_id}_model.joblib")
    shared_model_path = os.path.join("model", "model.joblib")
    joblib.dump(bundle, per_run_model_path)
    shutil.copyfile(per_run_model_path, shared_model_path)
    print(f"[*] Model bundle saved to {per_run_model_path} and copied to {shared_model_path}")

    # Step 8: Evaluate on Eval windows
    scored_eval_df, metrics, model_t3 = evaluate_forecasts(
        eval_df=eval_df,
        eval_episodes=eval_episodes,
        model=model_pipeline,
        threshold=chosen_thresh,
        baseline_feature=base_feature,
        baseline_threshold=base_thresh,
        min_lead_seconds=args.min_lead_seconds,
        lookback_seconds=args.lookback_seconds,
        label_horizon_seconds=args.label_horizon_seconds,
        window_seconds=args.window_seconds,
    )

    # Step 9: Criterion Check (Never adjust thresholds after seeing eval)
    t1_val = metrics["model"]["T1"]["t1_percentage"]
    t2_val = metrics["model"]["T2"]["rate_per_hour"]
    t3_p95 = model_t3["p95_ms"]
    t3_failed = model_t3["failed_scorings"]

    print("\n" + "=" * 70)
    print("CRITERION EVALUATION RESULTS:")
    t1_pass = t1_val >= 80.0
    t2_pass = t2_val <= 5.0
    t3_pass = (t3_p95 <= 200.0) and (t3_failed == 0)

    print(
        f"Criterion T1 (Warning Rate >= 80%): {t1_val}% -> "
        f"{'PASSED' if t1_pass else 'FAILED (T1 < 80%)'}"
    )
    print(
        f"Criterion T2 (False Alarms <= 5/hr): {t2_val}/hr -> "
        f"{'PASSED' if t2_pass else 'FAILED (T2 > 5/hr)'}"
    )
    print(
        f"Criterion T3 (p95 Latency <= 200ms, 0 failures): p95={t3_p95}ms, failures={t3_failed} -> "
        f"{'PASSED' if t3_pass else 'FAILED (T3 > 200ms or failures > 0)'}"
    )
    print("=" * 70 + "\n")

    # Step 10: Write Outputs
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. {run-id}_log.csv
    log_file = os.path.join(args.out_dir, f"{args.run_id}_log.csv")
    log_df = pd.DataFrame({
        "timestamp": scored_eval_df["window_start"],
        "split": scored_eval_df["split"],
        "true_label": scored_eval_df["true_label"],
        "episode_id": scored_eval_df["episode_id"],
        "score": scored_eval_df["score"].round(4),
        "alarm": scored_eval_df["alarm"],
        "latency_ms": scored_eval_df["latency_ms"].round(3),
        "run_id": args.run_id,
        "window_seconds": args.window_seconds,
    })
    log_df.to_csv(log_file, index=False)
    print(f"[*] Saved evaluation log to {log_file}")

    # 2. {run-id}_eval_features.csv (input for latency_test.py)
    features_file = os.path.join(args.out_dir, f"{args.run_id}_eval_features.csv")
    eval_features_df = scored_eval_df[CANONICAL_FEATURES].copy()
    eval_features_df.to_csv(features_file, index=False)
    print(f"[*] Saved canonical eval features to {features_file}")

    # 3. {run-id}_summary.json
    summary_file = os.path.join(args.out_dir, f"{args.run_id}_summary.json")
    summary_data = {
        "run_id": args.run_id,
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": vars(args),
        "timestamp_handling": ts_strategy,
        "metrics": metrics,
        "counts": {
            "total_windows": len(windows_df),
            "dev_windows": len(dev_df),
            "eval_windows": len(eval_df),
            "dev_episodes": len(dev_episodes),
            "eval_episodes": len(eval_episodes),
            "dev_episodes_by_type": dev_ep_types,
            "eval_episodes_by_type": eval_ep_types,
            "dev_benign_hours": round(dev_benign_hours, 3),
            "eval_benign_hours": round(eval_benign_hours, 3),
        },
        "criteria_passed": {
            "T1": t1_pass,
            "T2": t2_pass,
            "T3": t3_pass,
        },
        "definitions": {
            "T1": "Eligible episodes with first alarm in [onset - lookback, onset - min_lead] / eligible episodes. "
                  "Lead = onset - alarm window end time. Eligible = eval data contains >= min_lead_seconds before onset.",
            "T2": "False alarm events per benign hour. False alarm = alarm in benign window not within lookback or "
                  "label horizon of any attack onset. Consecutive false alarms merged into single event.",
            "T3": "p95 per-window model scoring latency in ms measured via time.perf_counter around scoring call.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[*] Saved run summary to {summary_file}")


if __name__ == "__main__":
    main()

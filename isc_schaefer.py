import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict
from nilearn import signal  # Added to handle high-pass filtering

BIDS_ROOT  = "/lustre/disk/home/shared/cusacklab/foundcog/bids"
DERIV_ROOT = os.path.join(BIDS_ROOT, "derivatives", "faizan_analysis")

BOLD_TEMPLATE = os.path.join(
    BIDS_ROOT,
    "{subject}", "ses-{session}", "func",
    "{subject}_ses-{session}_task-videos_dir-AP_run-{run:03d}_bold.nii.gz"
)

MASK_TEMPLATE = os.path.join(
    DERIV_ROOT, "schaefer_backnorm",
    "{subject}", "ses-{session}", "func", "mask_4d",
    "{subject}_ses-{session}_task-videos_run-{run:03d}_space-native_desc-mask4d.nii.gz"
)

LABELS_PATH = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "templates/rois/Schaefer2018_400Parcels_7Networks_order.lut"
)

DEFAULT_CSV = "per_order_alignment/segments_mapping_each_sub_usable.csv"
DEFAULT_OUT = os.path.join(DERIV_ROOT, "isc_schaefer", "across_brain_networks_analysis_and_high_pass_filtering")

# Canonical Yeo 7-network order (used to sort the 7×7 matrix rows/columns)
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-ROI and cross-network ISC using Schaefer atlas and 4D masks."
    )
    parser.add_argument("--csv",           default=DEFAULT_CSV)
    parser.add_argument("--out_dir",       default=DEFAULT_OUT)
    parser.add_argument("--bold_template", default=BOLD_TEMPLATE)
    parser.add_argument("--mask_template", default=MASK_TEMPLATE)
    parser.add_argument("--labels",        default=LABELS_PATH)
    parser.add_argument("--n_rois",        type=int, default=400)
    parser.add_argument("--nan_policy",    default="drop",
        choices=["drop", "interpolate"])
    return parser.parse_args()


# ── Path helpers ──────────────────────────────────────────────────────────────
def build_path(template, subject, session, run):
    return template.format(
        subject=str(subject),
        session=int(session),
        run=int(run)
    )

def preflight_check(df, bold_template, mask_template):
    print("\n" + "=" * 70)
    print("  PREFLIGHT — checking all (subject, session, run) file pairs")
    print("=" * 70)

    combos = (
        df[["subject", "session", "run"]]
        .drop_duplicates()
        .sort_values(["subject", "session", "run"])
    )

    found   = set()
    missing = []

    for _, row in combos.iterrows():
        sub = row["subject"]
        ses = int(row["session"])
        run = int(row["run"])
        bold_path = build_path(bold_template, sub, ses, run)
        mask_path = build_path(mask_template, sub, ses, run)
        bold_ok = os.path.exists(bold_path)
        mask_ok = os.path.exists(mask_path)
        if bold_ok and mask_ok:
            found.add((sub, ses, run))
        else:
            if not bold_ok:
                missing.append((bold_path, "BOLD"))
            if not mask_ok:
                missing.append((mask_path, "MASK"))

    n_total = len(combos)
    n_found = len(found)
    print(f"  Unique (subject, session, run) pairs in CSV : {n_total}")
    print(f"  Pairs where BOTH files exist                : {n_found}")
    print(f"  Pairs with at least one missing file        : {n_total - n_found}")

    if missing:
        print(f"\n  First 15 missing paths:")
        for path, kind in missing[:15]:
            print(f"    [{kind}]  {path}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more.")

    print("=" * 70)

    if n_found == 0:
        raise FileNotFoundError(
            "No valid file pairs found. Check --bold_template and --mask_template."
        )

    return found, missing


# ── Atlas helpers ─────────────────────────────────────────────────────────────
def load_lut(lut_path):
    """Parse Schaefer .lut → dict {roi_id (int): roi_name (str)}."""
    labels = {}
    if not os.path.exists(lut_path):
        print(f"[WARN] LUT not found: {lut_path}. ROIs labelled numerically.")
        return labels
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    labels[int(parts[0])] = parts[4]
                except ValueError:
                    continue
    print(f"[INFO] Loaded {len(labels)} ROI labels.")
    return labels


def get_network_from_label(roi_name):
    """
    Extract the network name from a Schaefer ROI label.

    Label format:  7Networks_{Hemi}_{Network}_{Index}
    Examples:
        '7Networks_LH_Vis_1'        → 'Vis'
        '7Networks_RH_Default_1'    → 'Default'
        '7Networks_LH_SalVentAttn_2'→ 'SalVentAttn'

    Returns 'Unknown' if the label cannot be parsed.
    """
    parts = roi_name.split("_")
    # parts[0]='7Networks', parts[1]=hemi, parts[2]=network, parts[3]=index
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"

def build_network_roi_map(roi_names):
    """
    Group ROI indices (0-based) by network name.

    Parameters
    ----------
    roi_names : list of str  length n_rois

    Returns
    -------
    ordered dict  {network_name: np.ndarray of 0-based ROI indices}
    sorted by NETWORK_ORDER; any unrecognised networks appended at the end.
    """
    net_map = defaultdict(list)
    for idx, name in enumerate(roi_names):
        net = get_network_from_label(name)
        net_map[net].append(idx)

    ordered = {}
    for net in NETWORK_ORDER:
        if net in net_map:
            ordered[net] = np.array(net_map[net])
    for net in net_map:        # catch any unlabelled / unexpected networks
        if net not in ordered:
            ordered[net] = np.array(net_map[net])

    return ordered

def apply_high_pass_filter(timecourses, tr, cutoff=0.01):
    """
    Applies high-pass filter to the time courses of each ROI.

    Parameters
    ----------
    timecourses : np.ndarray (n_rois, T)
        The extracted time courses for each ROI (before filtering).
    tr : float
        The repetition time (TR) in seconds.
    cutoff : float
        The cutoff frequency for high-pass filtering (in Hz).

    Returns
    -------
    filtered_timecourses : np.ndarray (n_rois, T)
        The time courses after applying the high-pass filter.
    """
    # Apply high-pass filter using Nilearn's signal.clean
    filtered_timecourses = signal.clean(timecourses, 
                                        t_r=tr, 
                                        high_pass=cutoff, 
                                        detrend=True,  # Optionally detrend (removes linear trends)
                                        standardize=True)  # Optionally standardize (z-score)
    return filtered_timecourses


def extract_roi_timecourse(bold_data, mask_data, start_idx, end_idx, n_rois=400, tr=0.61, high_pass_cutoff=0.01):
    """
    Extract mean BOLD timecourse per ROI for one segment, and apply high-pass filtering.

    Parameters
    ----------
    bold_data : np.ndarray
        The BOLD data (4D: X, Y, Z, T).
    mask_data : np.ndarray
        The mask data (4D: X, Y, Z, T).
    start_idx : int
        The start index for the segment.
    end_idx : int
        The end index for the segment.
    n_rois : int
        The number of ROIs (default: 400).
    tr : float
        The repetition time (TR) in seconds.
    high_pass_cutoff : float
        The cutoff frequency for the high-pass filter (default: 0.01 Hz).

    Returns
    -------
    signal : np.ndarray
        The extracted and filtered time course for each ROI.
    """
    # Extract the BOLD and mask segments
    n_vols  = bold_data.shape[-1]
    end_idx = min(end_idx, n_vols)
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) >= end_idx ({end_idx}).")

    bold_seg  = bold_data[:, :, :, start_idx:end_idx]
    mask_seg  = mask_data[:, :, :, start_idx:end_idx]
    T         = bold_seg.shape[-1]

    bold_flat = bold_seg.reshape(-1, T).astype(np.float64)
    mask_flat = mask_seg.reshape(-1, T).astype(np.int32)

    signal = np.full((n_rois, T), np.nan, dtype=np.float32)

    # Extract the ROI-wise signal (mean per ROI per timepoint)
    for t in range(T):
        m      = mask_flat[:, t]
        b      = bold_flat[:, t]
        counts = np.bincount(m, minlength=n_rois + 1)
        sums   = np.bincount(m, weights=b, minlength=n_rois + 1)
        c      = counts[1:n_rois + 1]
        s      = sums  [1:n_rois + 1]
        valid  = c > 0
        signal[valid, t] = (s[valid] / c[valid]).astype(np.float32)

    # Apply high-pass filtering to the extracted signal (ROI time courses)
    filtered_signal = apply_high_pass_filter(signal, tr, cutoff=high_pass_cutoff)
    return filtered_signal   # (n_rois, T_seg)


# # ── ROI-level timecourse extraction ──────────────────────────────────────────
# def extract_roi_timecourse(bold_data, mask_data, start_idx, end_idx, n_rois=400):
#     """
#     Extract mean BOLD timecourse per ROI for one segment.

#     The mask is 4-D (X, Y, Z, T): each timepoint independently maps voxels
#     to ROI labels, so motion censoring is respected per TR.

#     Returns  signal : (n_rois, T_seg)  — NaN where ROI has no valid voxels.
#     """
#     n_vols  = bold_data.shape[-1]
#     end_idx = min(end_idx, n_vols)
#     if start_idx >= end_idx:
#         raise ValueError(f"start_idx ({start_idx}) >= end_idx ({end_idx}).")

#     bold_seg  = bold_data[:, :, :, start_idx:end_idx]
#     mask_seg  = mask_data[:, :, :, start_idx:end_idx]
#     T         = bold_seg.shape[-1]

#     bold_flat = bold_seg.reshape(-1, T).astype(np.float64)
#     mask_flat = mask_seg.reshape(-1, T).astype(np.int32)

#     signal = np.full((n_rois, T), np.nan, dtype=np.float32)

#     for t in range(T):
#         m      = mask_flat[:, t]
#         b      = bold_flat[:, t]
#         counts = np.bincount(m, minlength=n_rois + 1)
#         sums   = np.bincount(m, weights=b, minlength=n_rois + 1)
#         c      = counts[1:n_rois + 1]
#         s      = sums  [1:n_rois + 1]
#         valid  = c > 0
#         signal[valid, t] = (s[valid] / c[valid]).astype(np.float32)

#     return signal   # (n_rois, T_seg)


def mean_centre(signal):
    """Subtract temporal mean per ROI row (NaN-safe)."""
    return signal - np.nanmean(signal, axis=1, keepdims=True)


def interpolate_nans(signal):
    """Linear interpolation of NaN timepoints per ROI."""
    out = signal.copy()
    for r in range(signal.shape[0]):
        ts   = signal[r]
        nans = np.isnan(ts)
        if nans.all() or not nans.any():
            continue
        idx    = np.arange(len(ts))
        out[r] = np.interp(idx, idx[~nans], ts[~nans])
    return out


def average_segments(segments):
    """nanmean a list of (n_rois, T) arrays along the list axis."""
    if not segments:
        return None
    min_T   = min(s.shape[1] for s in segments)
    stacked = np.stack([s[:, :min_T] for s in segments], axis=0)
    return np.nanmean(stacked, axis=0)   # (n_rois, T)


# ── Network-level timecourse ──────────────────────────────────────────────────
def roi_to_network_timecourses(roi_tc, network_roi_map):
    """
    Average ROI timecourses within each network.

    Parameters
    ----------
    roi_tc          : np.ndarray (n_rois, T)
    network_roi_map : dict {network_name: 0-based ROI indices}

    Returns
    -------
    net_tc_array : np.ndarray (n_networks, T)
        Row order follows network_roi_map key order.
    networks     : list of str
    """
    networks = list(network_roi_map.keys())
    net_tc_array = np.stack(
        [np.nanmean(roi_tc[indices, :], axis=0) for indices in network_roi_map.values()],
        axis=0
    )   # (n_networks, T)
    return net_tc_array, networks


# ── ISC functions ─────────────────────────────────────────────────────────────
def loo_isc_single_roi(tc_matrix):
    """
    Standard LOO-ISC for one ROI (or one network timecourse).

    tc_matrix : (n_subjects, T)
    Returns   : (n_subjects,) float32
    """
    n_subs, _ = tc_matrix.shape
    isc_vals  = np.full(n_subs, np.nan, dtype=np.float32)

    valid_mask = np.all(~np.isnan(tc_matrix), axis=0)
    if valid_mask.sum() < 3:
        return isc_vals

    tc_valid = tc_matrix[:, valid_mask].astype(np.float64)
    total    = tc_valid.sum(axis=0)

    for s in range(n_subs):
        others = (total - tc_valid[s]) / (n_subs - 1)
        if np.std(tc_valid[s]) < 1e-10 or np.std(others) < 1e-10:
            continue
        r, _        = pearsonr(tc_valid[s], others)
        isc_vals[s] = r

    return isc_vals


def loo_cross_network_isc(tc_a_stack, tc_b_stack):
    """
    Leave-one-out cross-network ISC.

    For each subject i:
        others_mean_B = mean( tc_b_stack[j]  for j != i )
        isc[i]        = pearsonr( tc_a_stack[i],  others_mean_B )

    When tc_a_stack == tc_b_stack this reduces to the standard LOO-ISC.

    Parameters
    ----------
    tc_a_stack : (n_subs, T)  — network A timecourse per subject
    tc_b_stack : (n_subs, T)  — network B timecourse per subject

    Returns
    -------
    isc_vals : (n_subs,) float32
    """
    n_subs = tc_a_stack.shape[0]
    isc_vals = np.full(n_subs, np.nan, dtype=np.float32)

    # Timepoints must be valid for ALL subjects in BOTH networks
    valid_mask = (
        np.all(~np.isnan(tc_a_stack), axis=0) &
        np.all(~np.isnan(tc_b_stack), axis=0)
    )
    if valid_mask.sum() < 3:
        return isc_vals

    tc_a_valid = tc_a_stack[:, valid_mask].astype(np.float64)
    tc_b_valid = tc_b_stack[:, valid_mask].astype(np.float64)
    total_b    = tc_b_valid.sum(axis=0)

    for s in range(n_subs):
        others_b = (total_b - tc_b_valid[s]) / (n_subs - 1)
        a_s      = tc_a_valid[s]
        if np.std(a_s) < 1e-10 or np.std(others_b) < 1e-10:
            continue
        r, _        = pearsonr(a_s, others_b)
        isc_vals[s] = r

    return isc_vals


# ── Printing helpers ──────────────────────────────────────────────────────────
def print_coverage_table(subject_order_tc, subjects, order_labels):
    print("\n" + "=" * 70)
    print("  SUBJECT × ORDER COVERAGE   ✓(T) = timecourse extracted")
    print("=" * 70)
    header = f"  {'Subject':<20}" + "".join(f" {o:<9}" for o in order_labels)
    print(header)
    print("  " + "-" * (20 + 10 * len(order_labels)))
    for sub in subjects:
        row = f"  {sub:<20}"
        for o in order_labels:
            if o in subject_order_tc[sub]:
                T = subject_order_tc[sub][o].shape[1]
                row += f" ✓({T:<5})"
            else:
                row += f" {'–':<9}"
        print(row)
    print("=" * 70)


def print_isc_summary(isc_results, order_labels):
    print("\n" + "=" * 65)
    print("  PER-ROI ISC SUMMARY")
    print("=" * 65)
    print(f"  {'Order':<8} {'N Subs':<8} {'Mean ISC':<12} {'Max ROI':<12} {'Min ROI'}")
    print("  " + "-" * 55)
    for order in order_labels:
        if order not in isc_results:
            print(f"  {order:<8} skipped (< 2 subjects)")
            continue
        mat      = isc_results[order]
        roi_mean = np.nanmean(mat, axis=0)
        print(f"  {order:<8} {mat.shape[0]:<8} "
              f"{np.nanmean(roi_mean):<12.4f} "
              f"{np.nanmax(roi_mean):<12.4f} "
              f"{np.nanmin(roi_mean):.4f}")
    print("=" * 65)


def print_cross_isc_matrix(mean_mat, networks, order):
    """Pretty-print the 7×7 cross-network ISC matrix to the terminal."""
    n = len(networks)
    col_w = 10
    print(f"\n  Cross-network ISC — order {order}  (mean over subjects)")
    print("  " + " " * 12 + "".join(f"{net:>{col_w}}" for net in networks))
    print("  " + "-" * (12 + col_w * n))
    for i, net_a in enumerate(networks):
        row_vals = "".join(f"{mean_mat[i, j]:>{col_w}.4f}" for j in range(n))
        print(f"  {net_a:<12}{row_vals}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  ISC — Schaefer 400-ROI + 7-Network cross-ISC, LOO method")
    print("=" * 65)
    for k, v in vars(args).items():
        print(f"  {k:<18}: {v}")
    print("=" * 65)

    # ── 1. Load & filter CSV ──────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df = df[~df["skip"].astype(bool) & ~df["short_segment"].astype(bool)].copy()

    df["subject"] = df["subject"].astype(str).apply(
        lambda s: s if s.startswith("sub-") else f"sub-{s}"
    )
    df["session"]     = df["session"].astype(int)
    df["run"]         = df["run"].astype(int)
    df["segment_num"] = df["segment_num"].astype(int)
    df["order_label"] = df["order_label"].astype(str).str.strip()

    subjects     = sorted(df["subject"].unique())
    order_labels = sorted(df["order_label"].unique())
    n_rois       = args.n_rois

    print(f"\n[INFO] Valid CSV rows : {len(df)}")
    print(f"[INFO] Subjects       : {len(subjects)}")
    print(f"[INFO] Orders         : {order_labels}")
    print(f"[INFO] ROIs           : {n_rois}")

    # ── 2. Preflight ──────────────────────────────────────────────────────────
    valid_pairs, _ = preflight_check(df, args.bold_template, args.mask_template)

    # ── 3. Atlas labels & network map ─────────────────────────────────────────
    labels    = load_lut(args.labels)
    roi_names = [labels.get(r, f"ROI_{r}") for r in range(1, n_rois + 1)]

    network_roi_map = build_network_roi_map(roi_names)
    networks        = list(network_roi_map.keys())
    n_networks      = len(networks)

    print(f"\n[INFO] Networks ({n_networks}): {networks}")
    print(f"[INFO] ROIs per network:")
    for net, idx in network_roi_map.items():
        print(f"         {net:<14}: {len(idx)} ROIs")

    # ── 4. Extract & average ROI timecourses per subject ──────────────────────
    #
    # subject_order_tc[subject][order] = (n_rois, T_total)
    #
    # After extraction we also compute network-averaged timecourses:
    # subject_order_net_tc[subject][order] = (n_networks, T_total)

    subject_order_tc     = defaultdict(dict)
    subject_order_net_tc = defaultdict(dict)

    for subject in tqdm(subjects, desc="Subjects", ncols=80):

        file_cache = {}
        sub_df     = df[df["subject"] == subject]

        for order in order_labels:
            ord_df = sub_df[sub_df["order_label"] == order]
            if ord_df.empty:
                continue

            seg_timecourses = defaultdict(list)

            for _, row in ord_df.iterrows():
                sub = row["subject"]
                ses = int(row["session"])
                run = int(row["run"])

                if (sub, ses, run) not in valid_pairs:
                    continue

                bold_path = build_path(args.bold_template, sub, ses, run)
                mask_path = build_path(args.mask_template, sub, ses, run)
                cache_key = (bold_path, mask_path)

                if cache_key not in file_cache:
                    tqdm.write(f"  [LOAD] {sub}  ses-{ses}  run-{run:03d}")
                    file_cache[cache_key] = (
                        nib.load(bold_path).get_fdata().astype(np.float32),
                        nib.load(mask_path).get_fdata().astype(np.int16)
                    )

                bold_data, mask_data = file_cache[cache_key]

                try:
                    sig = extract_roi_timecourse(
                        bold_data, mask_data,
                        int(row["scan_start_idx"]),
                        int(row["scan_end_idx"]),
                        n_rois=n_rois
                    )
                except ValueError as e:
                    tqdm.write(f"  [WARN] {sub} ses-{ses} run-{run:03d} "
                               f"seg-{row['segment_num']}: {e}")
                    continue

                if args.nan_policy == "interpolate":
                    sig = interpolate_nans(sig)

                sig = mean_centre(sig)
                seg_timecourses[int(row["segment_num"])].append(sig)

            if seg_timecourses:
                averaged = {k: average_segments(v)
                            for k, v in seg_timecourses.items()}
                full_roi_tc = np.concatenate(
                    [averaged[k] for k in sorted(averaged)], axis=1
                )   # (n_rois, T_total)
                subject_order_tc[subject][order] = full_roi_tc

                # ── Network-level average ─────────────────────────────────
                # Average across ROIs within each of the 7 networks.
                # NaN-safe so motion-censored ROIs don't inflate the average.
                net_tc, _ = roi_to_network_timecourses(full_roi_tc, network_roi_map)
                subject_order_net_tc[subject][order] = net_tc  # (n_networks, T_total)

        del file_cache

    print_coverage_table(subject_order_tc, subjects, order_labels)

    # ── 5a. LOO-ISC per ROI ───────────────────────────────────────────────────
    isc_results  = {}
    isc_subjects = {}

    for order in order_labels:
        subs_with_order = [s for s in subjects if order in subject_order_tc[s]]

        if len(subs_with_order) < 2:
            print(f"\n[WARN] Order {order}: {len(subs_with_order)} subject(s) — skipping.")
            continue

        lengths   = {s: subject_order_tc[s][order].shape[1] for s in subs_with_order}
        min_T     = min(lengths.values())

        tc_stack = np.stack(
            [subject_order_tc[s][order][:, :min_T] for s in subs_with_order], axis=0
        )   # (n_subs, n_rois, T)

        n_subs_ord = len(subs_with_order)
        isc_mat    = np.full((n_subs_ord, n_rois), np.nan, dtype=np.float32)

        for r in tqdm(range(n_rois),
                      desc=f"  LOO-ISC order-{order} ({n_subs_ord} subs, T={min_T})",
                      leave=False, ncols=80):
            isc_mat[:, r] = loo_isc_single_roi(tc_stack[:, r, :])

        isc_results[order]  = isc_mat
        isc_subjects[order] = subs_with_order
        print(f"[INFO] Per-ROI ISC order-{order}: shape {isc_mat.shape}")

    # ── 5b. Cross-network LOO-ISC  (7 × 7 matrix) ────────────────────────────
    #
    # For every ordered pair of networks (A, B) and every subject i:
    #
    #   ISC_AxB(i) = pearsonr( net_A_timecourse(i),
    #                          mean_j≠i( net_B_timecourse(j) ) )
    #
    # Then average over subjects:
    #   mean_ISC_AxB = mean_i( ISC_AxB(i) )
    #
    # The diagonal (A == B) is the standard within-network LOO-ISC.
    # Off-diagonal elements measure how well subject i's network A
    # tracks the group's network B.
    #
    # Output shapes:
    #   cross_isc_all[order]  : (n_networks, n_networks, n_subs)
    #   cross_isc_mean[order] : (n_networks, n_networks)

    cross_isc_all  = {}
    cross_isc_mean = {}

    for order in order_labels:
        subs_with_order = [s for s in subjects if order in subject_order_net_tc[s]]

        if len(subs_with_order) < 2:
            print(f"\n[WARN] Order {order}: too few subjects for cross-network ISC.")
            continue

        lengths = {s: subject_order_net_tc[s][order].shape[1]
                   for s in subs_with_order}
        min_T   = min(lengths.values())

        # Stack → (n_subs, n_networks, T)
        net_stack = np.stack(
            [subject_order_net_tc[s][order][:, :min_T] for s in subs_with_order],
            axis=0
        )

        n_subs_ord = len(subs_with_order)
        isc_cube   = np.full((n_networks, n_networks, n_subs_ord), np.nan, dtype=np.float32)
        mean_mat   = np.full((n_networks, n_networks), np.nan, dtype=np.float32)

        for i in tqdm(range(n_networks),
                      desc=f"  Cross-net ISC order-{order} ({n_subs_ord} subs)",
                      leave=False, ncols=80):
            for j in range(n_networks):
                tc_a = net_stack[:, i, :]   # (n_subs, T) — network A per subject
                tc_b = net_stack[:, j, :]   # (n_subs, T) — network B per subject
                vals = loo_cross_network_isc(tc_a, tc_b)
                isc_cube[i, j, :] = vals
                mean_mat[i, j]    = np.nanmean(vals)

        cross_isc_all[order]  = isc_cube
        cross_isc_mean[order] = mean_mat
        print(f"[INFO] Cross-network ISC order-{order}: matrix shape {mean_mat.shape}")
        print_cross_isc_matrix(mean_mat, networks, order)

    # ── 6. Save all outputs ───────────────────────────────────────────────────

    # 6a. Per-ROI ISC
    for order, isc_mat in isc_results.items():
        subs     = isc_subjects[order]
        mean_isc = np.nanmean(isc_mat, axis=0)

        np.save(os.path.join(args.out_dir, f"isc_order-{order}.npy"), isc_mat)
        np.save(os.path.join(args.out_dir, f"isc_mean_order-{order}.npy"), mean_isc)

        pd.Series(subs, name="subject").to_csv(
            os.path.join(args.out_dir, f"subjects_order-{order}.csv"), index=False)

        df_out = pd.DataFrame({
            "roi_id"  : range(1, n_rois + 1),
            "roi_name": roi_names,
            "mean_isc": mean_isc,
        })
        for i, sub in enumerate(subs):
            df_out[sub] = isc_mat[i]
        df_out.to_csv(
            os.path.join(args.out_dir, f"isc_order-{order}.csv"), index=False)

        print(f"[INFO] Saved per-ROI ISC order-{order}")

    # 6b. Cross-network ISC
    for order, mean_mat in cross_isc_mean.items():
        # Mean 7×7 matrix as CSV (networks as index and columns — easy to read in pandas)
        df_cross = pd.DataFrame(mean_mat, index=networks, columns=networks)
        df_cross.to_csv(
            os.path.join(args.out_dir, f"cross_isc_order-{order}.csv"))

        # Full (n_networks, n_networks, n_subs) cube
        np.save(
            os.path.join(args.out_dir, f"cross_isc_order-{order}.npy"),
            cross_isc_all[order])

        # Per-network (diagonal) as a tidy CSV
        diag_vals = np.diag(mean_mat)
        pd.DataFrame({
            "network" : networks,
            "mean_isc": diag_vals,
        }).to_csv(
            os.path.join(args.out_dir, f"network_isc_order-{order}.csv"), index=False)

        print(f"[INFO] Saved cross-network ISC order-{order}  "
              f"(.npy cube + .csv mean matrix + per-network diagonal)")

    # 6c. Combined 3-D per-ROI array if subject sets match
    if isc_results:
        same_subs = len({tuple(v) for v in isc_subjects.values()}) == 1
        if same_subs:
            orders_sorted = sorted(isc_results.keys())
            isc_3d = np.stack([isc_results[o] for o in orders_sorted], axis=0)
            np.save(os.path.join(args.out_dir, "isc_all_orders.npy"), isc_3d)
            pd.Series(orders_sorted, name="order").to_csv(
                os.path.join(args.out_dir, "order_index.csv"), index=False)
            print(f"\n[INFO] Combined per-ROI: isc_all_orders.npy  shape={isc_3d.shape}")

    print_isc_summary(isc_results, order_labels)
    print(f"\n[DONE] Output in: {args.out_dir}\n")


if __name__ == "__main__":
    main()
import os
import json
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict
from nilearn import signal

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
DEFAULT_OUT = os.path.join(DERIV_ROOT, "isc_schaefer",
                           "across_brain_networks_analysis_and_high_pass_filtering_debug")

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-ROI and cross-network ISC with optional bootstrap CIs."
    )
    parser.add_argument("--csv",           default=DEFAULT_CSV)
    parser.add_argument("--out_dir",       default=DEFAULT_OUT)
    parser.add_argument("--bold_template", default=BOLD_TEMPLATE)
    parser.add_argument("--mask_template", default=MASK_TEMPLATE)
    parser.add_argument("--labels",        default=LABELS_PATH)
    parser.add_argument("--n_rois",        type=int, default=400)
    parser.add_argument("--nan_policy",    default="drop",
                        choices=["drop", "interpolate"])

    # === NEW (BOOTSTRAP) === CLI flags ==========================================
    # Toggle bootstrap on/off. All bootstrap logic is gated on args.bootstrap,
    # so leaving this flag off gives you exactly the original behaviour + speed.
    parser.add_argument("--bootstrap", action="store_true",
                        help="Enable bootstrap confidence intervals "
                             "(cross-network matrices + combined summary).")
    parser.add_argument("--bootstrap_per_roi", action="store_true",
                        help="Also bootstrap per-ROI ISC (expensive with 400 ROIs).")
    parser.add_argument("--n_boot",    type=int,   default=1000,
                        help="Number of bootstrap iterations (default 1000).")
    parser.add_argument("--ci_alpha",  type=float, default=0.05,
                        help="Alpha for CIs. 0.05 → 95%% CI (default).")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="RNG seed for reproducible bootstrap.")
    # ============================================================================

    return parser.parse_args()


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

    combos = (df[["subject", "session", "run"]]
              .drop_duplicates()
              .sort_values(["subject", "session", "run"]))

    found, missing = set(), []
    for _, row in combos.iterrows():
        sub, ses, run = row["subject"], int(row["session"]), int(row["run"])
        bold_path = build_path(bold_template, sub, ses, run)
        mask_path = build_path(mask_template, sub, ses, run)
        bold_ok   = os.path.exists(bold_path)
        mask_ok   = os.path.exists(mask_path)
        if bold_ok and mask_ok:
            found.add((sub, ses, run))
        else:
            if not bold_ok: missing.append((bold_path, "BOLD"))
            if not mask_ok: missing.append((mask_path, "MASK"))

    n_total, n_found = len(combos), len(found)
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
        raise FileNotFoundError("No valid file pairs found.")
    return found, missing

def load_lut(lut_path):
    labels = {}
    if not os.path.exists(lut_path):
        print(f"[WARN] LUT not found: {lut_path}.")
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
    parts = roi_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"


def build_network_roi_map(roi_names):
    net_map = defaultdict(list)
    for idx, name in enumerate(roi_names):
        net_map[get_network_from_label(name)].append(idx)

    ordered = {}
    for net in NETWORK_ORDER:
        if net in net_map:
            ordered[net] = np.array(net_map[net])
    for net in net_map:
        if net not in ordered:
            ordered[net] = np.array(net_map[net])
    return ordered


def apply_high_pass_filter(timecourses, tr, cutoff=0.01):
    return signal.clean(timecourses, t_r=tr, high_pass=cutoff,
                        detrend=True, standardize=True)


def extract_roi_timecourse(bold_data, mask_data, start_idx, end_idx,
                           n_rois=400, tr=0.61, high_pass_cutoff=0.01):
    n_vols  = bold_data.shape[-1]
    end_idx = min(end_idx, n_vols)
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) >= end_idx ({end_idx}).")

    bold_seg  = bold_data[:, :, :, start_idx:end_idx]
    mask_seg  = mask_data[:, :, :, start_idx:end_idx]
    T         = bold_seg.shape[-1]
    bold_flat = bold_seg.reshape(-1, T).astype(np.float64)
    mask_flat = mask_seg.reshape(-1, T).astype(np.int32)

    sig = np.full((n_rois, T), np.nan, dtype=np.float32)
    for t in range(T):
        m, b   = mask_flat[:, t], bold_flat[:, t]
        counts = np.bincount(m, minlength=n_rois + 1)
        sums   = np.bincount(m, weights=b, minlength=n_rois + 1)
        c, s   = counts[1:n_rois + 1], sums[1:n_rois + 1]
        valid  = c > 0
        sig[valid, t] = (s[valid] / c[valid]).astype(np.float32)

    return apply_high_pass_filter(sig, tr, cutoff=high_pass_cutoff)


def mean_centre(signal):
    return signal - np.nanmean(signal, axis=1, keepdims=True)


def interpolate_nans(signal):
    out = signal.copy()
    for r in range(signal.shape[0]):
        ts, nans = signal[r], np.isnan(signal[r])
        if nans.all() or not nans.any():
            continue
        idx = np.arange(len(ts))
        out[r] = np.interp(idx, idx[~nans], ts[~nans])
    return out


def average_segments(segments):
    if not segments:
        return None
    min_T   = min(s.shape[1] for s in segments)
    stacked = np.stack([s[:, :min_T] for s in segments], axis=0)
    return np.nanmean(stacked, axis=0)


def roi_to_network_timecourses(roi_tc, network_roi_map):
    networks = list(network_roi_map.keys())
    net_tc_array = np.stack(
        [np.nanmean(roi_tc[indices, :], axis=0) for indices in network_roi_map.values()],
        axis=0
    )
    return net_tc_array, networks


# === NEW (VECTORIZED) === Fast ISC routines =================================
# These replace the per-subject scipy.pearsonr loops with a single einsum/dot
# product. Result is mathematically identical (up to floating-point noise) but
# much faster, especially when called thousands of times inside a bootstrap.
def _vectorized_pearson(a, b, axis=-1, eps=1e-10):
    """
    Pearson correlation between paired arrays along `axis`.
    a, b : arrays with matching shape.
    Returns: correlations with that axis reduced.
    NaN returned where one side has zero variance.
    """
    a_c = a - a.mean(axis=axis, keepdims=True)
    b_c = b - b.mean(axis=axis, keepdims=True)
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis)) * np.sqrt((b_c ** 2).sum(axis=axis))
    out = np.full_like(num, np.nan, dtype=np.float64)
    ok  = den > eps
    out[ok] = num[ok] / den[ok]
    return out


def compute_per_roi_isc_fast(tc_stack):
    """
    Vectorized LOO-ISC per ROI.
    tc_stack : (n_subs, n_rois, T)   — may contain NaN timepoints.
    Returns  : (n_subs, n_rois) float32 ISC values.
    """
    n_subs, n_rois, _ = tc_stack.shape
    isc = np.full((n_subs, n_rois), np.nan, dtype=np.float32)

    # Drop timepoints where ANY subject has NaN — per-ROI, because different
    # ROIs have different censoring patterns.
    valid_per_roi = ~np.any(np.isnan(tc_stack), axis=0)   # (n_rois, T)

    for r in range(n_rois):
        vt = valid_per_roi[r]
        if vt.sum() < 3:
            continue
        x = tc_stack[:, r, vt].astype(np.float64)         # (n_subs, T_valid)
        total = x.sum(axis=0)
        others = (total[None, :] - x) / (n_subs - 1)       # (n_subs, T_valid)
        isc[:, r] = _vectorized_pearson(x, others, axis=1).astype(np.float32)

    return isc

def compute_cross_network_cube_fast(net_stack):
    """
    Vectorized LOO cross-network ISC. Computes the full (n_nets, n_nets) matrix
    for all subjects at once.
    net_stack : (n_subs, n_nets, T)
    Returns   : isc_cube  (n_nets, n_nets, n_subs)  — per-subject values
                mean_mat  (n_nets, n_nets)          — nanmean across subjects
    """
    n_subs, n_nets, T = net_stack.shape

    # Drop timepoints where ANY subject has NaN in ANY network
    valid_t = ~np.any(np.isnan(net_stack), axis=(0, 1))    # (T,)
    if valid_t.sum() < 3:
        return (np.full((n_nets, n_nets, n_subs), np.nan, dtype=np.float32),
                np.full((n_nets, n_nets), np.nan, dtype=np.float32))

    x = net_stack[:, :, valid_t].astype(np.float64)        # (n_subs, n_nets, Tv)

    total  = x.sum(axis=0)                                  # (n_nets, Tv)
    others = (total[None, :, :] - x) / (n_subs - 1)         # (n_subs, n_nets, Tv)

    x_c = x - x.mean(axis=2, keepdims=True)
    o_c = others - others.mean(axis=2, keepdims=True)
    x_n = np.sqrt((x_c ** 2).sum(axis=2))                   # (n_subs, n_nets)
    o_n = np.sqrt((o_c ** 2).sum(axis=2))                   # (n_subs, n_nets)

    # numerator[i, a, b] = sum_t x_c[i, a, t] * o_c[i, b, t]
    numer = np.einsum('iat,ibt->iab', x_c, o_c)             # (n_subs, n_nets, n_nets)
    denom = x_n[:, :, None] * o_n[:, None, :]               # (n_subs, n_nets, n_nets)

    isc_per_sub = np.full_like(numer, np.nan)
    ok = denom > 1e-10
    isc_per_sub[ok] = numer[ok] / denom[ok]

    # Transpose to (n_nets, n_nets, n_subs) to match your existing save format
    isc_cube = np.transpose(isc_per_sub, (1, 2, 0)).astype(np.float32)
    mean_mat = np.nanmean(isc_cube, axis=2).astype(np.float32)
    return isc_cube, mean_mat


#  NEW (BOOTSTRAP) Bootstrap + CI helpers
def find_common_subjects(subject_order_net_tc, order_labels):
    """Subjects present in ALL orders (required for matched-index combined boot)."""
    sets = [
        {s for s in subject_order_net_tc if order in subject_order_net_tc[s]}
        for order in order_labels
    ]
    return sorted(set.intersection(*sets)) if sets else []


def bootstrap_cross_network_per_order(net_stack, n_boot, rng):
    """
    Per-order bootstrap — NOT matched across orders.
    net_stack : (n_subs, n_nets, T)
    Returns   : (n_boot, n_nets, n_nets) of bootstrap-replicated mean matrices.
    """
    n_subs, n_nets, _ = net_stack.shape
    out = np.full((n_boot, n_nets, n_nets), np.nan, dtype=np.float32)
    for b in tqdm(range(n_boot), desc="    bootstrap", leave=False, ncols=80):
        idx = rng.integers(0, n_subs, size=n_subs)
        _, mean_mat = compute_cross_network_cube_fast(net_stack[idx])
        out[b] = mean_mat
    return out


def bootstrap_combined(net_stacks_by_order, n_boot, rng):
    """
    Matched-index bootstrap for the combined (across-orders) figure.

    net_stacks_by_order : dict {order: (n_subs, n_nets, T)}
                          MUST all share the same subject ordering.
    Returns:
        boot_by_order : dict {order: (n_boot, n_nets, n_nets)}
        combined_boot : (n_boot, n_nets, n_nets)
                        — mean across orders at matched bootstrap index.
    """
    orders = list(net_stacks_by_order.keys())
    first  = next(iter(net_stacks_by_order.values()))
    n_subs, n_nets, _ = first.shape

    boot_by_order = {o: np.full((n_boot, n_nets, n_nets), np.nan, dtype=np.float32)
                     for o in orders}

    for b in tqdm(range(n_boot), desc="    combined bootstrap",
                  leave=False, ncols=80):
        idx = rng.integers(0, n_subs, size=n_subs)   # ONE draw, applied to all orders
        for order in orders:
            resampled = net_stacks_by_order[order][idx]
            _, mean_mat = compute_cross_network_cube_fast(resampled)
            boot_by_order[order][b] = mean_mat

    combined = np.mean(
        np.stack([boot_by_order[o] for o in orders], axis=0),
        axis=0
    )
    return boot_by_order, combined


def bootstrap_per_roi(tc_stack, n_boot, rng):
    """
    Per-ROI bootstrap. Returns (n_boot, n_rois) of mean ISCs per bootstrap sample.
    Expensive — gated behind --bootstrap_per_roi.
    """
    n_subs = tc_stack.shape[0]
    n_rois = tc_stack.shape[1]
    out = np.full((n_boot, n_rois), np.nan, dtype=np.float32)
    for b in tqdm(range(n_boot), desc="    bootstrap per-ROI",
                  leave=False, ncols=80):
        idx = rng.integers(0, n_subs, size=n_subs)
        isc = compute_per_roi_isc_fast(tc_stack[idx])
        out[b] = np.nanmean(isc, axis=0)
    return out


def basic_ci(boot_samples, point_estimate, alpha=0.05):
    """
    'Basic' bootstrap CI (Efron & Tibshirani).
    boot_samples  : (n_boot, ...)
    point_estimate: (...)
    Returns       : lower, upper — same shape as point_estimate.
    """
    lo_q = np.nanpercentile(boot_samples, 100 * (1 - alpha / 2), axis=0)
    hi_q = np.nanpercentile(boot_samples, 100 * (alpha / 2),     axis=0)
    lower = 2 * point_estimate - lo_q
    upper = 2 * point_estimate - hi_q
    return lower, upper


def significance_mask(lower, upper):
    """True where the CI excludes zero (i.e., the cell is 'significant')."""
    return (lower > 0) | (upper < 0)


# Printing helpers
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
    n, col_w = len(networks), 10
    print(f"\n  Cross-network ISC — order {order}  (mean over subjects)")
    print("  " + " " * 12 + "".join(f"{net:>{col_w}}" for net in networks))
    print("  " + "-" * (12 + col_w * n))
    for i, net_a in enumerate(networks):
        row_vals = "".join(f"{mean_mat[i, j]:>{col_w}.4f}" for j in range(n))
        print(f"  {net_a:<12}{row_vals}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  ISC — Schaefer 400-ROI + 7-Network cross-ISC, LOO method")
    if args.bootstrap:
        print(f"  BOOTSTRAP ENABLED — {args.n_boot} iterations, "
              f"alpha={args.ci_alpha}, seed={args.seed}")
    print("=" * 65)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")
    print("=" * 65)

    # 1. Load & filter CSV
    df = pd.read_csv(args.csv)
    df = df[~df["skip"].astype(bool) & ~df["short_segment"].astype(bool)].copy()
    df["subject"] = df["subject"].astype(str).apply(
        lambda s: s if s.startswith("sub-") else f"sub-{s}")
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

    # 2. Preflight
    valid_pairs, _ = preflight_check(df, args.bold_template, args.mask_template)

    # 3. Atlas labels & network map
    labels    = load_lut(args.labels)
    roi_names = [labels.get(r, f"ROI_{r}") for r in range(1, n_rois + 1)]

    network_roi_map = build_network_roi_map(roi_names)
    networks        = list(network_roi_map.keys())
    n_networks      = len(networks)

    print(f"\n[INFO] Networks ({n_networks}): {networks}")
    for net, idx in network_roi_map.items():
        print(f"         {net:<14}: {len(idx)} ROIs")

    # 4. Extract timecourses
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
                sub, ses, run = row["subject"], int(row["session"]), int(row["run"])
                if (sub, ses, run) not in valid_pairs:
                    continue
                bold_path = build_path(args.bold_template, sub, ses, run)
                mask_path = build_path(args.mask_template, sub, ses, run)
                cache_key = (bold_path, mask_path)

                if cache_key not in file_cache:
                    tqdm.write(f"  [LOAD] {sub}  ses-{ses}  run-{run:03d}")
                    file_cache[cache_key] = (
                        nib.load(bold_path).get_fdata().astype(np.float32),
                        nib.load(mask_path).get_fdata().astype(np.int16))

                bold_data, mask_data = file_cache[cache_key]

                try:
                    sig = extract_roi_timecourse(
                        bold_data, mask_data,
                        int(row["scan_start_idx"]),
                        int(row["scan_end_idx"]),
                        n_rois=n_rois)
                except ValueError as e:
                    tqdm.write(f"  [WARN] {sub} ses-{ses} run-{run:03d} "
                               f"seg-{row['segment_num']}: {e}")
                    continue

                if args.nan_policy == "interpolate":
                    sig = interpolate_nans(sig)
                sig = mean_centre(sig)
                seg_timecourses[int(row["segment_num"])].append(sig)

            if seg_timecourses:
                averaged = {k: average_segments(v) for k, v in seg_timecourses.items()}
                full_roi_tc = np.concatenate(
                    [averaged[k] for k in sorted(averaged)], axis=1)
                subject_order_tc[subject][order] = full_roi_tc
                net_tc, _ = roi_to_network_timecourses(full_roi_tc, network_roi_map)
                subject_order_net_tc[subject][order] = net_tc

        del file_cache

    print_coverage_table(subject_order_tc, subjects, order_labels)

    # 5a. Per-ROI ISC (vectorized)
    isc_results, isc_subjects = {}, {}
    for order in order_labels:
        subs_with_order = [s for s in subjects if order in subject_order_tc[s]]
        if len(subs_with_order) < 2:
            print(f"\n[WARN] Order {order}: {len(subs_with_order)} subject(s) — skipping.")
            continue
        min_T    = min(subject_order_tc[s][order].shape[1] for s in subs_with_order)
        tc_stack = np.stack(
            [subject_order_tc[s][order][:, :min_T] for s in subs_with_order], axis=0)
        # === NEW (VECTORIZED) ===
        isc_mat = compute_per_roi_isc_fast(tc_stack)
        isc_results[order]  = isc_mat
        isc_subjects[order] = subs_with_order
        print(f"[INFO] Per-ROI ISC order-{order}: shape {isc_mat.shape}")

    # 5b. Cross-network ISC (vectorized)
    cross_isc_all, cross_isc_mean = {}, {}
    net_stacks_by_order           = {}   # kept around for bootstrap use
    cross_isc_subjects            = {}
    for order in order_labels:
        subs_with_order = [s for s in subjects if order in subject_order_net_tc[s]]
        if len(subs_with_order) < 2:
            print(f"\n[WARN] Order {order}: too few subjects for cross-network ISC.")
            continue
        min_T     = min(subject_order_net_tc[s][order].shape[1] for s in subs_with_order)
        net_stack = np.stack(
            [subject_order_net_tc[s][order][:, :min_T] for s in subs_with_order], axis=0)

        # === NEW (VECTORIZED) ===
        isc_cube, mean_mat        = compute_cross_network_cube_fast(net_stack)
        cross_isc_all[order]      = isc_cube
        cross_isc_mean[order]     = mean_mat
        net_stacks_by_order[order] = net_stack
        cross_isc_subjects[order] = subs_with_order
        print(f"[INFO] Cross-network ISC order-{order}: matrix shape {mean_mat.shape}")
        print_cross_isc_matrix(mean_mat, networks, order)

    # ========================================================================
    # === NEW (BOOTSTRAP) === Section 5c+5d: per-order boot, combined boot, CIs
    # ========================================================================
    boot_results = None    # will hold everything we want to save for plotting
    if args.bootstrap:
        rng = np.random.default_rng(args.seed)
        print("\n" + "=" * 65)
        print(f"  BOOTSTRAP — {args.n_boot} iterations")
        print("=" * 65)

        # --- 5c-1: per-order bootstrap (each order uses its own subject set) ---
        cross_isc_boot_per_order = {}
        for order, net_stack in net_stacks_by_order.items():
            print(f"\n  [boot] order-{order}  ({net_stack.shape[0]} subjects)")
            cross_isc_boot_per_order[order] = bootstrap_cross_network_per_order(
                net_stack, args.n_boot, rng)

        # --- 5c-2: combined bootstrap (matched indices over common subjects) ---
        common_subs = find_common_subjects(subject_order_net_tc,
                                           list(net_stacks_by_order.keys()))
        print(f"\n  [boot] common subjects for combined figure: {len(common_subs)}")

        combined_boot = None
        combined_boot_by_order = None
        if len(common_subs) >= 2 and len(net_stacks_by_order) >= 2:
            # Rebuild stacks restricted to common subjects, in consistent order
            common_stacks = {}
            for order in net_stacks_by_order.keys():
                min_T = min(subject_order_net_tc[s][order].shape[1] for s in common_subs)
                common_stacks[order] = np.stack(
                    [subject_order_net_tc[s][order][:, :min_T] for s in common_subs],
                    axis=0)
            combined_boot_by_order, combined_boot = bootstrap_combined(
                common_stacks, args.n_boot, rng)
        else:
            print("  [boot] skipping combined bootstrap "
                  "(need >=2 orders and >=2 common subjects).")

        # --- 5d: basic-method CIs + significance masks ------------------------
        ci_per_order = {}
        for order, boot_cube in cross_isc_boot_per_order.items():
            lo, hi = basic_ci(boot_cube, cross_isc_mean[order], alpha=args.ci_alpha)
            ci_per_order[order] = {
                "lower": lo, "upper": hi,
                "sig":   significance_mask(lo, hi),
            }

        combined_ci = None
        if combined_boot is not None:
            # Point estimate for combined figure: mean of per-order matrices
            # computed on common subjects (so it matches what boot replicates average).
            point_combined = np.mean(
                np.stack([compute_cross_network_cube_fast(common_stacks[o])[1]
                          for o in common_stacks], axis=0),
                axis=0)
            lo, hi = basic_ci(combined_boot, point_combined, alpha=args.ci_alpha)
            combined_ci = {
                "point": point_combined,
                "lower": lo, "upper": hi,
                "sig":   significance_mask(lo, hi),
            }

        # --- optional per-ROI bootstrap --------------------------------------
        per_roi_boot = {}
        if args.bootstrap_per_roi:
            print("\n  [boot] per-ROI (this will take a while)")
            for order, subs in isc_subjects.items():
                min_T = min(subject_order_tc[s][order].shape[1] for s in subs)
                tc_stack = np.stack(
                    [subject_order_tc[s][order][:, :min_T] for s in subs], axis=0)
                per_roi_boot[order] = bootstrap_per_roi(tc_stack, args.n_boot, rng)

        boot_results = {
            "cross_isc_boot_per_order": cross_isc_boot_per_order,
            "combined_boot_by_order":   combined_boot_by_order,
            "combined_boot":            combined_boot,
            "ci_per_order":             ci_per_order,
            "combined_ci":              combined_ci,
            "per_roi_boot":             per_roi_boot,
            "common_subjects":          common_subs,
        }

    # ========================================================================
    # 6. Save outputs
    # ========================================================================
    # 6a. Per-ROI ISC (unchanged)
    for order, isc_mat in isc_results.items():
        subs     = isc_subjects[order]
        mean_isc = np.nanmean(isc_mat, axis=0)
        np.save(os.path.join(args.out_dir, f"isc_order-{order}.npy"), isc_mat)
        np.save(os.path.join(args.out_dir, f"isc_mean_order-{order}.npy"), mean_isc)
        pd.Series(subs, name="subject").to_csv(
            os.path.join(args.out_dir, f"subjects_order-{order}.csv"), index=False)
        df_out = pd.DataFrame({
            "roi_id":   range(1, n_rois + 1),
            "roi_name": roi_names,
            "mean_isc": mean_isc,
        })
        for i, sub in enumerate(subs):
            df_out[sub] = isc_mat[i]
        df_out.to_csv(os.path.join(args.out_dir, f"isc_order-{order}.csv"),
                      index=False)
        print(f"[INFO] Saved per-ROI ISC order-{order}")

    # 6b. Cross-network ISC (unchanged point-estimate outputs)
    for order, mean_mat in cross_isc_mean.items():
        df_cross = pd.DataFrame(mean_mat, index=networks, columns=networks)
        df_cross.to_csv(os.path.join(args.out_dir, f"cross_isc_order-{order}.csv"))
        np.save(os.path.join(args.out_dir, f"cross_isc_order-{order}.npy"),
                cross_isc_all[order])
        diag_vals = np.diag(mean_mat)
        pd.DataFrame({"network": networks, "mean_isc": diag_vals}).to_csv(
            os.path.join(args.out_dir, f"network_isc_order-{order}.csv"), index=False)
        print(f"[INFO] Saved cross-network ISC order-{order}")

    # 6c. Combined per-ROI cube if subject sets match
    if isc_results and len({tuple(v) for v in isc_subjects.values()}) == 1:
        orders_sorted = sorted(isc_results.keys())
        isc_3d = np.stack([isc_results[o] for o in orders_sorted], axis=0)
        np.save(os.path.join(args.out_dir, "isc_all_orders.npy"), isc_3d)
        pd.Series(orders_sorted, name="order").to_csv(
            os.path.join(args.out_dir, "order_index.csv"), index=False)
        print(f"\n[INFO] Combined per-ROI: isc_all_orders.npy  shape={isc_3d.shape}")

    # === NEW (BOOTSTRAP) === 6d: save all bootstrap artifacts ================
    if boot_results is not None:
        boot_dir = os.path.join(args.out_dir, "bootstrap")
        os.makedirs(boot_dir, exist_ok=True)

        # Per-order bootstrap cubes  (n_boot, 7, 7)
        for order, cube in boot_results["cross_isc_boot_per_order"].items():
            np.save(os.path.join(boot_dir, f"cross_isc_boot_order-{order}.npy"), cube)

        # Per-order CIs + significance as a single .npz per order
        for order, ci in boot_results["ci_per_order"].items():
            np.savez(
                os.path.join(boot_dir, f"cross_isc_ci_order-{order}.npz"),
                point=cross_isc_mean[order],
                lower=ci["lower"], upper=ci["upper"], sig=ci["sig"],
                networks=np.array(networks))
            # Also write CSVs for easy inspection in a spreadsheet
            pd.DataFrame(ci["lower"], index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, f"cross_isc_lower_order-{order}.csv"))
            pd.DataFrame(ci["upper"], index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, f"cross_isc_upper_order-{order}.csv"))
            pd.DataFrame(ci["sig"].astype(int), index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, f"cross_isc_sig_order-{order}.csv"))

        # Combined bootstrap
        if boot_results["combined_boot"] is not None:
            np.save(os.path.join(boot_dir, "cross_isc_boot_combined.npy"),
                    boot_results["combined_boot"])
            for order, cube in boot_results["combined_boot_by_order"].items():
                np.save(os.path.join(boot_dir,
                        f"cross_isc_boot_combined_contrib_order-{order}.npy"), cube)

            ci = boot_results["combined_ci"]
            np.savez(
                os.path.join(boot_dir, "cross_isc_ci_combined.npz"),
                point=ci["point"], lower=ci["lower"], upper=ci["upper"], sig=ci["sig"],
                networks=np.array(networks))
            pd.DataFrame(ci["point"], index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, "cross_isc_combined_point.csv"))
            pd.DataFrame(ci["lower"], index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, "cross_isc_combined_lower.csv"))
            pd.DataFrame(ci["upper"], index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, "cross_isc_combined_upper.csv"))
            pd.DataFrame(ci["sig"].astype(int), index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, "cross_isc_combined_sig.csv"))

            # Also produce the thresholded "significant cells only" matrix
            thresholded = np.where(ci["sig"], ci["point"], np.nan)
            pd.DataFrame(thresholded, index=networks, columns=networks).to_csv(
                os.path.join(boot_dir, "cross_isc_combined_thresholded.csv"))

        # Per-ROI bootstrap
        for order, arr in boot_results["per_roi_boot"].items():
            np.save(os.path.join(boot_dir, f"per_roi_boot_order-{order}.npy"), arr)

        # Metadata file for reproducibility + plotting scripts
        meta = {
            "n_boot":          args.n_boot,
            "ci_alpha":        args.ci_alpha,
            "seed":            args.seed,
            "networks":        networks,
            "orders":          list(cross_isc_mean.keys()),
            "common_subjects": boot_results["common_subjects"],
            "subjects_per_order": {o: list(s) for o, s in cross_isc_subjects.items()},
            "bootstrap_per_roi": args.bootstrap_per_roi,
        }
        with open(os.path.join(boot_dir, "bootstrap_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[INFO] Bootstrap outputs saved to: {boot_dir}")
        print(f"       - cross_isc_boot_order-*.npy        ({args.n_boot} × 7 × 7 cubes)")
        print(f"       - cross_isc_ci_order-*.npz          (point + lower + upper + sig)")
        if boot_results["combined_boot"] is not None:
            print(f"       - cross_isc_boot_combined.npy       (combined cube)")
            print(f"       - cross_isc_ci_combined.npz         (combined CI + mask)")
            print(f"       - cross_isc_combined_thresholded.csv (for your figure)")
        print(f"       - bootstrap_meta.json               (plotting metadata)")

    print_isc_summary(isc_results, order_labels)
    print(f"\n[DONE] Output in: {args.out_dir}\n")


if __name__ == "__main__":
    main()
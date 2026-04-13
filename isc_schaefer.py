import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict



BIDS_ROOT  = "/lustre/disk/home/shared/cusacklab/foundcog/bids"
DERIV_ROOT = os.path.join(BIDS_ROOT, "derivatives", "faizan_analysis")

# Raw BOLD
# .../bids/sub-IRN78/ses-1/func/sub-IRN78_ses-1_task-videos_dir-AP_run-001_bold.nii.gz
BOLD_TEMPLATE = os.path.join(
    BIDS_ROOT,
    "{subject}", "ses-{session}", "func",
    "{subject}_ses-{session}_task-videos_dir-AP_run-{run:03d}_bold.nii.gz"
)

# 4-D Schaefer mask
# .../schaefer_backnorm/sub-IRN78/ses-1/func/mask_4d/
#     sub-IRN78_ses-1_task-videos_run-001_space-native_desc-mask4d.nii.gz
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
DEFAULT_OUT = os.path.join(DERIV_ROOT, "isc_schaefer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-ROI ISC using Schaefer atlas and 4D motion-aware masks."
    )
    parser.add_argument("--csv",           default=DEFAULT_CSV)
    parser.add_argument("--out_dir",       default=DEFAULT_OUT)
    parser.add_argument("--bold_template", default=BOLD_TEMPLATE,
        help="Path template. Placeholders: {subject} {session} {run}")
    parser.add_argument("--mask_template", default=MASK_TEMPLATE,
        help="Path template for 4D Schaefer mask. Same placeholders.")
    parser.add_argument("--labels",        default=LABELS_PATH,
        help="Schaefer .lut file: id  R  G  B  name")
    parser.add_argument("--n_rois",        type=int, default=400)
    parser.add_argument("--nan_policy",    default="drop",
        choices=["drop", "interpolate"],
        help="'drop' = intersect valid timepoints across subjects (default). "
             "'interpolate' = linear interpolation per ROI.")
    return parser.parse_args()



def build_path(template, subject, session, run):
    """
    Fill a path template for a specific subject / session / run.

    subject  : str   e.g. "sub-IRN78"   (must already have sub- prefix)
    session  : int   e.g. 1
    run      : int   e.g. 1  →  formatted as 001 via {run:03d} in template

    Called once for BOLD and once for mask — both share the same signature
    so there is no risk of mixing up arguments between the two.
    """
    return template.format(
        subject=str(subject),
        session=int(session),
        run=int(run)
    )


def preflight_check(df, bold_template, mask_template):
    """
    Before any heavy computation, resolve every (subject, session, run) pair
    in the CSV and report which files exist and which are missing.

    Prints a summary table and returns:
        found    : set of (subject, session, run) tuples where BOTH files exist
        missing  : list of (path, kind) for files not found
    """
    print("\n" + "=" * 70)
    print("  PREFLIGHT — checking all (subject, session, run) file pairs")
    print("=" * 70)

    # Unique combinations in the CSV
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

        # Show an example of what a resolved path looks like for the
        # first row, so the user can spot template mistakes immediately
        first = combos.iloc[0]
        print(f"\n  Example resolved BOLD path (first CSV row):")
        print(f"    {build_path(bold_template, first['subject'], first['session'], first['run'])}")
        print(f"  Example resolved MASK path:")
        print(f"    {build_path(mask_template, first['subject'], first['session'], first['run'])}")

    print("=" * 70)

    if n_found == 0:
        raise FileNotFoundError(
            "No valid file pairs found. Check --bold_template and --mask_template."
        )

    return found, missing


def load_lut(lut_path):
    """
    Parse a FreeSurfer-style .lut file.
    Format per line:  id   R   G   B   name
    e.g.:             1  0.47059  0.066667  0.50196  7Networks_LH_Vis_1
    Returns dict {roi_id (int): roi_name (str)}.
    """
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

def extract_roi_timecourse(bold_data, mask_data, start_idx, end_idx, n_rois=400):
    """
    Extract mean BOLD timecourse per ROI for one segment.

    The mask is 4-D (X, Y, Z, T): each timepoint independently maps voxels
    to ROI labels, so motion censoring is respected per TR.

    Parameters
    ----------
    bold_data  : np.ndarray (X, Y, Z, T_full)
    mask_data  : np.ndarray (X, Y, Z, T_full)  integer labels 0..n_rois
    start_idx  : int  inclusive
    end_idx    : int  exclusive

    Returns
    -------
    signal : np.ndarray (n_rois, T_seg)
        NaN where an ROI has no valid voxels at a given timepoint.
    """
    n_vols  = bold_data.shape[-1]
    end_idx = min(end_idx, n_vols)
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) >= end_idx ({end_idx}).")

    bold_seg  = bold_data[:, :, :, start_idx:end_idx]
    mask_seg  = mask_data[:, :, :, start_idx:end_idx]
    T         = bold_seg.shape[-1]

    # Flatten spatial dims → 1-D per timepoint
    bold_flat = bold_seg.reshape(-1, T).astype(np.float64)   # (n_vox, T)
    mask_flat = mask_seg.reshape(-1, T).astype(np.int32)     # (n_vox, T)

    signal = np.full((n_rois, T), np.nan, dtype=np.float32)

    # Loop over T timepoints (e.g. 50-150) instead of 400 ROIs.
    # np.bincount does a single optimised C pass over a 1-D array —
    # far faster than creating a large boolean (n_vox, T) mask 400 times.
    for t in range(T):
        m      = mask_flat[:, t]                          # (n_vox,)
        b      = bold_flat[:, t]                          # (n_vox,)
        counts = np.bincount(m, minlength=n_rois + 1)    # (n_rois+1,)
        sums   = np.bincount(m, weights=b, minlength=n_rois + 1)
        c = counts[1:n_rois + 1]                         # drop label-0 background
        s = sums  [1:n_rois + 1]
        valid = c > 0
        signal[valid, t] = (s[valid] / c[valid]).astype(np.float32)

    return signal   # (n_rois, T_seg)


def mean_centre(signal):
    """Subtract temporal mean per ROI row (NaN-safe)."""
    return signal - np.nanmean(signal, axis=1, keepdims=True)


def interpolate_nans(signal):
    """Linear interpolation of NaN timepoints per ROI. Entirely-NaN rows stay NaN."""
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
    """
    Average a list of (n_rois, T) arrays across the list axis.
    Truncates to min T if lengths differ (shouldn't happen within same
    segment_num but handled defensively).
    Uses nanmean so a motion-censored timepoint in one repetition does not
    destroy the average if other repetitions are valid.
    """
    if not segments:
        return None
    min_T   = min(s.shape[1] for s in segments)
    stacked = np.stack([s[:, :min_T] for s in segments], axis=0)
    return np.nanmean(stacked, axis=0)   # (n_rois, T)



#  ISC  (leave-one-out)
def loo_isc_single_roi(tc_matrix):
    """
    Leave-one-out ISC for one ROI.

    tc_matrix : (n_subjects, T)

    For each subject s:
        others_mean = mean of all other subjects at each timepoint
        isc[s]      = pearson_r(subject_s, others_mean)

    NaN policy: only timepoints valid for ALL subjects are used.

    Returns (n_subjects,) float32. NaN where correlation cannot be computed.
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
    print("  ISC SUMMARY")
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  ISC — Schaefer 400-ROI Atlas, LOO method")
    print("=" * 65)
    for k, v in vars(args).items():
        print(f"  {k:<18}: {v}")
    print("=" * 65)

    # ── 1. Load & filter CSV ──────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df = df[~df["skip"].astype(bool) & ~df["short_segment"].astype(bool)].copy()

    # Normalise subject column: always keep the "sub-" prefix so it matches
    # the filesystem directory names exactly (e.g. sub-IRN78, sub-ICC103A)
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

    # ── 2. Preflight: validate all paths before any heavy computation ─────────
    valid_pairs, _ = preflight_check(df, args.bold_template, args.mask_template)
    # valid_pairs is a set of (subject, session, run) tuples where both files exist

    # ── 3. Atlas labels ───────────────────────────────────────────────────────
    labels    = load_lut(args.labels)
    roi_names = [labels.get(r, f"ROI_{r}") for r in range(1, n_rois + 1)]

    # ── 4. Extract & average ROI timecourses ──────────────────────────────────
    #
    # Strategy:
    #   For each subject (outer loop):
    #     Load NIfTI files for this subject's runs into a small per-subject cache.
    #     For each (order, segment_num): collect timecourses across all valid rows.
    #     Average repetitions within the same segment_num (handles variable rep counts).
    #     Concatenate averaged segments in ascending segment_num order.
    #     → subject_order_tc[subject][order] = (n_rois, T_total)
    #     Clear the per-subject cache before moving to the next subject
    #     so memory never accumulates across subjects.

    subject_order_tc = defaultdict(dict)

    for subject in tqdm(subjects, desc="Subjects", ncols=80):

        # Per-subject file cache: cleared at the end of this subject's loop.
        # Keyed by (bold_path, mask_path) so the same run file is loaded
        # only once even if multiple segments / orders reference it.
        file_cache = {}

        sub_df = df[df["subject"] == subject]

        for order in order_labels:
            ord_df = sub_df[sub_df["order_label"] == order]
            if ord_df.empty:
                continue

            seg_timecourses = defaultdict(list)   # seg_num → list of (n_rois, T_seg)

            for _, row in ord_df.iterrows():
                sub = row["subject"]   # always == subject (same filter)
                ses = int(row["session"])
                run = int(row["run"])

                # Skip rows whose files did not pass the preflight check
                if (sub, ses, run) not in valid_pairs:
                    continue

                bold_path = build_path(args.bold_template, sub, ses, run)
                mask_path = build_path(args.mask_template, sub, ses, run)

                # Load into per-subject cache if not already loaded
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
                full_tc  = np.concatenate(
                    [averaged[k] for k in sorted(averaged)], axis=1
                )   # (n_rois, T_total)
                subject_order_tc[subject][order] = full_tc

        # ── Clear this subject's NIfTI cache before moving on ─────────────────
        # Without this, every loaded 4D file stays in RAM for the entire job,
        # which would exhaust memory across 100+ subjects.
        del file_cache

    print_coverage_table(subject_order_tc, subjects, order_labels)

    # ── 5. LOO-ISC per order ──────────────────────────────────────────────────
    isc_results  = {}
    isc_subjects = {}

    for order in order_labels:
        subs_with_order = [s for s in subjects if order in subject_order_tc[s]]

        if len(subs_with_order) < 2:
            print(f"\n[WARN] Order {order}: {len(subs_with_order)} subject(s) — "
                  "skipping ISC (need ≥ 2).")
            continue

        # Verify / harmonise temporal lengths
        lengths   = {s: subject_order_tc[s][order].shape[1] for s in subs_with_order}
        unique_Ts = set(lengths.values())
        if len(unique_Ts) > 1:
            print(f"\n[WARN] Order {order}: temporal length mismatch:")
            for s, T in sorted(lengths.items()):
                print(f"         {s}: T={T}")
            min_T = min(unique_Ts)
            print(f"       Truncating all to T={min_T}.")
        else:
            min_T = next(iter(unique_Ts))

        # Stack → (n_subs, n_rois, T)
        tc_stack = np.stack(
            [subject_order_tc[s][order][:, :min_T] for s in subs_with_order],
            axis=0
        )

        n_subs_ord = len(subs_with_order)
        isc_mat    = np.full((n_subs_ord, n_rois), np.nan, dtype=np.float32)

        for r in tqdm(range(n_rois),
                      desc=f"  LOO-ISC order-{order} ({n_subs_ord} subs, T={min_T})",
                      leave=False, ncols=80):
            isc_mat[:, r] = loo_isc_single_roi(tc_stack[:, r, :])

        isc_results[order]  = isc_mat
        isc_subjects[order] = subs_with_order
        print(f"[INFO] Order {order}: ISC done — shape {isc_mat.shape}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
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

        print(f"[INFO] Saved order-{order}: .npy + .csv")

    # Combined 3D array when subject sets are identical across all orders
    if isc_results:
        same_subs = len({tuple(v) for v in isc_subjects.values()}) == 1
        if same_subs:
            orders_sorted = sorted(isc_results.keys())
            isc_3d = np.stack([isc_results[o] for o in orders_sorted], axis=0)
            np.save(os.path.join(args.out_dir, "isc_all_orders.npy"), isc_3d)
            pd.Series(orders_sorted, name="order").to_csv(
                os.path.join(args.out_dir, "order_index.csv"), index=False)
            print(f"\n[INFO] Combined: isc_all_orders.npy  "
                  f"shape={isc_3d.shape}  (n_orders × n_subjects × n_rois)")
        else:
            print("\n[INFO] Subject sets differ across orders — "
                  "no combined array saved.")

    print_isc_summary(isc_results, order_labels)
    print(f"\n[DONE] Output in: {args.out_dir}\n")


if __name__ == "__main__":
    main()
import os
import glob
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm



def parse_args():
    DEFAULT_BASE = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_analysis"
    DEFAULT_CSV  = "per_order_alignment/segments_mapping_each_sub_usable.csv"
    DEFAULT_OUT  = os.path.join(DEFAULT_BASE, "isc_data")

    parser = argparse.ArgumentParser(
        description="Extract mean BOLD time courses per order/session/run for ISC."
    )
    parser.add_argument(
        "--base_dir", default=DEFAULT_BASE,
        help=f"Root derivatives folder containing sub-* directories. "
             f"Default: {DEFAULT_BASE}"
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to segmentation CSV file. Default: {DEFAULT_CSV}"
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT,
        help=f"Output directory for ISC .npy files. Default: {DEFAULT_OUT}"
    )
    return parser.parse_args()



def find_masked_bold(base_dir, subject, session):
    """
    Glob for masked BOLD files for a given subject / session.
    Returns a dict: {run_number (int): filepath}
    """
    pattern = os.path.join(
        base_dir, subject,
        f"ses-{session}", "func", "masked_bold",
        f"{subject}_ses-{session}_task-videos_run-*_space-native_desc-maskedbold.nii.gz"
    )
    files   = sorted(glob.glob(pattern))
    run_map = {}
    for f in files:
        basename = os.path.basename(f)
        run_str  = [p for p in basename.split("_") if p.startswith("run-")][0]
        run_num  = int(run_str.split("-")[1])
        run_map[run_num] = f
    return run_map


def extract_mean_timecourse(bold_data, start_idx, end_idx):
    """
    bold_data : 4-D array (X, Y, Z, T)  — already masked (zeros outside brain)
    Slices timepoints [start_idx:end_idx] and computes the mean across all
    non-zero voxels → shape (T_seg,).
    """
    segment    = bold_data[:, :, :, start_idx:end_idx]   # (X, Y, Z, T_seg)
    brain_mask = segment[:, :, :, 0] != 0                # (X, Y, Z) bool
    voxel_ts   = segment[brain_mask]                      # (n_voxels, T_seg)
    mean_tc    = voxel_ts.mean(axis=0)                    # (T_seg,)
    return mean_tc


def save_npy(out_dir, subject, data):
    """Create directory if needed and save numpy array."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject}.npy")
    np.save(out_path, data)
    return out_path


def print_file_table(isc_out_dir):
    """
    Scans the saved ISC output directory and prints a table with
    the number of unique subjects per order / session / run cell.
    """
    print("\n" + "=" * 55)
    print("  FILE COUNT TABLE  (scanned from disk)")
    print("=" * 55)

    if not os.path.exists(isc_out_dir):
        print(f"  [ERROR] Output directory not found: {isc_out_dir}")
        print("=" * 55)
        return

    print(f"  {'Order':<12} {'Session':<10} {'Run':<8} {'N Subjects'}")
    print("  " + "-" * 43)

    grand_total = 0

    for order in sorted(os.listdir(isc_out_dir)):
        order_path = os.path.join(isc_out_dir, order)
        if not os.path.isdir(order_path) or not order.startswith("order-"):
            continue
        for ses in sorted(os.listdir(order_path)):
            ses_path = os.path.join(order_path, ses)
            if not os.path.isdir(ses_path) or not ses.startswith("ses-"):
                continue
            for run in sorted(os.listdir(ses_path)):
                run_path = os.path.join(ses_path, run)
                if not os.path.isdir(run_path) or not run.startswith("run-"):
                    continue

                n = len(glob.glob(os.path.join(run_path, "sub-*.npy")))
                grand_total += n
                print(f"  {order:<12} {ses:<10} {run:<8} {n}")

    print("  " + "-" * 43)
    print(f"  {'TOTAL':<30} {grand_total}")
    print("=" * 55)


def sanity_check_alignment(isc_out_dir):
    """
    For every order → session → run leaf directory, loads all subject .npy
    files and verifies they all have the same number of timepoints (T).
    Prints a per-cell result and a final overall verdict.
    """
    print("\n" + "=" * 55)
    print("  SANITY CHECK: Time Alignment Across Subjects")
    print("=" * 55)
    print(f"  {'Cell':<35} {'T':<8} Status")
    print("  " + "-" * 50)

    all_pass  = True
    n_checked = 0
    n_failed  = 0

    if not os.path.exists(isc_out_dir):
        print(f"  [ERROR] Output directory not found: {isc_out_dir}")
        return False

    for order in sorted(os.listdir(isc_out_dir)):
        order_path = os.path.join(isc_out_dir, order)
        if not os.path.isdir(order_path) or not order.startswith("order-"):
            continue
        for ses in sorted(os.listdir(order_path)):
            ses_path = os.path.join(order_path, ses)
            if not os.path.isdir(ses_path) or not ses.startswith("ses-"):
                continue
            for run in sorted(os.listdir(ses_path)):
                run_path = os.path.join(ses_path, run)
                if not os.path.isdir(run_path) or not run.startswith("run-"):
                    continue

                npy_files = sorted(glob.glob(os.path.join(run_path, "sub-*.npy")))
                if not npy_files:
                    continue

                lengths     = {}
                load_errors = []
                for f in npy_files:
                    sub_name = os.path.splitext(os.path.basename(f))[0]
                    try:
                        arr = np.load(f)
                        lengths[sub_name] = arr.shape[0]
                    except Exception as e:
                        load_errors.append(f"{sub_name}: {e}")

                cell_label = f"{order}/{ses}/{run}"
                unique_Ts  = set(lengths.values())

                if load_errors:
                    all_pass = False
                    n_failed += 1
                    print(f"  {cell_label:<35} {'':8} LOAD ERROR")
                    for err in load_errors:
                        print(f"    └─ {err}")

                elif len(unique_Ts) == 1:
                    T_val = next(iter(unique_Ts))
                    print(f"  {cell_label:<35} {T_val:<8} OK")

                else:
                    all_pass = False
                    n_failed += 1
                    detail   = "  |  ".join(
                        f"{s}: T={t}" for s, t in sorted(lengths.items())
                    )
                    print(f"  {cell_label:<35} {'mixed':<8} MISMATCH")
                    print(f"    └─ {detail}")

                n_checked += 1

    print("  " + "-" * 50)
    if n_checked == 0:
        print("  [WARN] No .npy files found to check.")
    elif all_pass:
        print(f"\n  ALL CHECKS OK — {n_checked} cell(s), all subjects time-aligned.\n")
    else:
        print(f"\n  {n_failed}/{n_checked} cell(s) FAILED. See details above.\n")
    print("=" * 55)

    return all_pass


def main():
    args = parse_args()

    BASE_DIR    = args.base_dir
    ISC_OUT_DIR = args.out_dir
    CSV_PATH    = args.csv

    print("=" * 55)
    print("  fMRI ISC Data Extraction")
    print("=" * 55)
    print(f"  base_dir : {BASE_DIR}")
    print(f"  csv      : {CSV_PATH}")
    print(f"  out_dir  : {ISC_OUT_DIR}")
    print("=" * 55)

    # ── 1. Load & filter CSV ─────────────────────────────────────────────
    print(f"\n[INFO] Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Total rows in CSV       : {len(df)}")

    df_valid = df[~df["skip"].astype(bool) & ~df["short_segment"].astype(bool)].copy()
    print(f"[INFO] Rows after filtering    : {len(df_valid)}")

    # ── 2. Discover subjects from filesystem ─────────────────────────────
    subject_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("sub-") and os.path.isdir(os.path.join(BASE_DIR, d))
    ])
    print(f"[INFO] Subjects found on disk  : {len(subject_dirs)}")

    # ── 3. Build full list of jobs for the progress bar ──────────────────
    # Each job = one valid CSV row that has a corresponding file on disk.
    # We pre-collect them so tqdm has an accurate total.
    jobs = []
    for subject in subject_dirs:
        sub_id = subject.replace("sub-", "")
        sub_df = df_valid[df_valid["subject"].astype(str) == sub_id]
        if sub_df.empty:
            sub_df = df_valid[df_valid["subject"].astype(str) == subject]
        if sub_df.empty:
            continue
        for session in sub_df["session"].unique():
            ses_df  = sub_df[sub_df["session"] == session]
            run_map = find_masked_bold(BASE_DIR, subject, int(session))
            if not run_map:
                continue
            for _, row in ses_df.iterrows():
                run_num = int(row["run"])
                if run_num in run_map:
                    jobs.append((subject, session, run_map, row))

    print(f"[INFO] Total segments to process: {len(jobs)}\n")

    # ── 4. Main extraction loop with progress bar ─────────────────────────
    bold_cache  = {}   # keyed by (subject, session, run_num)
    total_saved = 0
    total_skip  = 0

    with tqdm(total=len(jobs), unit="seg", ncols=72,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
              ) as pbar:

        for (subject, session, run_map, row) in jobs:
            run_num   = int(row["run"])
            order     = str(row["order_label"]).strip()
            seg_num   = int(row["segment_num"])
            start_idx = int(row["scan_start_idx"])
            end_idx   = int(row["scan_end_idx"])

            pbar.set_description(
                f"{subject} | ses-{session} | run-{run_num} | order-{order}"
            )

            # ── Load BOLD (cached per subject/session/run) ────────────
            cache_key = (subject, session, run_num)
            if cache_key not in bold_cache:
                img = nib.load(run_map[run_num])
                bold_cache[cache_key] = img.get_fdata()

            bold_data = bold_cache[cache_key]
            n_vols    = bold_data.shape[-1]

            # ── Bounds check ─────────────────────────────────────────
            if end_idx > n_vols:
                tqdm.write(f"  [WARN] scan_end_idx {end_idx} > n_vols {n_vols} "
                           f"for {subject} ses-{session} run-{run_num} — clamping.")
                end_idx = n_vols

            # ── Extract & save ────────────────────────────────────────
            mean_tc = extract_mean_timecourse(bold_data, start_idx, end_idx)

            out_dir = os.path.join(
                ISC_OUT_DIR,
                f"order-{order}",
                f"ses-{int(session)}",
                f"run-{run_num}"
            )
            save_npy(out_dir, subject, mean_tc)
            total_saved += 1
            pbar.update(1)

    print(f"\n[INFO] Done — {total_saved} files saved, {total_skip} skipped.")

    # ── 5. File count table (from disk) ──────────────────────────────────
    print_file_table(ISC_OUT_DIR)

    # ── 6. Sanity check: time alignment ──────────────────────────────────
    sanity_check_alignment(ISC_OUT_DIR)


if __name__ == "__main__":
    main()
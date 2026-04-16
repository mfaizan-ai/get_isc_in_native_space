"""
compute_isc.py

Usage
-----
    python compute_isc.py                         # default paths
    python compute_isc.py --isc_dir /path/to/isc_data \\
                           --out_dir /path/to/results \\
                        --min_subjects 20
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as t_dist


def parse_args():
    DEFAULT_ISC = (
        "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives"
        "/faizan_analysis/isc_data"
    )
    DEFAULT_OUT = (
        "isc_results"
    )

    parser = argparse.ArgumentParser(
        description="Compute LOO ISC from pre-extracted BOLD time courses."
    )
    parser.add_argument(
        "--isc_dir", default=DEFAULT_ISC,
        help=f"Directory containing order/ses/run/.npy structure. "
             f"Default: {DEFAULT_ISC}"
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT,
        help=f"Directory to write results and figures. Default: {DEFAULT_OUT}"
    )
    parser.add_argument(
        "--min_subjects", type=int, default=20,
        help="Minimum number of subjects required to compute ISC for a cell. "
             "Cells below this threshold are skipped. Default: 20"
    )
    return parser.parse_args()

# Data loading 
def load_cell(cell_dir):
    """
    Load all sub-*.npy files from a leaf directory.
    Returns:
        subjects  : list of subject ID strings
        timecourses : (N, T) array — one row per subject
    """
    files    = sorted(glob.glob(os.path.join(cell_dir, "sub-*.npy")))
    subjects = [os.path.splitext(os.path.basename(f))[0] for f in files]
    arrays   = [np.load(f) for f in files]

    # Verify all time courses have the same length
    lengths = [a.shape[0] for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Time course length mismatch in {cell_dir}: {set(lengths)}"
        )

    return subjects, np.stack(arrays, axis=0)   # (N, T)


def iter_cells(isc_dir):
    """
    Walk the order / ses / run hierarchy.
    Yields (order_label, session, run, cell_dir) for every leaf directory.
    """
    for order in sorted(os.listdir(isc_dir)):
        op = os.path.join(isc_dir, order)
        if not os.path.isdir(op) or not order.startswith("order-"):
            continue
        for ses in sorted(os.listdir(op)):
            sp = os.path.join(op, ses)
            if not os.path.isdir(sp) or not ses.startswith("ses-"):
                continue
            for run in sorted(os.listdir(sp)):
                rp = os.path.join(sp, run)
                if not os.path.isdir(rp) or not run.startswith("run-"):
                    continue
                yield order, ses, run, rp

# compute ISC   
def fisher_z(r):
    """Apply Fisher z-transform, clipping to avoid inf at ±1."""
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def fisher_z_inv(z):
    """Inverse Fisher z-transform."""
    return np.tanh(z)


def loo_isc(timecourses):
    """
    Leave-one-out ISC.

    Parameters
    ----------
    timecourses : (N, T) array

    Returns
    -------
    isc_values : (N,) array — per-subject LOO ISC
    mean_isc   : float — group ISC (Fisher z averaged, back-transformed)
    std_isc    : float — SD of LOO ISC values (on z-scale, back-transformed)
    ci_low     : float — 95% CI lower bound (t-distribution on z-scores)
    ci_high    : float — 95% CI upper bound
    """
    N, T = timecourses.shape
    isc_values = np.zeros(N)

    for i in range(N):
        # Leave-one-out mean (average of all others)
        loo_mean = timecourses[np.arange(N) != i].mean(axis=0)
        # Pearson correlation between subject i and LOO mean
        isc_values[i] = np.corrcoef(timecourses[i], loo_mean)[0, 1]

    # Fisher z-transform for averaging and CI
    z_values = fisher_z(isc_values)
    mean_z   = z_values.mean()
    std_z    = z_values.std(ddof=1)
    se_z     = std_z / np.sqrt(N)
    t_crit   = t_dist.ppf(0.975, df=N - 1)

    mean_isc = fisher_z_inv(mean_z)
    std_isc  = fisher_z_inv(std_z)
    ci_low   = fisher_z_inv(mean_z - t_crit * se_z)
    ci_high  = fisher_z_inv(mean_z + t_crit * se_z)

    return isc_values, mean_isc, std_isc, ci_low, ci_high

# visualization
PALETTE = {
    "bg":        "#0f1117",
    "panel":     "#1a1d27",
    "accent":    "#5c7cfa",
    "mean":      "#f03e3e",
    "positive":  "#51cf66",
    "neutral":   "#868e96",
    "text":      "#ced4da",
    "subtext":   "#636e72",
    "grid":      "#2c2f3a",
}

def apply_base_style(fig, axes_list):
    """Apply dark theme to figure and all axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_list:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def plot_timecourses(subjects, timecourses, cell_label, out_path):
    """
    Plot individual subject time courses (thin, semi-transparent) with the
    group mean overlaid in red. One row per subject, plus a top panel for
    the grand mean.
    """
    N, T = timecourses.shape
    time = np.arange(T)
    group_mean = timecourses.mean(axis=0)

    # Cap height so the figure stays readable
    max_rows  = min(N, 12)
    fig_h     = 2.2 + max_rows * 0.55
    fig, axes = plt.subplots(
        max_rows + 1, 1, figsize=(14, fig_h),
        gridspec_kw={"height_ratios": [2.5] + [1] * max_rows}
    )

    apply_base_style(fig, axes)

    # ── Top panel: group mean ─────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(time, group_mean, color=PALETTE["mean"], lw=1.6, label="Group mean")
    ax0.fill_between(
        time,
        group_mean - timecourses.std(axis=0),
        group_mean + timecourses.std(axis=0),
        color=PALETTE["mean"], alpha=0.15, label="±1 SD"
    )
    ax0.set_title(f"{cell_label}  |  Time Courses  (N={N})",
                  fontsize=11, pad=8, color=PALETTE["text"])
    ax0.set_ylabel("Mean BOLD", fontsize=8)
    ax0.legend(fontsize=7, framealpha=0.2, labelcolor=PALETTE["text"])
    ax0.set_xlim(0, T - 1)

    # ── Per-subject rows ──────────────────────────────────────────────────
    shown = min(N, max_rows)
    for idx in range(shown):
        ax = axes[idx + 1]
        ax.plot(time, timecourses[idx], color=PALETTE["accent"],
                lw=0.7, alpha=0.85)
        ax.plot(time, group_mean, color=PALETTE["mean"],
                lw=0.6, alpha=0.45, linestyle="--")
        ax.set_ylabel(subjects[idx].replace("sub-", ""),
                      fontsize=6, rotation=0, labelpad=38, va="center")
        ax.set_xlim(0, T - 1)
        ax.yaxis.set_major_locator(MaxNLocator(2))
        if idx < shown - 1:
            ax.set_xticklabels([])

    if N > max_rows:
        axes[-1].set_xlabel(
            f"Volume  (showing {max_rows}/{N} subjects)", fontsize=8
        )
    else:
        axes[-1].set_xlabel("Volume", fontsize=8)

    fig.tight_layout(h_pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_isc_bar(subjects, isc_values, mean_isc, ci_low, ci_high,
                 cell_label, out_path):
    """
    Horizontal bar chart of per-subject LOO ISC values, sorted by magnitude.
    Group mean ± 95 % CI shown as a vertical band.
    """
    order_idx = np.argsort(isc_values)
    sorted_r  = isc_values[order_idx]
    sorted_s  = [subjects[i].replace("sub-", "") for i in order_idx]

    colors = [PALETTE["positive"] if r > 0 else "#fa5252" for r in sorted_r]

    fig_h = max(4, len(subjects) * 0.32)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    apply_base_style(fig, [ax])

    y = np.arange(len(sorted_r))
    ax.barh(y, sorted_r, color=colors, alpha=0.82, height=0.7, zorder=3)

    # Group mean line + CI band
    ax.axvline(mean_isc, color=PALETTE["mean"], lw=1.8,
               linestyle="-", label=f"Mean ISC = {mean_isc:.3f}", zorder=4)
    ax.axvspan(ci_low, ci_high, color=PALETTE["mean"], alpha=0.12,
               label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]", zorder=2)
    ax.axvline(0, color=PALETTE["neutral"], lw=0.8, linestyle="--", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_s, fontsize=7)
    ax.set_xlabel("Pearson r  (LOO ISC)", fontsize=9)
    ax.set_title(f"{cell_label}  |  Per-subject LOO ISC", fontsize=11,
                 pad=8, color=PALETTE["text"])
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=PALETTE["text"],
              loc="lower right")
    ax.set_xlim(-0.5, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_isc_summary(summary_df, out_path):
    """
    Grouped bar chart: mean ISC per order, grouped by session × run.
    One panel per order.
    """
    orders = sorted(summary_df["order"].unique())
    n_ord  = len(orders)

    fig, axes = plt.subplots(1, n_ord, figsize=(4.5 * n_ord, 5), sharey=True)
    if n_ord == 1:
        axes = [axes]
    apply_base_style(fig, axes)

    for ax, order in zip(axes, orders):
        sub_df  = summary_df[summary_df["order"] == order].copy()
        labels  = [f"ses{r.session}\nrun{r.run}" for r in sub_df.itertuples()]
        means   = sub_df["mean_isc"].values
        ci_lo   = means - sub_df["ci_low"].values
        ci_hi   = sub_df["ci_high"].values - means
        n_subs  = sub_df["n_subjects"].values

        x      = np.arange(len(means))
        colors = [PALETTE["positive"] if m > 0 else "#fa5252" for m in means]

        bars = ax.bar(x, means, color=colors, alpha=0.82, zorder=3,
                      yerr=[ci_lo, ci_hi],
                      error_kw=dict(ecolor=PALETTE["text"], lw=1.2,
                                    capsize=4, capthick=1.2))
        ax.axhline(0, color=PALETTE["neutral"], lw=0.8, linestyle="--")

        # Annotate N subjects
        for xi, (m, n) in enumerate(zip(means, n_subs)):
            offset = 0.02 if m >= 0 else -0.04
            ax.text(xi, m + np.sign(m) * (ci_hi[xi] + 0.03),
                    f"N={n}", ha="center", va="bottom",
                    fontsize=7, color=PALETTE["subtext"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(order, fontsize=10, color=PALETTE["text"])
        ax.set_xlabel("Session / Run", fontsize=8)

    axes[0].set_ylabel("Mean LOO ISC (Pearson r)", fontsize=9)
    fig.suptitle("Inter-Subject Correlation Summary", fontsize=13,
                 color=PALETTE["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)


def save_per_cell_results(out_dir, label, subjects, isc_values,
                          mean_isc, std_isc, ci_low, ci_high):
    """Save per-subject LOO ISC values as a CSV for this cell."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "subject":   subjects,
        "loo_isc":   isc_values,
        "mean_isc":  mean_isc,
        "std_isc":   std_isc,
        "ci_low":    ci_low,
        "ci_high":   ci_high,
    })
    fname = os.path.join(out_dir, f"{label}_loo_isc.csv")
    df.to_csv(fname, index=False)


def print_summary_table(summary_df):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 72)
    print("  ISC SUMMARY TABLE")
    print("=" * 72)
    print(f"  {'Order':<10} {'Session':<10} {'Run':<8} {'N':<6} "
          f"{'Mean ISC':>10} {'SD':>8} {'95% CI':>20}")
    print("  " + "-" * 68)
    for r in summary_df.itertuples():
        ci = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        print(f"  {r.order:<10} {r.session:<10} {r.run:<8} {r.n_subjects:<6} "
              f"{r.mean_isc:>+10.4f} {r.std_isc:>8.4f} {ci:>20}")
    print("=" * 72)


def main():
    args        = parse_args()
    ISC_DIR     = args.isc_dir
    OUT_DIR     = args.out_dir
    MIN_SUBS    = args.min_subjects

    print("=" * 60)
    print("  Inter-Subject Correlation  (Leave-One-Out)")
    print("=" * 60)
    print(f"  isc_dir      : {ISC_DIR}")
    print(f"  out_dir      : {OUT_DIR}")
    print(f"  min_subjects : {MIN_SUBS}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []
    skipped      = []

    for order, ses, run, cell_dir in iter_cells(ISC_DIR):
        cell_label = f"{order}/{ses}/{run}"

        # ── Load ─────────────────────────────────────────────────────────
        try:
            subjects, timecourses = load_cell(cell_dir)
        except Exception as e:
            print(f"\n[ERROR] {cell_label}: {e} — skipping.")
            continue

        N = len(subjects)

        # ── Minimum N gate ────────────────────────────────────────────────
        if N < MIN_SUBS:
            print(f"\n[SKIP] {cell_label}  N={N} < {MIN_SUBS} — insufficient "
                  f"subjects for reliable ISC.")
            skipped.append({"cell": cell_label, "n_subjects": N,
                            "reason": f"N < {MIN_SUBS}"})
            continue

        print(f"\n[CELL] {cell_label}  |  N={N}  |  T={timecourses.shape[1]}")

        # ── Compute LOO ISC ───────────────────────────────────────────────
        isc_values, mean_isc, std_isc, ci_low, ci_high = loo_isc(timecourses)
        print(f"       mean ISC = {mean_isc:+.4f}  "
              f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")

        # ── Output sub-directory for this cell ────────────────────────────
        safe_label = cell_label.replace("/", "_")
        cell_out   = os.path.join(OUT_DIR, order, ses, run)
        os.makedirs(cell_out, exist_ok=True)

        # ── Save per-subject CSV ──────────────────────────────────────────
        save_per_cell_results(
            cell_out, safe_label, subjects,
            isc_values, mean_isc, std_isc, ci_low, ci_high
        )

        # ── Plot time courses ─────────────────────────────────────────────
        tc_path = os.path.join(cell_out, f"{safe_label}_timecourses.png")
        plot_timecourses(subjects, timecourses, cell_label, tc_path)
        print(f"       [PLOT] time courses → {tc_path}")

        # ── Plot ISC bar chart ────────────────────────────────────────────
        isc_path = os.path.join(cell_out, f"{safe_label}_isc_bar.png")
        plot_isc_bar(subjects, isc_values, mean_isc, ci_low, ci_high,
                     cell_label, isc_path)
        print(f"       [PLOT] ISC bar     → {isc_path}")

        # ── Accumulate summary ────────────────────────────────────────────
        summary_rows.append({
            "order":      order,
            "session":    ses,
            "run":        run,
            "n_subjects": N,
            "mean_isc":   mean_isc,
            "std_isc":    std_isc,
            "ci_low":     ci_low,
            "ci_high":    ci_high,
        })

    # ── Summary CSV ───────────────────────────────────────────────────────
    if not summary_rows:
        print("\n[WARN] No cells passed the minimum-subjects threshold.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "isc_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[SAVE] Summary CSV → {summary_csv}")

    # ── Summary plot ──────────────────────────────────────────────────────
    summary_plot = os.path.join(OUT_DIR, "isc_summary_plot.png")
    plot_isc_summary(summary_df, summary_plot)
    print(f"[SAVE] Summary plot → {summary_plot}")

    # ── Skipped cells ─────────────────────────────────────────────────────
    if skipped:
        skip_df  = pd.DataFrame(skipped)
        skip_csv = os.path.join(OUT_DIR, "skipped_cells.csv")
        skip_df.to_csv(skip_csv, index=False)
        print(f"[SAVE] Skipped cells → {skip_csv}  ({len(skipped)} cells)")

    # ── Print summary table ───────────────────────────────────────────────
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
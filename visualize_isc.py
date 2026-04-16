"""
visualise_isc.py
─────────────────
Reads the ISC CSV files produced by isc_schaefer_bootstrap.py and generates
presentation-ready figures:

  1.  network_summary.png          — grouped bar chart: all orders × 7 networks
  2.  heatmap.png                  — networks × orders colour matrix
  3.  top_rois_<order>.png         — top-20 ROI bar chart per order
  4.  brain_<order>.png            — brain surface/glass map per order (needs nilearn)
  5.  subject_distribution.png     — per-subject ISC violin plot
  6.  cross_isc_order-<N>.png      — 7×7 cross-network ISC heatmap per order
  7.  cross_isc_grid.png           — all orders' 7×7 matrices in one comparison grid
  8.  cross_isc_offdiag.png        — off-diagonal cross-network structure (diagonal subtracted)

  === NEW (BOOTSTRAP) — only when --bootstrap flag is passed ===
  9.  cross_isc_order-<N>_boot.png — per-order 7×7 with significance stars + CI annotations
  10. cross_isc_combined_boot.png  — combined (mean over orders) matrix with CIs + thresholded
  11. cross_isc_offdiag_boot.png   — off-diagonal with thresholded panel (non-sig cells blanked)
  12. bootstrap_distributions.png  — histogram of bootstrap samples for diagonal + select cells

Usage
-----
  # Original behaviour (no bootstrap):
  python visualise_isc.py --isc_dir /path/to/isc_schaefer --out_dir ./figures

  # With bootstrap data:
  python visualise_isc.py --isc_dir /path/to/isc_schaefer --out_dir ./figures --bootstrap
"""
import os
import json
import argparse
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_ISC_DIR = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/"
    "derivatives/faizan_analysis/isc_schaefer/"
    "across_brain_networks_analysis_and_high_pass_filtering"
)
DEFAULT_OUT_DIR = os.path.join(DEFAULT_ISC_DIR, "figures")

NETWORK_COLOURS = {
    "Vis"        : "#4878CF",
    "SomMot"     : "#6ACC65",
    "DorsAttn"   : "#D65F5F",
    "SalVentAttn": "#B47CC7",
    "Limbic"     : "#C4AD66",
    "Cont"       : "#77BEDB",
    "Default"    : "#EE854A",
}

NETWORK_FULLNAMES = {
    "Vis"        : "Visual",
    "SomMot"     : "Somatomotor",
    "DorsAttn"   : "Dorsal Attention",
    "SalVentAttn": "Salience / Ventral Attn",
    "Limbic"     : "Limbic",
    "Cont"       : "Control",
    "Default"    : "Default Mode",
}

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                 "Limbic", "Cont", "Default"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise ISC results.")
    parser.add_argument("--isc_dir",   default=DEFAULT_ISC_DIR)
    parser.add_argument("--out_dir",   default=DEFAULT_OUT_DIR)
    parser.add_argument("--top_n",     type=int, default=20)
    parser.add_argument("--no_brain",  action="store_true")
    parser.add_argument("--dpi",       type=int, default=150)

    # === NEW (BOOTSTRAP) === flag to enable bootstrap figures ================
    parser.add_argument("--bootstrap", action="store_true",
                        help="Load bootstrap data from {isc_dir}/bootstrap/ "
                             "and generate CI-annotated + thresholded figures.")
    # =========================================================================
    return parser.parse_args()


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_all_orders(isc_dir):
    pattern = os.path.join(isc_dir, "isc_order-*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No isc_order-*.csv files found in {isc_dir}.")
    data = {}
    for f in files:
        order = os.path.basename(f).replace("isc_order-", "").replace(".csv", "")
        df    = pd.read_csv(f)
        data[order] = df
        print(f"  [LOAD] order-{order}: {len(df)} ROIs, "
              f"mean ISC = {df['mean_isc'].mean():.4f}")
    print(f"\n[INFO] Loaded {len(data)} orders: {sorted(data.keys())}")
    return data


def load_cross_isc_orders(isc_dir):
    pattern = os.path.join(isc_dir, "cross_isc_order-*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  [WARN] No cross_isc_order-*.csv files found in {isc_dir}.")
        return {}
    cross_data = {}
    for f in files:
        order = os.path.basename(f).replace("cross_isc_order-", "").replace(".csv", "")
        df    = pd.read_csv(f, index_col=0)
        cross_data[order] = df
        print(f"  [LOAD] cross_isc order-{order}: "
              f"{df.shape[0]}×{df.shape[1]} matrix")
    print(f"\n[INFO] Loaded {len(cross_data)} cross-network ISC matrices.")
    return cross_data


# === NEW (BOOTSTRAP) === bootstrap data loader ===============================
def load_bootstrap_data(isc_dir):
    """
    Load all bootstrap outputs from {isc_dir}/bootstrap/.

    Returns a dict with keys:
        meta              : dict from bootstrap_meta.json
        ci_per_order      : {order: {point, lower, upper, sig, networks}}
        combined_ci       : {point, lower, upper, sig, networks} or None
        boot_per_order    : {order: (n_boot, 7, 7) ndarray}
        combined_boot     : (n_boot, 7, 7) ndarray or None
        thresholded       : DataFrame or None (combined thresholded matrix)

    Returns None if the bootstrap directory doesn't exist.
    """
    boot_dir = os.path.join(isc_dir, "bootstrap")
    if not os.path.isdir(boot_dir):
        print(f"  [WARN] Bootstrap directory not found: {boot_dir}")
        print("         Run isc_schaefer_bootstrap.py with --bootstrap first.")
        return None

    result = {}

    # Metadata
    meta_path = os.path.join(boot_dir, "bootstrap_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            result["meta"] = json.load(f)
        print(f"  [LOAD] bootstrap_meta.json — "
              f"n_boot={result['meta']['n_boot']}, "
              f"seed={result['meta']['seed']}, "
              f"alpha={result['meta']['ci_alpha']}")
    else:
        result["meta"] = {}

    # Per-order CIs
    ci_per_order = {}
    ci_pattern = os.path.join(boot_dir, "cross_isc_ci_order-*.npz")
    for f in sorted(glob.glob(ci_pattern)):
        order = (os.path.basename(f)
                 .replace("cross_isc_ci_order-", "")
                 .replace(".npz", ""))
        d = np.load(f, allow_pickle=True)
        ci_per_order[order] = {
            "point":    d["point"],
            "lower":    d["lower"],
            "upper":    d["upper"],
            "sig":      d["sig"],
            "networks": list(d["networks"]),
        }
        n_sig = d["sig"].sum()
        n_tot = d["sig"].size
        print(f"  [LOAD] CI order-{order}: "
              f"{n_sig}/{n_tot} cells significant")
    result["ci_per_order"] = ci_per_order

    # Per-order bootstrap cubes
    boot_per_order = {}
    boot_pattern = os.path.join(boot_dir, "cross_isc_boot_order-*.npy")
    for f in sorted(glob.glob(boot_pattern)):
        order = (os.path.basename(f)
                 .replace("cross_isc_boot_order-", "")
                 .replace(".npy", ""))
        boot_per_order[order] = np.load(f)
        print(f"  [LOAD] bootstrap cube order-{order}: "
              f"shape {boot_per_order[order].shape}")
    result["boot_per_order"] = boot_per_order

    # Combined CI
    combined_path = os.path.join(boot_dir, "cross_isc_ci_combined.npz")
    if os.path.exists(combined_path):
        d = np.load(combined_path, allow_pickle=True)
        result["combined_ci"] = {
            "point":    d["point"],
            "lower":    d["lower"],
            "upper":    d["upper"],
            "sig":      d["sig"],
            "networks": list(d["networks"]),
        }
        n_sig = d["sig"].sum()
        n_tot = d["sig"].size
        print(f"  [LOAD] combined CI: {n_sig}/{n_tot} cells significant")
    else:
        result["combined_ci"] = None

    # Combined bootstrap cube
    combined_boot_path = os.path.join(boot_dir, "cross_isc_boot_combined.npy")
    if os.path.exists(combined_boot_path):
        result["combined_boot"] = np.load(combined_boot_path)
        print(f"  [LOAD] combined bootstrap cube: "
              f"shape {result['combined_boot'].shape}")
    else:
        result["combined_boot"] = None

    # Thresholded CSV
    thresh_path = os.path.join(boot_dir, "cross_isc_combined_thresholded.csv")
    if os.path.exists(thresh_path):
        result["thresholded"] = pd.read_csv(thresh_path, index_col=0)
        print(f"  [LOAD] thresholded combined matrix")
    else:
        result["thresholded"] = None

    return result
# =============================================================================


# ── Existing helpers (unchanged) ─────────────────────────────────────────────
def parse_network(roi_name):
    parts = roi_name.split("_")
    if len(parts) >= 3:
        return parts[2], parts[1]
    return "Unknown", "?"


def add_network_column(df):
    parsed        = df["roi_name"].apply(parse_network)
    df            = df.copy()
    df["network"] = parsed.apply(lambda x: x[0])
    df["hemi"]    = parsed.apply(lambda x: x[1])
    return df


def build_network_summary(data):
    rows = []
    for order, df in data.items():
        df = add_network_column(df)
        for net, grp in df.groupby("network"):
            rows.append({
                "order"   : order,
                "network" : net,
                "mean_isc": grp["mean_isc"].mean(),
                "sem_isc" : grp["mean_isc"].sem(),
            })
    return pd.DataFrame(rows)


def _reorder_matrix(df):
    present    = [n for n in NETWORK_ORDER if n in df.index]
    extra      = [n for n in df.index if n not in NETWORK_ORDER]
    ordered    = present + extra
    df         = df.loc[ordered, ordered]
    full_names = [NETWORK_FULLNAMES.get(n, n) for n in ordered]
    df.index   = full_names
    df.columns = full_names
    return df


def _reorder_array(arr, networks_from):
    """
    Reorder a numpy array so rows/cols match NETWORK_ORDER.
    networks_from: the current order of rows/cols in the array.
    Returns: reordered array, list of full names in new order.
    """
    present = [n for n in NETWORK_ORDER if n in networks_from]
    extra   = [n for n in networks_from if n not in NETWORK_ORDER]
    ordered = present + extra
    idx     = [networks_from.index(n) for n in ordered]
    out     = arr[np.ix_(idx, idx)]
    names   = [NETWORK_FULLNAMES.get(n, n) for n in ordered]
    return out, names


def _cross_isc_vrange(cross_data):
    all_vals = np.concatenate([df.values.ravel() for df in cross_data.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax     = np.abs(all_vals).max()
    return round(float(vmax) + 0.005, 3)


# ── Original figure functions (unchanged) ────────────────────────────────────
def plot_network_summary(summary_df, out_path, dpi=150):
    networks = [n for n in NETWORK_COLOURS if n in summary_df["network"].unique()]
    orders   = sorted(summary_df["order"].unique())
    n_nets   = len(networks)
    n_orders = len(orders)
    bar_w    = 0.7 / n_orders
    x        = np.arange(n_nets)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    order_alphas = np.linspace(1.0, 0.4, n_orders)

    for oi, order in enumerate(orders):
        sub     = summary_df[summary_df["order"] == order].set_index("network")
        means   = [sub.loc[n, "mean_isc"] if n in sub.index else np.nan
                   for n in networks]
        sems    = [sub.loc[n, "sem_isc"]  if n in sub.index else 0
                   for n in networks]
        colours = [NETWORK_COLOURS.get(n, "#aaaaaa") for n in networks]
        offset  = (oi - n_orders / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, color=colours,
               alpha=order_alphas[oi], label=f"Order {order}",
               edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([NETWORK_FULLNAMES.get(n, n) for n in networks],
                       fontsize=11, rotation=25, ha="right")
    ax.set_ylabel("Mean ISC (Pearson r)", fontsize=12)
    ax.set_title("Inter-Subject Correlation by Brain Network and Video Order",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Video Order", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=10)
    ax.set_ylim(bottom=min(0, summary_df["mean_isc"].min() - 0.02))
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_heatmap(summary_df, out_path, dpi=150):
    networks = [n for n in NETWORK_COLOURS if n in summary_df["network"].unique()]
    orders   = sorted(summary_df["order"].unique())
    matrix   = pd.DataFrame(index=networks, columns=orders, dtype=float)
    for _, row in summary_df.iterrows():
        if row["network"] in networks:
            matrix.loc[row["network"], row["order"]] = row["mean_isc"]
    matrix.index = [NETWORK_FULLNAMES.get(n, n) for n in matrix.index]
    vmax = max(abs(matrix.values[~np.isnan(matrix.values)].max()),
               abs(matrix.values[~np.isnan(matrix.values)].min()))
    fig, ax = plt.subplots(figsize=(max(8, len(orders) * 1.2), 6))
    fig.patch.set_facecolor("white")
    sns.heatmap(matrix.astype(float), ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                annot_kws={"size": 10}, linewidths=0.5, linecolor="#dddddd",
                cbar_kws={"label": "Mean ISC (Pearson r)", "shrink": 0.8})
    ax.set_title("ISC Heatmap: Brain Networks × Video Order",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Video Order", fontsize=12)
    ax.set_ylabel("Brain Network", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_top_rois(df, order, top_n, out_path, dpi=150):
    df  = add_network_column(df)
    top = df.nlargest(top_n, "mean_isc").sort_values("mean_isc")
    colours = [NETWORK_COLOURS.get(n, "#aaaaaa") for n in top["network"]]
    fig, ax = plt.subplots(figsize=(10, top_n * 0.38 + 1.5))
    fig.patch.set_facecolor("white")
    ax.barh(range(len(top)), top["mean_isc"],
            color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["roi_name"], fontsize=9)
    ax.set_xlabel("Mean ISC (Pearson r)", fontsize=11)
    ax.set_title(f"Top {top_n} ROIs — Order {order}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    seen_nets = top["network"].unique()
    patches   = [mpatches.Patch(color=NETWORK_COLOURS.get(n, "#aaaaaa"),
                                label=NETWORK_FULLNAMES.get(n, n))
                 for n in NETWORK_COLOURS if n in seen_nets]
    ax.legend(handles=patches, title="Network", fontsize=8,
              loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_brain_map(df, order, out_path, dpi=150):
    try:
        import nibabel as nib
        from nilearn import datasets, plotting
    except ImportError:
        print("  [SKIP] nilearn not installed — skipping brain maps.")
        return
    print(f"  [INFO] Fetching Schaefer atlas via nilearn (order-{order})...")
    atlas     = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    atlas_arr = atlas_img.get_fdata().astype(np.int16)
    isc_arr   = np.zeros(atlas_arr.shape, dtype=np.float32)
    for _, row in df.iterrows():
        isc_arr[atlas_arr == int(row["roi_id"])] = row["mean_isc"]
    isc_img = nib.Nifti1Image(isc_arr, atlas_img.affine)
    vmax    = round(max(abs(df["mean_isc"].max()),
                        abs(df["mean_isc"].min())) + 0.005, 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"ISC Brain Map — Order {order}",
                 fontsize=15, fontweight="bold", y=1.02)
    for ax, (view, kwargs) in zip(axes, {
        "sagittal": dict(display_mode="x", cut_coords=6),
        "coronal" : dict(display_mode="y", cut_coords=6),
        "axial"   : dict(display_mode="z", cut_coords=6),
    }.items()):
        plotting.plot_stat_map(isc_img, colorbar=True, cmap="RdBu_r",
                               symmetric_cbar=True, vmax=vmax,
                               title=view.capitalize(), axes=ax, **kwargs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_subject_distribution(data, out_path, dpi=150):
    rows = []
    for order, df in data.items():
        sub_cols = [c for c in df.columns if c.startswith("sub-")]
        for sub in sub_cols:
            rows.append({"order": order, "subject": sub,
                         "mean_isc": df[sub].mean(skipna=True)})
    if not rows:
        print("  [SKIP] No per-subject columns — skipping distribution plot.")
        return
    plot_df = pd.DataFrame(rows)
    orders  = sorted(plot_df["order"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(orders) * 1.5), 5))
    fig.patch.set_facecolor("white")
    sns.violinplot(data=plot_df, x="order", y="mean_isc", order=orders, ax=ax,
                   palette="muted", inner=None, linewidth=1.2, alpha=0.6)
    sns.stripplot(data=plot_df, x="order", y="mean_isc", order=orders, ax=ax,
                  color="black", size=4, alpha=0.6, jitter=True)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Video Order", fontsize=12)
    ax.set_ylabel("Mean ISC across ROIs (Pearson r)", fontsize=12)
    ax.set_title("Per-Subject ISC Distribution by Video Order",
                 fontsize=14, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# ── Original cross-network figures (unchanged) ───────────────────────────────
def plot_cross_isc_single(matrix_df, order, out_path, vmax=None, dpi=150):
    mat = _reorder_matrix(matrix_df.copy())
    if vmax is None:
        vals = mat.values[~np.isnan(mat.values)]
        vmax = round(float(np.abs(vals).max()) + 0.005, 3)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    sns.heatmap(mat.astype(float), ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                annot_kws={"size": 10, "weight": "normal"},
                linewidths=0.8, linecolor="#dddddd", square=True,
                cbar_kws={"label": "Mean ISC (Pearson r)", "shrink": 0.75})
    n = len(mat)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                     edgecolor="black", linewidth=2.5, clip_on=False))
    ax.set_title(f"Cross-Network ISC Matrix — Order {order}",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Column network  (group mean timecourse)", fontsize=11, labelpad=8)
    ax.set_ylabel("Row network  (left-out subject timecourse)", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", labelsize=10, rotation=30)
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_grid(cross_data, out_path, dpi=150):
    orders   = sorted(cross_data.keys())
    n_orders = len(orders)
    n_cols   = min(3, n_orders)
    n_rows   = int(np.ceil(n_orders / n_cols))
    vmax     = _cross_isc_vrange(cross_data)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7.5, n_rows * 6.5))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx, order in enumerate(orders):
        ax  = axes[idx // n_cols, idx % n_cols]
        mat = _reorder_matrix(cross_data[order].copy())
        sns.heatmap(mat.astype(float), ax=ax, cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                    annot_kws={"size": 8}, linewidths=0.6, linecolor="#dddddd",
                    square=True, cbar=True,
                    cbar_kws={"label": "Mean ISC (r)", "shrink": 0.7})
        n = len(mat)
        for i in range(n):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                         edgecolor="black", linewidth=2.0, clip_on=False))
        ax.set_title(f"Order {order}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Group mean network", fontsize=9, labelpad=6)
        ax.set_ylabel("Left-out subject network", fontsize=9, labelpad=6)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8, rotation=0)
    for idx in range(n_orders, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.suptitle("Cross-Network ISC Matrices — All Video Orders",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_offdiag(cross_data, out_path, dpi=150):
    orders = sorted(cross_data.keys())
    if not orders:
        return
    sample_mat = _reorder_matrix(cross_data[orders[0]].copy())
    net_labels = list(sample_mat.index)
    n_networks = len(net_labels)
    cube = np.full((len(orders), n_networks, n_networks), np.nan)
    for oi, order in enumerate(orders):
        cube[oi] = _reorder_matrix(cross_data[order].copy()).values.astype(float)
    mean_mat = np.nanmean(cube, axis=0)
    diag     = np.diag(mean_mat)
    offdiag  = mean_mat - diag[:, np.newaxis]
    np.fill_diagonal(offdiag, np.nan)
    offdiag_df = pd.DataFrame(offdiag, index=net_labels, columns=net_labels)
    vals = offdiag[~np.isnan(offdiag)]
    vmax = round(float(np.abs(vals).max()) + 0.002, 3)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    # Left: mean matrix
    mean_df = pd.DataFrame(mean_mat, index=net_labels, columns=net_labels)
    raw_max = round(float(np.abs(mean_mat[~np.isnan(mean_mat)]).max()) + 0.005, 3)
    sns.heatmap(mean_df.astype(float), ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-raw_max, vmax=raw_max, annot=True, fmt=".3f",
                annot_kws={"size": 10}, linewidths=0.8, linecolor="#dddddd",
                square=True,
                cbar_kws={"label": "Mean ISC (r)", "shrink": 0.75})
    n = len(net_labels)
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                          edgecolor="black", linewidth=2.5, clip_on=False))
    axes[0].set_title("Mean cross-network ISC\n(averaged over all orders)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[0].tick_params(axis="x", labelsize=9, rotation=30)
    axes[0].tick_params(axis="y", labelsize=9, rotation=0)

    # Right: off-diagonal
    sns.heatmap(offdiag_df.astype(float), ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                annot_kws={"size": 10}, linewidths=0.8, linecolor="#dddddd",
                square=True, mask=np.isnan(offdiag_df.values),
                cbar_kws={"label": "ISC deviation from diagonal (r)",
                           "shrink": 0.75})
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                          facecolor="#cccccc", edgecolor="black",
                          linewidth=1.5, clip_on=False))
    axes[1].set_title("Off-diagonal cross-network structure\n"
                      "(diagonal subtracted)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[1].tick_params(axis="x", labelsize=9, rotation=30)
    axes[1].tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================
# === NEW (BOOTSTRAP) === Bootstrap-specific figures ==========================
# =============================================================================

def _make_annot_with_stars(point_mat, sig_mat):
    """
    Build a string annotation matrix: value with * for significant cells.
    point_mat, sig_mat: same-shaped 2D arrays.
    Returns: 2D list of strings suitable for sns.heatmap annot.
    """
    n = point_mat.shape[0]
    annot = []
    for i in range(n):
        row = []
        for j in range(n):
            val = point_mat[i, j]
            if np.isnan(val):
                row.append("")
            elif sig_mat[i, j]:
                row.append(f"{val:.3f}*")
            else:
                row.append(f"{val:.3f}")
        annot.append(row)
    return annot


def _make_annot_with_ci(point_mat, lower_mat, upper_mat, sig_mat):
    """
    Build annotation: "0.123*\n[0.05, 0.20]" for significant,
                      "0.123\n[−0.01, 0.25]" for non-significant.
    """
    n = point_mat.shape[0]
    annot = []
    for i in range(n):
        row = []
        for j in range(n):
            v = point_mat[i, j]
            lo, hi = lower_mat[i, j], upper_mat[i, j]
            if np.isnan(v):
                row.append("")
            else:
                star = "*" if sig_mat[i, j] else ""
                row.append(f"{v:.3f}{star}\n[{lo:.2f}, {hi:.2f}]")
        annot.append(row)
    return annot


def plot_cross_isc_single_boot(ci_data, order, out_path, vmax=None, dpi=150):
    """
    Figure 9: Per-order 7×7 with significance stars and CI annotations.
    """
    point, lower, upper, sig, nets = (
        ci_data["point"], ci_data["lower"], ci_data["upper"],
        ci_data["sig"], ci_data["networks"])

    point_r, names = _reorder_array(point, nets)
    lower_r, _     = _reorder_array(lower, nets)
    upper_r, _     = _reorder_array(upper, nets)
    sig_r, _       = _reorder_array(sig.astype(float), nets)
    sig_r          = sig_r.astype(bool)
    n              = len(names)

    if vmax is None:
        vals = point_r[~np.isnan(point_r)]
        vmax = round(float(np.abs(vals).max()) + 0.005, 3)

    annot = _make_annot_with_ci(point_r, lower_r, upper_r, sig_r)
    annot_arr = np.array(annot, dtype=object)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        pd.DataFrame(point_r, index=names, columns=names).astype(float),
        ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=annot_arr, fmt="", annot_kws={"size": 8},
        linewidths=0.8, linecolor="#dddddd", square=True,
        cbar_kws={"label": "Mean ISC (Pearson r)", "shrink": 0.75})

    # Highlight diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                     edgecolor="black", linewidth=2.5, clip_on=False))

    # Bold border around significant off-diagonal cells
    for i in range(n):
        for j in range(n):
            if i != j and sig_r[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor="#FFD700", linewidth=2.0,
                             clip_on=False))

    ax.set_title(
        f"Cross-Network ISC with Bootstrap CIs — Order {order}\n"
        f"* = CI excludes zero · gold border = significant off-diagonal",
        fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Group mean network", fontsize=11, labelpad=8)
    ax.set_ylabel("Left-out subject network", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", labelsize=10, rotation=30)
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_combined_boot(combined_ci, out_path, dpi=150):
    """
    Figure 10: Combined (mean over orders) matrix: left = full with CIs,
    right = thresholded (non-significant cells blanked).
    """
    if combined_ci is None:
        print("  [SKIP] No combined CI data — skipping combined bootstrap figure.")
        return

    point, lower, upper, sig, nets = (
        combined_ci["point"], combined_ci["lower"], combined_ci["upper"],
        combined_ci["sig"], combined_ci["networks"])

    point_r, names = _reorder_array(point, nets)
    lower_r, _     = _reorder_array(lower, nets)
    upper_r, _     = _reorder_array(upper, nets)
    sig_r, _       = _reorder_array(sig.astype(float), nets)
    sig_r          = sig_r.astype(bool)
    n              = len(names)

    vals = point_r[~np.isnan(point_r)]
    vmax = round(float(np.abs(vals).max()) + 0.005, 3)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("white")

    # --- Left panel: full matrix with CI annotations ---
    annot = _make_annot_with_ci(point_r, lower_r, upper_r, sig_r)
    annot_arr = np.array(annot, dtype=object)

    df_full = pd.DataFrame(point_r, index=names, columns=names)
    sns.heatmap(df_full.astype(float), ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=annot_arr, fmt="",
                annot_kws={"size": 8}, linewidths=0.8, linecolor="#dddddd",
                square=True,
                cbar_kws={"label": "Mean ISC (r)", "shrink": 0.75})
    for i in range(n):
        ax = axes[0]
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                     edgecolor="black", linewidth=2.5, clip_on=False))
        for j in range(n):
            if i != j and sig_r[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor="#FFD700", linewidth=2.0,
                             clip_on=False))
    axes[0].set_title(
        "Combined cross-network ISC (all orders)\n"
        "with bootstrap 95% CIs · * = significant",
        fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[0].tick_params(axis="x", labelsize=9, rotation=30)
    axes[0].tick_params(axis="y", labelsize=9, rotation=0)

    # --- Right panel: thresholded (only significant cells shown) ---
    thresholded = np.where(sig_r, point_r, np.nan)
    df_thresh   = pd.DataFrame(thresholded, index=names, columns=names)
    mask_ns     = np.isnan(thresholded)

    sns.heatmap(df_thresh.astype(float), ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                annot_kws={"size": 10}, linewidths=0.8, linecolor="#dddddd",
                square=True, mask=mask_ns,
                cbar_kws={"label": "Mean ISC (r)", "shrink": 0.75})
    # Grey out non-significant cells
    for i in range(n):
        for j in range(n):
            if mask_ns[i, j]:
                axes[1].add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=True, facecolor="#e8e8e8",
                    edgecolor="#dddddd", linewidth=0.5, clip_on=False))
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                          edgecolor="black", linewidth=2.5, clip_on=False))

    axes[1].set_title(
        "Thresholded: only significant cells\n"
        "(bootstrap CI excludes zero)",
        fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[1].tick_params(axis="x", labelsize=9, rotation=30)
    axes[1].tick_params(axis="y", labelsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_offdiag_boot(cross_data, combined_ci, out_path, dpi=150):
    """
    Figure 11: Off-diagonal plot with a THIRD panel showing thresholded
    version where non-significant off-diagonal cells are blanked.
    """
    if combined_ci is None:
        print("  [SKIP] No combined CI — skipping bootstrap offdiag figure.")
        return

    orders = sorted(cross_data.keys())
    if not orders:
        return

    # Build mean matrix across orders (same as original offdiag)
    sample_mat = _reorder_matrix(cross_data[orders[0]].copy())
    net_labels = list(sample_mat.index)
    n          = len(net_labels)

    cube = np.full((len(orders), n, n), np.nan)
    for oi, order in enumerate(orders):
        cube[oi] = _reorder_matrix(cross_data[order].copy()).values.astype(float)
    mean_mat = np.nanmean(cube, axis=0)

    # Off-diagonal deviation
    diag    = np.diag(mean_mat)
    offdiag = mean_mat - diag[:, np.newaxis]
    np.fill_diagonal(offdiag, np.nan)

    # Get significance from combined CI, reordered to match
    sig_r, _ = _reorder_array(combined_ci["sig"].astype(float),
                              combined_ci["networks"])
    sig_r = sig_r.astype(bool)

    # Thresholded off-diagonal: blank non-significant cells AND diagonal
    offdiag_thresh = offdiag.copy()
    for i in range(n):
        for j in range(n):
            if i == j or not sig_r[i, j]:
                offdiag_thresh[i, j] = np.nan

    vals = offdiag[~np.isnan(offdiag)]
    vmax_off = round(float(np.abs(vals).max()) + 0.002, 3)
    raw_max  = round(float(np.abs(mean_mat[~np.isnan(mean_mat)]).max()) + 0.005, 3)

    fig, axes = plt.subplots(1, 3, figsize=(26, 7),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor("white")

    # Panel 1: mean matrix
    mean_df = pd.DataFrame(mean_mat, index=net_labels, columns=net_labels)
    sns.heatmap(mean_df.astype(float), ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-raw_max, vmax=raw_max, annot=True, fmt=".3f",
                annot_kws={"size": 9}, linewidths=0.8, linecolor="#dddddd",
                square=True,
                cbar_kws={"label": "Mean ISC (r)", "shrink": 0.7})
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                          edgecolor="black", linewidth=2.5, clip_on=False))
    axes[0].set_title("Mean cross-network ISC\n(all orders)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[0].set_xlabel("Group mean network", fontsize=9, labelpad=6)
    axes[0].set_ylabel("Left-out subject network", fontsize=9, labelpad=6)
    axes[0].tick_params(axis="x", labelsize=8, rotation=30)
    axes[0].tick_params(axis="y", labelsize=8, rotation=0)

    # Panel 2: off-diagonal (all)
    offdiag_df = pd.DataFrame(offdiag, index=net_labels, columns=net_labels)
    sns.heatmap(offdiag_df.astype(float), ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-vmax_off, vmax=vmax_off, annot=True, fmt=".3f",
                annot_kws={"size": 9}, linewidths=0.8, linecolor="#dddddd",
                square=True, mask=np.isnan(offdiag),
                cbar_kws={"label": "Deviation (r)", "shrink": 0.7})
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                          facecolor="#cccccc", edgecolor="black",
                          linewidth=1.5, clip_on=False))
    axes[1].set_title("Off-diagonal deviation\n(diagonal subtracted)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[1].set_xlabel("Group mean network", fontsize=9, labelpad=6)
    axes[1].set_ylabel("Left-out subject network", fontsize=9, labelpad=6)
    axes[1].tick_params(axis="x", labelsize=8, rotation=30)
    axes[1].tick_params(axis="y", labelsize=8, rotation=0)

    # Panel 3: thresholded off-diagonal (only significant cells)
    offdiag_thresh_df = pd.DataFrame(offdiag_thresh, index=net_labels,
                                     columns=net_labels)
    mask_blank = np.isnan(offdiag_thresh)
    sns.heatmap(offdiag_thresh_df.astype(float), ax=axes[2], cmap="RdBu_r",
                center=0, vmin=-vmax_off, vmax=vmax_off, annot=True, fmt=".3f",
                annot_kws={"size": 9}, linewidths=0.8, linecolor="#dddddd",
                square=True, mask=mask_blank,
                cbar_kws={"label": "Deviation (r)", "shrink": 0.7})
    for i in range(n):
        for j in range(n):
            if mask_blank[i, j]:
                axes[2].add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=True, facecolor="#e8e8e8",
                    edgecolor="#dddddd", linewidth=0.5, clip_on=False))
    for i in range(n):
        axes[2].add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                          facecolor="#cccccc", edgecolor="black",
                          linewidth=1.5, clip_on=False))
    axes[2].set_title("Thresholded off-diagonal\n(significant cells only)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[2].set_xlabel("Group mean network", fontsize=9, labelpad=6)
    axes[2].set_ylabel("Left-out subject network", fontsize=9, labelpad=6)
    axes[2].tick_params(axis="x", labelsize=8, rotation=30)
    axes[2].tick_params(axis="y", labelsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_bootstrap_distributions(boot_cube, ci_data, out_path, dpi=150):
    """
    Figure 12: Histograms of bootstrap samples for selected cells.
    Shows: all 7 diagonal cells + the 3 strongest off-diagonal cells.
    """
    if boot_cube is None or ci_data is None:
        print("  [SKIP] No combined bootstrap cube — skipping distribution plot.")
        return

    point, sig, nets = ci_data["point"], ci_data["sig"], ci_data["networks"]
    point_r, names = _reorder_array(point, nets)

    # Reorder the boot cube to match
    present = [n for n in NETWORK_ORDER if n in nets]
    extra   = [n for n in nets if n not in NETWORK_ORDER]
    ordered = present + extra
    idx     = [nets.index(n) for n in ordered]
    boot_r  = boot_cube[:, np.ix_(idx, idx)[0], :][:, :, idx]

    n = len(names)

    # Select cells to plot: all diagonals + top-3 off-diagonal by absolute value
    cells = []
    for i in range(n):
        cells.append((i, i, f"{names[i]}\n(within-network)", True))

    offdiag_vals = []
    for i in range(n):
        for j in range(n):
            if i != j:
                offdiag_vals.append((abs(point_r[i, j]), i, j))
    offdiag_vals.sort(reverse=True)
    for _, i, j in offdiag_vals[:3]:
        cells.append((i, j, f"{names[i]} →\n{names[j]}", False))

    n_cells = len(cells)
    n_cols  = min(5, n_cells)
    n_rows  = int(np.ceil(n_cells / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3))
    fig.patch.set_facecolor("white")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx_c, (i, j, label, is_diag) in enumerate(cells):
        ax = axes[idx_c // n_cols, idx_c % n_cols]
        samples = boot_r[:, i, j]
        colour  = "#4878CF" if is_diag else "#D65F5F"

        ax.hist(samples[~np.isnan(samples)], bins=40, color=colour,
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(point_r[i, j], color="black", linewidth=1.5,
                   linestyle="-", label=f"point={point_r[i,j]:.3f}")
        ax.axvline(0, color="grey", linewidth=1, linestyle="--", alpha=0.6)

        # Show CI
        lo = ci_data["lower"]
        hi = ci_data["upper"]
        lo_r, _ = _reorder_array(lo, nets)
        hi_r, _ = _reorder_array(hi, nets)
        ax.axvline(lo_r[i, j], color="red", linewidth=1, linestyle=":")
        ax.axvline(hi_r[i, j], color="red", linewidth=1, linestyle=":")

        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("ISC (r)", fontsize=7)

    # Hide unused axes
    for idx_c in range(n_cells, n_rows * n_cols):
        axes[idx_c // n_cols, idx_c % n_cols].set_visible(False)

    fig.suptitle(
        "Bootstrap Distributions (combined across orders)\n"
        "black = point estimate · red dashed = 95% CI bounds · "
        "grey dashed = zero",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# =============================================================================


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  ISC Visualisation")
    if args.bootstrap:
        print("  BOOTSTRAP figures enabled")
    print("=" * 60)
    print(f"  isc_dir : {args.isc_dir}")
    print(f"  out_dir : {args.out_dir}")
    print("=" * 60 + "\n")

    # Load data
    print("[LOAD] Per-ROI ISC files...")
    data = load_all_orders(args.isc_dir)

    print("\n[LOAD] Cross-network ISC matrix files...")
    cross_data = load_cross_isc_orders(args.isc_dir)

    # === NEW (BOOTSTRAP) === load bootstrap data if flag is set ===============
    boot_data = None
    if args.bootstrap:
        print("\n[LOAD] Bootstrap data...")
        boot_data = load_bootstrap_data(args.isc_dir)
        if boot_data is None:
            print("  [WARN] --bootstrap flag set but no bootstrap data found.")
            print("         Bootstrap figures will be skipped.")
    # =========================================================================

    # Network summary
    summary_df = build_network_summary(data)

    # Figure 1
    print("\n[FIG 1] Network summary bar chart...")
    plot_network_summary(summary_df,
                         os.path.join(args.out_dir, "network_summary.png"),
                         dpi=args.dpi)

    # Figure 2
    print("\n[FIG 2] Network × Order heatmap...")
    plot_heatmap(summary_df,
                 os.path.join(args.out_dir, "heatmap.png"),
                 dpi=args.dpi)

    # Figure 3
    print("\n[FIG 3] Top-ROI bar charts per order...")
    for order, df in data.items():
        plot_top_rois(df, order, args.top_n,
                      os.path.join(args.out_dir, f"top_rois_order-{order}.png"),
                      dpi=args.dpi)

    # Figure 4
    if not args.no_brain:
        print("\n[FIG 4] Brain glass maps per order...")
        for order, df in data.items():
            plot_brain_map(df, order,
                           os.path.join(args.out_dir,
                                        f"brain_order-{order}.png"),
                           dpi=args.dpi)
    else:
        print("\n[FIG 4] Brain maps skipped (--no_brain).")

    # Figure 5
    print("\n[FIG 5] Per-subject ISC distribution...")
    plot_subject_distribution(data,
                              os.path.join(args.out_dir,
                                           "subject_distribution.png"),
                              dpi=args.dpi)

    # Figures 6–8: cross-network (original, unchanged)
    if cross_data:
        shared_vmax = _cross_isc_vrange(cross_data)

        print("\n[FIG 6] Cross-network ISC matrix per order...")
        for order, mat_df in cross_data.items():
            plot_cross_isc_single(mat_df, order,
                                  os.path.join(args.out_dir,
                                               f"cross_isc_order-{order}.png"),
                                  vmax=shared_vmax, dpi=args.dpi)

        print("\n[FIG 7] Cross-network ISC grid (all orders)...")
        plot_cross_isc_grid(cross_data,
                            os.path.join(args.out_dir, "cross_isc_grid.png"),
                            dpi=args.dpi)

        print("\n[FIG 8] Cross-network off-diagonal structure...")
        plot_cross_isc_offdiag(cross_data,
                               os.path.join(args.out_dir,
                                            "cross_isc_offdiag.png"),
                               dpi=args.dpi)
    else:
        print("\n[FIG 6–8] Cross-network figures skipped.")

    # === NEW (BOOTSTRAP) === Figures 9–12 ====================================
    if boot_data is not None and cross_data:

        # Figure 9: per-order 7×7 with CIs and significance
        if boot_data["ci_per_order"]:
            print("\n[FIG 9] Per-order cross-network ISC with bootstrap CIs...")
            for order, ci in boot_data["ci_per_order"].items():
                plot_cross_isc_single_boot(
                    ci, order,
                    os.path.join(args.out_dir,
                                 f"cross_isc_order-{order}_boot.png"),
                    dpi=args.dpi)

        # Figure 10: combined matrix with CIs + thresholded
        print("\n[FIG 10] Combined cross-network ISC with bootstrap CIs...")
        plot_cross_isc_combined_boot(
            boot_data["combined_ci"],
            os.path.join(args.out_dir, "cross_isc_combined_boot.png"),
            dpi=args.dpi)

        # Figure 11: off-diagonal with thresholded third panel
        print("\n[FIG 11] Off-diagonal with bootstrap thresholding...")
        plot_cross_isc_offdiag_boot(
            cross_data,
            boot_data["combined_ci"],
            os.path.join(args.out_dir, "cross_isc_offdiag_boot.png"),
            dpi=args.dpi)

        # Figure 12: bootstrap distribution histograms
        print("\n[FIG 12] Bootstrap distribution histograms...")
        plot_bootstrap_distributions(
            boot_data["combined_boot"],
            boot_data["combined_ci"],
            os.path.join(args.out_dir, "bootstrap_distributions.png"),
            dpi=args.dpi)

    elif args.bootstrap:
        print("\n[FIG 9–12] Bootstrap figures skipped (no data found).")
    # =========================================================================

    # Summary
    print("\n" + "=" * 60)
    print("  Done. Figures saved:")
    print("=" * 60)
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".png"):
            print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
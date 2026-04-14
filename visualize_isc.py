"""
visualise_isc.py
─────────────────
Reads the ISC CSV files produced by isc_schaefer.py and generates
presentation-ready figures:

  1. network_summary.png        — grouped bar chart: all orders × 7 networks
  2. heatmap.png                — networks × orders colour matrix
  3. top_rois_<order>.png       — top-20 ROI bar chart per order
  4. brain_<order>.png          — brain surface/glass map per order (needs nilearn)
  5. subject_distribution.png   — per-subject ISC violin plot
  6. cross_isc_order-<N>.png    — 7×7 cross-network ISC heatmap per order
  7. cross_isc_grid.png         — all orders' 7×7 matrices in one comparison grid
  8. cross_isc_offdiag.png      — off-diagonal cross-network structure (diagonal subtracted)

Usage
-----
  python visualise_isc.py --isc_dir /path/to/isc_schaefer --out_dir ./figures
"""
import os
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
    "derivatives/faizan_analysis/isc_schaefer/across_brain_networks_analysis"
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

# Canonical order for matrix rows/columns
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise ISC results.")
    parser.add_argument("--isc_dir", default=DEFAULT_ISC_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top_n",   type=int, default=20)
    parser.add_argument("--no_brain", action="store_true")
    parser.add_argument("--dpi",     type=int, default=150)
    return parser.parse_args()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_all_orders(isc_dir):
    """
    Load every isc_order-*.csv file.
    Returns dict {order_label: DataFrame}
    Each DataFrame has columns: roi_id, roi_name, mean_isc, [sub-XXX ...]
    """
    pattern = os.path.join(isc_dir, "isc_order-*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No isc_order-*.csv files found in {isc_dir}.\n"
            "Check --isc_dir points to the right folder."
        )

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
    """
    Load every cross_isc_order-*.csv file produced by isc_schaefer_networks.py.

    Each CSV is a 7×7 DataFrame where the index and columns are network names
    and values are mean LOO-ISC across subjects.

    Returns dict {order_label: DataFrame (7×7)}
    """
    pattern = os.path.join(isc_dir, "cross_isc_order-*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  [WARN] No cross_isc_order-*.csv files found in {isc_dir}.")
        print("         Cross-network ISC figures will be skipped.")
        return {}

    cross_data = {}
    for f in files:
        order = os.path.basename(f).replace("cross_isc_order-", "").replace(".csv", "")
        df    = pd.read_csv(f, index_col=0)
        cross_data[order] = df
        print(f"  [LOAD] cross_isc order-{order}: {df.shape[0]}×{df.shape[1]} matrix")

    print(f"\n[INFO] Loaded {len(cross_data)} cross-network ISC matrices.")
    return cross_data


# ── Existing helpers ──────────────────────────────────────────────────────────

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


# ── Existing figures ──────────────────────────────────────────────────────────

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
        means   = [sub.loc[n, "mean_isc"] if n in sub.index else np.nan for n in networks]
        sems    = [sub.loc[n, "sem_isc"]  if n in sub.index else 0      for n in networks]
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
    vmax    = round(max(abs(df["mean_isc"].max()), abs(df["mean_isc"].min())) + 0.005, 2)

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


# ── NEW: Cross-network ISC matrix figures ─────────────────────────────────────

def _reorder_matrix(df):
    """
    Reorder the rows and columns of a 7×7 cross-ISC DataFrame to match
    NETWORK_ORDER. Any networks not in NETWORK_ORDER are appended at the end.
    Replaces abbreviated names with full names for axis labels.
    """
    present    = [n for n in NETWORK_ORDER if n in df.index]
    extra      = [n for n in df.index if n not in NETWORK_ORDER]
    ordered    = present + extra
    df         = df.loc[ordered, ordered]
    full_names = [NETWORK_FULLNAMES.get(n, n) for n in ordered]
    df.index   = full_names
    df.columns = full_names
    return df


def _cross_isc_vrange(cross_data):
    """Compute a symmetric colour scale across all orders for comparability."""
    all_vals = np.concatenate([df.values.ravel() for df in cross_data.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax     = np.abs(all_vals).max()
    return round(float(vmax) + 0.005, 3)


def plot_cross_isc_single(matrix_df, order, out_path, vmax=None, dpi=150):
    """
    Plot one 7×7 cross-network ISC heatmap for a single order.

    Rows = network A  (subject i's timecourse)
    Cols = network B  (mean of other subjects' timecourses)
    Cell = mean_ISC_AxB across all subjects in this order.

    The diagonal is the standard within-network ISC.
    Off-diagonal cells show cross-network stimulus-driven coupling.
    """
    mat = _reorder_matrix(matrix_df.copy())

    if vmax is None:
        vals = mat.values[~np.isnan(mat.values)]
        vmax = round(float(np.abs(vals).max()) + 0.005, 3)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        mat.astype(float),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 10, "weight": "normal"},
        linewidths=0.8,
        linecolor="#dddddd",
        square=True,
        cbar_kws={"label": "Mean ISC (Pearson r)", "shrink": 0.75},
    )

    # Highlight the diagonal cells with a bold border to distinguish
    # within-network ISC from cross-network ISC
    n = len(mat)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                   fill=False, edgecolor="black",
                                   linewidth=2.5, clip_on=False))

    ax.set_title(
        f"Cross-Network ISC Matrix — Order {order}\n"
        "Each cell = mean correlation between row network and column network across all subjects",
        fontsize=12, fontweight="bold", pad=14
    )
    ax.set_xlabel("Column network  (group mean timecourse)", fontsize=11, labelpad=8)
    ax.set_ylabel("Row network  (left-out subject timecourse)", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", labelsize=10, rotation=30)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_grid(cross_data, out_path, dpi=150):
    """
    Plot all orders' 7×7 matrices side by side in one figure so you can
    compare cross-network structure across video orders at a glance.

    Uses a shared colour scale so values are directly comparable.
    """
    orders   = sorted(cross_data.keys())
    n_orders = len(orders)

    # Layout: up to 3 columns
    n_cols = min(3, n_orders)
    n_rows = int(np.ceil(n_orders / n_cols))

    # Shared symmetric colour scale across all orders
    vmax = _cross_isc_vrange(cross_data)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7.5, n_rows * 6.5))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(n_rows, n_cols)  # always 2-D

    for idx, order in enumerate(orders):
        row_i = idx // n_cols
        col_i = idx  % n_cols
        ax    = axes[row_i, col_i]

        mat = _reorder_matrix(cross_data[order].copy())

        sns.heatmap(
            mat.astype(float),
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax, vmax=vmax,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 8},
            linewidths=0.6,
            linecolor="#dddddd",
            square=True,
            cbar=True,
            cbar_kws={"label": "Mean ISC (r)", "shrink": 0.7},
        )

        # Bold box around diagonal entries
        n = len(mat)
        for i in range(n):
            ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                       fill=False, edgecolor="black",
                                       linewidth=2.0, clip_on=False))

        ax.set_title(f"Order {order}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Group mean network", fontsize=9, labelpad=6)
        ax.set_ylabel("Left-out subject network", fontsize=9, labelpad=6)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8, rotation=0)

    # Hide any unused subplot panels
    for idx in range(n_orders, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        "Cross-Network ISC Matrices — All Video Orders\n"
        "(diagonal = within-network ISC, off-diagonal = cross-network ISC)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_cross_isc_offdiag(cross_data, out_path, dpi=150):
    """
    Off-diagonal structure plot: subtract each row's diagonal value from
    all entries in that row, then average across orders.

    This removes the within-network ISC baseline and reveals which
    cross-network pairs show *unexpectedly strong* coupling given the
    overall responsiveness of each network.

    A positive off-diagonal value means: network B is more coupled with
    network A than you would expect from chance, over and above each
    network's own within-network ISC.
    """
    orders = sorted(cross_data.keys())
    if not orders:
        return

    # Build a (n_orders, n_networks, n_networks) array after reordering
    sample_mat  = _reorder_matrix(cross_data[orders[0]].copy())
    net_labels  = list(sample_mat.index)
    n_networks  = len(net_labels)

    cube = np.full((len(orders), n_networks, n_networks), np.nan)
    for oi, order in enumerate(orders):
        mat = _reorder_matrix(cross_data[order].copy()).values.astype(float)
        cube[oi] = mat

    # Mean matrix across orders
    mean_mat = np.nanmean(cube, axis=0)

    # Subtract diagonal: for each row i, subtract mean_mat[i, i] from all columns
    diag      = np.diag(mean_mat)
    offdiag   = mean_mat - diag[:, np.newaxis]   # broadcast row-wise

    # Zero out the diagonal in the display (it's by definition 0 after subtraction)
    np.fill_diagonal(offdiag, np.nan)

    offdiag_df = pd.DataFrame(offdiag, index=net_labels, columns=net_labels)

    # Symmetric scale around 0 for the off-diagonal values
    vals = offdiag[~np.isnan(offdiag)]
    vmax = round(float(np.abs(vals).max()) + 0.002, 3)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    # Left panel: mean 7×7 matrix (averaged over orders)
    mean_df = pd.DataFrame(mean_mat, index=net_labels, columns=net_labels)
    raw_max = round(float(np.abs(mean_mat[~np.isnan(mean_mat)]).max()) + 0.005, 3)
    sns.heatmap(
        mean_df.astype(float),
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        vmin=-raw_max, vmax=raw_max,
        annot=True, fmt=".3f",
        annot_kws={"size": 10},
        linewidths=0.8, linecolor="#dddddd",
        square=True,
        cbar_kws={"label": "Mean ISC (r)", "shrink": 0.75},
    )
    n = len(net_labels)
    for i in range(n):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1,
                                        fill=False, edgecolor="black",
                                        linewidth=2.5, clip_on=False))
    axes[0].set_title("Mean cross-network ISC\n(averaged over all orders)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[0].tick_params(axis="x", labelsize=9, rotation=30)
    axes[0].tick_params(axis="y", labelsize=9, rotation=0)

    # Right panel: off-diagonal deviation (diagonal set to NaN → shown as grey)
    sns.heatmap(
        offdiag_df.astype(float),
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f",
        annot_kws={"size": 10},
        linewidths=0.8, linecolor="#dddddd",
        square=True,
        mask=np.isnan(offdiag_df.values),
        cbar_kws={"label": "ISC deviation from diagonal (r)", "shrink": 0.75},
    )
    # Show diagonal cells as grey
    for i in range(n):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1,
                                        fill=True, facecolor="#cccccc",
                                        edgecolor="black",
                                        linewidth=1.5, clip_on=False))
    axes[1].set_title(
        "Off-diagonal cross-network structure\n"
        "(diagonal subtracted — reveals cross-network coupling above baseline)",
        fontsize=12, fontweight="bold", pad=12
    )
    axes[1].set_xlabel("Group mean network", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Left-out subject network", fontsize=10, labelpad=6)
    axes[1].tick_params(axis="x", labelsize=9, rotation=30)
    axes[1].tick_params(axis="y", labelsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  ISC Visualisation")
    print("=" * 60)
    print(f"  isc_dir : {args.isc_dir}")
    print(f"  out_dir : {args.out_dir}")
    print("=" * 60 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[LOAD] Per-ROI ISC files...")
    data = load_all_orders(args.isc_dir)

    print("\n[LOAD] Cross-network ISC matrix files...")
    cross_data = load_cross_isc_orders(args.isc_dir)

    # ── Network summary ───────────────────────────────────────────────────────
    summary_df = build_network_summary(data)

    # ── Figure 1: Grouped bar chart ───────────────────────────────────────────
    print("\n[FIG 1] Network summary bar chart...")
    plot_network_summary(
        summary_df,
        os.path.join(args.out_dir, "network_summary.png"),
        dpi=args.dpi
    )

    # ── Figure 2: Network × Order heatmap ────────────────────────────────────
    print("\n[FIG 2] Network × Order heatmap...")
    plot_heatmap(
        summary_df,
        os.path.join(args.out_dir, "heatmap.png"),
        dpi=args.dpi
    )

    # ── Figure 3: Top-ROI bar charts per order ────────────────────────────────
    print("\n[FIG 3] Top-ROI bar charts per order...")
    for order, df in data.items():
        plot_top_rois(
            df, order, args.top_n,
            os.path.join(args.out_dir, f"top_rois_order-{order}.png"),
            dpi=args.dpi
        )

    # ── Figure 4: Brain maps per order ───────────────────────────────────────
    if not args.no_brain:
        print("\n[FIG 4] Brain glass maps per order...")
        for order, df in data.items():
            plot_brain_map(
                df, order,
                os.path.join(args.out_dir, f"brain_order-{order}.png"),
                dpi=args.dpi
            )
    else:
        print("\n[FIG 4] Brain maps skipped (--no_brain).")

    # ── Figure 5: Per-subject distribution ───────────────────────────────────
    print("\n[FIG 5] Per-subject ISC distribution...")
    plot_subject_distribution(
        data,
        os.path.join(args.out_dir, "subject_distribution.png"),
        dpi=args.dpi
    )

    # ── Figures 6–8: Cross-network ISC matrices ───────────────────────────────
    if cross_data:
        # Shared colour scale across all orders so matrices are comparable
        shared_vmax = _cross_isc_vrange(cross_data)

        # Figure 6: individual 7×7 heatmap per order
        print("\n[FIG 6] Cross-network ISC matrix per order...")
        for order, mat_df in cross_data.items():
            plot_cross_isc_single(
                mat_df, order,
                os.path.join(args.out_dir, f"cross_isc_order-{order}.png"),
                vmax=shared_vmax,
                dpi=args.dpi
            )

        # Figure 7: all orders in one comparison grid
        print("\n[FIG 7] Cross-network ISC grid (all orders)...")
        plot_cross_isc_grid(
            cross_data,
            os.path.join(args.out_dir, "cross_isc_grid.png"),
            dpi=args.dpi
        )

        # Figure 8: mean matrix + off-diagonal deviation
        print("\n[FIG 8] Cross-network off-diagonal structure...")
        plot_cross_isc_offdiag(
            cross_data,
            os.path.join(args.out_dir, "cross_isc_offdiag.png"),
            dpi=args.dpi
        )
    else:
        print("\n[FIG 6–8] Cross-network ISC figures skipped "
              "(no cross_isc_order-*.csv files found).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done. Figures saved:")
    print("=" * 60)
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".png"):
            print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
visualise_isc.py
─────────────────
Reads the ISC CSV files produced by isc_schaefer.py and generates
presentation-ready figures:

  1. network_summary.png   — grouped bar chart: all orders × 7 networks
  2. heatmap.png           — networks × orders colour matrix
  3. top_rois_<order>.png  — top-20 ROI bar chart per order
  4. brain_<order>.png     — brain surface/glass map per order (needs nilearn)

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

# Default paths 
DEFAULT_ISC_DIR = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/"
    "derivatives/faizan_analysis/isc_schaefer"
)
DEFAULT_OUT_DIR = os.path.join(DEFAULT_ISC_DIR, "figures")

# Schaefer 7-network names and their display colours
NETWORK_COLOURS = {
    "Vis"        : "#4878CF",   # blue
    "SomMot"     : "#6ACC65",   # green
    "DorsAttn"   : "#D65F5F",   # red
    "SalVentAttn": "#B47CC7",   # purple
    "Limbic"     : "#C4AD66",   # tan
    "Cont"       : "#77BEDB",   # light blue
    "Default"    : "#EE854A",   # orange
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


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise ISC results.")
    parser.add_argument("--isc_dir", default=DEFAULT_ISC_DIR,
        help="Directory containing isc_order-*.csv files.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
        help="Where to save figures.")
    parser.add_argument("--top_n", type=int, default=20,
        help="Number of top ROIs to show in per-order bar charts (default 20).")
    parser.add_argument("--no_brain", action="store_true",
        help="Skip brain surface maps (use if nilearn not installed).")
    parser.add_argument("--dpi", type=int, default=150,
        help="Figure DPI (default 150; use 300 for publication).")
    return parser.parse_args()


def load_all_orders(isc_dir):
    """
    Load every isc_order-*.csv file.
    Returns a dict  {order_label: DataFrame}
    Each DataFrame has columns: roi_id, roi_name, mean_isc, [sub-XXX, ...]
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


def parse_network(roi_name):
    """
    Extract network and hemisphere from a Schaefer ROI name.
    e.g.  '7Networks_LH_Vis_1'          → ('Vis', 'LH')
          '7Networks_RH_Default_PFC_1'   → ('Default', 'RH')
    Returns (network_str, hemisphere_str).
    """
    parts = roi_name.split("_")
    # parts[0] = '7Networks', parts[1] = hemisphere, parts[2] = network
    if len(parts) >= 3:
        hemi    = parts[1]              # LH or RH
        network = parts[2]              # Vis, SomMot, Default, etc.
        return network, hemi
    return "Unknown", "?"


def add_network_column(df):
    """Add 'network' and 'hemi' columns to a ROI dataframe."""
    parsed          = df["roi_name"].apply(parse_network)
    df              = df.copy()
    df["network"]   = parsed.apply(lambda x: x[0])
    df["hemi"]      = parsed.apply(lambda x: x[1])
    return df


def build_network_summary(data):
    """
    Collapse 400 ROIs → 7 networks per order.
    Returns a DataFrame with columns: order, network, mean_isc, sem_isc
    """
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




def plot_network_summary(summary_df, out_path, dpi=150):
    """
    Grouped bar chart: x = networks, groups = orders, colour per network.
    One of the clearest slides-friendly figures.
    """
    networks = [n for n in NETWORK_COLOURS if n in summary_df["network"].unique()]
    orders   = sorted(summary_df["order"].unique())

    n_nets   = len(networks)
    n_orders = len(orders)
    bar_w    = 0.7 / n_orders
    x        = np.arange(n_nets)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Use a greyscale gradient for orders so network colour dominates
    order_alphas = np.linspace(1.0, 0.4, n_orders)

    for oi, order in enumerate(orders):
        sub = summary_df[summary_df["order"] == order]
        sub = sub.set_index("network")

        means = [sub.loc[n, "mean_isc"] if n in sub.index else np.nan
                 for n in networks]
        sems  = [sub.loc[n, "sem_isc"]  if n in sub.index else 0
                 for n in networks]
        colours = [NETWORK_COLOURS.get(n, "#aaaaaa") for n in networks]

        offset = (oi - n_orders / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, means, bar_w,
                        color=colours,
                        alpha=order_alphas[oi],
                        label=f"Order {order}",
                        edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [NETWORK_FULLNAMES.get(n, n) for n in networks],
        fontsize=11, rotation=25, ha="right"
    )
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
    """
    Colour matrix where rows = networks, columns = orders, value = mean ISC.
    Clean single figure, great for comparing patterns at a glance.
    """
    networks = [n for n in NETWORK_COLOURS if n in summary_df["network"].unique()]
    orders   = sorted(summary_df["order"].unique())

    # Build pivot matrix
    matrix = pd.DataFrame(index=networks, columns=orders, dtype=float)
    for _, row in summary_df.iterrows():
        if row["network"] in networks:
            matrix.loc[row["network"], row["order"]] = row["mean_isc"]

    # Replace network abbreviations with full names for readability
    matrix.index = [NETWORK_FULLNAMES.get(n, n) for n in matrix.index]

    vmax = max(abs(matrix.values[~np.isnan(matrix.values)].max()),
               abs(matrix.values[~np.isnan(matrix.values)].min()))

    fig, ax = plt.subplots(figsize=(max(8, len(orders) * 1.2), 6))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f",
        annot_kws={"size": 10},
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"label": "Mean ISC (Pearson r)", "shrink": 0.8}
    )
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
    """
    Horizontal bar chart of the top N ROIs for one order.
    Bars are coloured by network.
    """
    df = add_network_column(df)
    top = df.nlargest(top_n, "mean_isc").sort_values("mean_isc")

    colours = [NETWORK_COLOURS.get(n, "#aaaaaa") for n in top["network"]]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.38 + 1.5))
    fig.patch.set_facecolor("white")

    bars = ax.barh(range(len(top)), top["mean_isc"],
                   color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["roi_name"], fontsize=9)
    ax.set_xlabel("Mean ISC (Pearson r)", fontsize=11)
    ax.set_title(f"Top {top_n} ROIs — Order {order}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Network legend
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
    """
    Project mean ISC values onto the Schaefer 400 atlas and display as a
    glass brain (3 projections: sagittal, coronal, axial).
    Requires nilearn.
    """
    try:
        import nibabel as nib
        from nilearn import datasets, plotting, image
    except ImportError:
        print("  [SKIP] nilearn not installed — skipping brain maps.")
        return

    # Fetch Schaefer 400 atlas in MNI space from nilearn
    print(f"  [INFO] Fetching Schaefer atlas via nilearn (order-{order})...")
    atlas     = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    atlas_arr = atlas_img.get_fdata().astype(np.int16)

    # Build ISC volume: fill each parcel with its mean ISC value
    isc_arr = np.zeros(atlas_arr.shape, dtype=np.float32)
    for _, row in df.iterrows():
        roi_id = int(row["roi_id"])
        isc_arr[atlas_arr == roi_id] = row["mean_isc"]

    isc_img = nib.Nifti1Image(isc_arr, atlas_img.affine)

    vmax = max(abs(df["mean_isc"].max()), abs(df["mean_isc"].min()))
    vmax = round(vmax + 0.005, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"ISC Brain Map — Order {order}",
                 fontsize=15, fontweight="bold", y=1.02)

    cuts = {
        "sagittal" : dict(display_mode="x", cut_coords=6),
        "coronal"  : dict(display_mode="y", cut_coords=6),
        "axial"    : dict(display_mode="z", cut_coords=6),
    }

    for ax, (view, kwargs) in zip(axes, cuts.items()):
        disp = plotting.plot_stat_map(
            isc_img,
            colorbar=True,
            cmap="RdBu_r",
            symmetric_cbar=True,
            vmax=vmax,
            title=view.capitalize(),
            axes=ax,
            **kwargs
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_subject_distribution(data, out_path, dpi=150):
    """
    For each order, show the distribution of per-subject mean ISC
    (averaged across all ROIs per subject) as a violin + strip plot.
    Answers: do subjects vary a lot within an order?
    """
    subject_cols_exist = False
    rows = []

    for order, df in data.items():
        # Subject columns are everything after roi_id, roi_name, mean_isc
        sub_cols = [c for c in df.columns if c.startswith("sub-")]
        if sub_cols:
            subject_cols_exist = True
            for sub in sub_cols:
                # Mean ISC for this subject across all ROIs
                val = df[sub].mean(skipna=True)
                rows.append({"order": order, "subject": sub, "mean_isc": val})

    if not subject_cols_exist or not rows:
        print("  [SKIP] No per-subject columns found — skipping distribution plot.")
        return

    plot_df = pd.DataFrame(rows)
    orders  = sorted(plot_df["order"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(orders) * 1.5), 5))
    fig.patch.set_facecolor("white")

    sns.violinplot(data=plot_df, x="order", y="mean_isc",
                   order=orders, ax=ax,
                   palette="muted", inner=None, linewidth=1.2, alpha=0.6)
    sns.stripplot(data=plot_df, x="order", y="mean_isc",
                  order=orders, ax=ax,
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
    data = load_all_orders(args.isc_dir)

    # ── Network-level summary ─────────────────────────────────────────────────
    summary_df = build_network_summary(data)

    # ── Figure 1: Grouped bar chart (headline figure) ─────────────────────────
    print("\n[FIG 1] Network summary bar chart...")
    plot_network_summary(
        summary_df,
        os.path.join(args.out_dir, "network_summary.png"),
        dpi=args.dpi
    )

    # ── Figure 2: Heatmap ─────────────────────────────────────────────────────
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
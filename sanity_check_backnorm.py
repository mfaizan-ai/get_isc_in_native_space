#!/usr/bin/env python3

"""
Sanity check for back-normalization outputs.
--------------------------------------------


for my previous experience, i leanred that, not only hte stats are important but looking at 
the saved plots are important to see the registration accuracy. 
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

# Solid red colormap for mask overlays (autumn renders yellow at value=1)
_RED = ListedColormap([(1, 0, 0, 1)])

_BIDS_DIR = "/lustre/disk/home/shared/cusacklab/foundcog/bids"

DEFAULTS = dict(
    output_dir    = f"{_BIDS_DIR}/derivatives/faizan_analysis/schaefer_backnorm",
    bids_dir      = _BIDS_DIR,
    template_mask = (
        f"{_BIDS_DIR}/derivatives/templates/rois/"
        "Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
    ),
    template_bg   = (
        f"{_BIDS_DIR}/derivatives/templates/mask/"
        "nihpd_asym_02-05_t2w_2mm.nii.gz"
    ),
    save_dir      = "./sanity_checks",
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--subject",       required=True,
                   help="Subject ID, e.g. IRN78")
    p.add_argument("--session",       default=None,
                   help="Session label e.g. '1'. If omitted, first session found is used.")
    p.add_argument("--run",           default=None,
                   help="Run label e.g. '001'. If omitted, first run found is used.")
    p.add_argument("--output_dir",    default=DEFAULTS["output_dir"],
                   help="Pipeline output root (faizan_analysis)")
    p.add_argument("--bids_dir",      default=DEFAULTS["bids_dir"],
                   help="BIDS root directory")
    p.add_argument("--template_mask", default=DEFAULTS["template_mask"],
                   help="Binary mask in template space")
    p.add_argument("--template_bg",   default=DEFAULTS["template_bg"],
                   help="T2w template image for background (overlay)")
    p.add_argument("--save_dir",      default=DEFAULTS["save_dir"],
                   help="Directory to save all output figures")
    return p.parse_args()

def resolve_paths(args) -> dict:
    """
    Locate all required files for the chosen subject/session/run.

    Output structure (after pipeline update):
      func/
        mask_native/   {pfx}_space-native_mask.nii.gz
        mask_4d/       {pfx}_space-native_desc-mask4d.nii.gz
        masked_bold/   {pfx}_space-native_desc-maskedbold.nii.gz

    mean_bold is NOT saved by the pipeline -- it is computed on the fly
    from the raw BOLD (fslmaths -Tmean equivalent via numpy mean).
    """
    subject = args.subject
    
    # -- locate session -------------------------------------------------------
    func_base = Path(args.output_dir) / f"sub-{subject}"
    if args.session:
        session = args.session
    else:
        ses_dirs = sorted(func_base.glob("ses-*"))
        if not ses_dirs:
            raise FileNotFoundError(f"No ses-* directories found under {func_base}")
        session = ses_dirs[0].name[4:]
        print(f"  Auto-selected session: {session}")

    func_dir    = func_base / f"ses-{session}" / "func"
    # Run auto-detection: look inside mask_native/ subdirectory
    mask_native_dir = func_dir / "mask_native"
    pfx_glob = f"sub-{subject}_ses-{session}_task-videos_run-*_space-native_mask.nii.gz"

    # -- locate run -----------------------------------------------------------
    if args.run:
        run = args.run
    else:
        hits = sorted(mask_native_dir.glob(pfx_glob))
        if not hits:
            raise FileNotFoundError(
                f"No pipeline outputs found in {mask_native_dir}\n"
                f"  Pattern: {pfx_glob}\n"
                f"  Make sure the pipeline has completed for sub-{subject}."
            )
        run = next(
            (p[4:] for p in hits[0].name.split("_") if p.startswith("run-")), None
        )
        print(f"  Auto-selected run: {run}")

    pfx = f"sub-{subject}_ses-{session}_task-videos_run-{run}"

    files = dict(
        # Pipeline outputs -- now in subdirectories
        mask_3d     = func_dir / "mask_native" / f"{pfx}_space-native_mask.nii.gz",
        mask_4d     = func_dir / "mask_4d"     / f"{pfx}_space-native_desc-mask4d.nii.gz",
        masked_bold = func_dir / "masked_bold" / f"{pfx}_space-native_desc-maskedbold.nii.gz",
        # Raw BOLD from BIDS (never modified)
        bold_raw    = (
            Path(args.bids_dir)
            / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}_bold.nii.gz"
        ),
        # Template files
        template_mask = Path(args.template_mask),
        template_bg   = Path(args.template_bg),
        # mean_bold is computed on the fly -- not a saved file
        # It is added to this dict after loading bold_raw (see load_mean_bold)
    )
    # -- check all files exist ------------------------------------------------
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        msg = "Missing files:\n" + "\n".join(f"  [{k}]  {files[k]}" for k in missing)
        raise FileNotFoundError(msg)

    print(f"\n  Subject : sub-{subject}")
    print(f"  Session : ses-{session}")
    print(f"  Run     : run-{run}")
    for k, v in files.items():
        print(f"    {k:<16} {v}")
    return files, subject, session, run

def load(path) -> tuple:
    """Load NIfTI, return (data_float32, img)."""
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img

def load_mean_bold(bold_path) -> tuple:
    """
    Compute mean BOLD on the fly (pipeline does not save it separately).
    Returns (mean_vol_float32, img) where mean_vol is 3D.
    """
    print("  Computing mean BOLD on the fly (not saved by pipeline)...")
    img  = nib.load(str(bold_path))
    data = img.get_fdata(dtype=np.float32)
    mean_vol = data.mean(axis=-1)
    return mean_vol, img

def mask_volume_mm3(mask_data, img) -> float:
    vox_mm3 = float(np.prod(np.abs(img.header.get_zooms()[:3])))
    return float((mask_data > 0).sum()) * vox_mm3

def percentile_clim(data, lo=2, hi=98):
    nonzero = data[data > 0]
    if nonzero.size == 0:
        return 0, 1
    return float(np.percentile(nonzero, lo)), float(np.percentile(nonzero, hi))

def axial_slices_with_mask(bg, mask, n=9):
    """Pick n axial slices that contain mask voxels."""
    z_idx = np.where(mask.any(axis=(0, 1)))[0]
    if len(z_idx) == 0:
        z_idx = np.arange(bg.shape[2])
    return z_idx[np.linspace(0, len(z_idx) - 1, n, dtype=int)]


def overlay_mosaic(ax_row, bg, mask, slices, bg_clim, title=""):
    for ax, z in zip(ax_row, slices):
        ax.imshow(bg[:, :, z].T, origin="lower", cmap="gray",
                  vmin=bg_clim[0], vmax=bg_clim[1], aspect="auto")
        m = np.ma.masked_where(mask[:, :, z] == 0, mask[:, :, z])
        ax.imshow(m.T, origin="lower", cmap=_RED,
                  vmin=0, vmax=1, alpha=0.6, aspect="auto")
        ax.set_title(f"z={z}", color="white", fontsize=7)
        ax.axis("off")
    if title and len(ax_row) > 0:
        ax_row[0].set_ylabel(title, color="white", fontsize=8, rotation=90,
                             labelpad=4)


def check_dimensions(files: dict) -> dict:
    print("\n" + "=" * 60)
    print("CHECK 1 — DIMENSIONS")
    print("=" * 60)

    _, bold_img    = load(files["bold_raw"])
    _, mask3d_img  = load(files["mask_3d"])
    _, mask4d_img  = load(files["mask_4d"])
    _, mbold_img   = load(files["masked_bold"])

    bold_shape   = bold_img.shape
    mask3d_shape = mask3d_img.shape
    mask4d_shape = mask4d_img.shape
    mbold_shape  = mbold_img.shape

    results = {}

    def chk(label, got, expected, store_key):
        ok  = got == expected
        sym = "OK  " if ok else "FAIL"
        print(f"  [{sym}] {label}")
        print(f"         got {got}  |  expected {expected}")
        results[store_key] = ok
        return ok

    chk("3D mask spatial dims match BOLD spatial dims",
        mask3d_shape[:3], bold_shape[:3], "mask3d_spatial")
    chk("4D mask spatial dims match BOLD spatial dims",
        mask4d_shape[:3], bold_shape[:3], "mask4d_spatial")
    chk("4D mask temporal dim matches BOLD temporal dim",
        mask4d_shape[3],  bold_shape[3],  "mask4d_temporal")
    chk("Masked BOLD spatial dims match raw BOLD spatial dims",
        mbold_shape[:3],  bold_shape[:3], "maskedbold_spatial")
    chk("Masked BOLD temporal dim matches raw BOLD temporal dim",
        mbold_shape[3],   bold_shape[3],  "maskedbold_temporal")

    bz = bold_img.header.get_zooms()[:3]
    mz = mask3d_img.header.get_zooms()[:3]
    print(f"\n  BOLD voxel size   : {tuple(round(float(v),2) for v in bz)} mm")
    print(f"  Mask voxel size   : {tuple(round(float(v),2) for v in mz)} mm")
    results["all_ok"] = all(results.values())
    return results


def check_voxel_counts(files: dict) -> dict:
    print("\n" + "=" * 60)
    print("CHECK 2 — VOXEL COUNTS & VOLUMES")
    print("=" * 60)

    tmask, t_img        = load(files["template_mask"])
    nmask, n_img        = load(files["mask_3d"])
    mask4d, m4d_img     = load(files["mask_4d"])
    masked_bold, mb_img = load(files["masked_bold"])

    t_vox = int((tmask > 0).sum())
    n_vox = int((nmask > 0).sum())
    t_mm3 = mask_volume_mm3(tmask, t_img)
    n_mm3 = mask_volume_mm3(nmask, n_img)
    ratio = t_mm3 / n_mm3 if n_mm3 > 0 else float("nan")

    print(f"  Template mask   : {t_vox:>8,} voxels   {t_mm3:>12,.1f} mm³")
    print(f"  Native 3D mask  : {n_vox:>8,} voxels   {n_mm3:>12,.1f} mm³")
    print(f"  Volume ratio (template/native mm³) : {ratio:.3f}  (expect ~1.0)")

    # Per-timepoint voxel count from the actual 4D mask
    n_vols = mask4d.shape[3]
    counts = np.array([(mask4d[..., t] > 0).sum() for t in range(n_vols)])
    n_zero = int((counts == 0).sum())
    print(f"\n  Per-volume mask voxel count in 4D mask ({n_vols} volumes):")
    print(f"    min voxels : {counts.min():,}")
    print(f"    max voxels : {counts.max():,}")
    print(f"    mean       : {counts.mean():,.1f}")
    print(f"    std        : {counts.std():,.1f}")
    print(f"    zero frames: {n_zero}  {'<-- WARNING' if n_zero > 0 else 'OK'}")

    return dict(t_vox=t_vox, n_vox=n_vox, t_mm3=t_mm3, n_mm3=n_mm3,
                ratio=ratio, counts=counts, n_zero=n_zero)


def check_motion_effect(files: dict) -> dict:
    """
    Use the actual saved 4D mask to track centroid shifts across timepoints.
    """
    print("\n" + "=" * 60)
    print("CHECK 3 — MOTION EFFECT ON 4D MASK CENTROID")
    print("=" * 60)

    mask4d, img = load(files["mask_4d"])
    n_vols      = mask4d.shape[3]

    xi = np.arange(mask4d.shape[0])
    yi = np.arange(mask4d.shape[1])
    zi = np.arange(mask4d.shape[2])
    XX, YY, ZZ = np.meshgrid(xi, yi, zi, indexing="ij")

    cx, cy, cz = [], [], []
    for t in range(n_vols):
        vol = (mask4d[..., t] > 0).astype(np.float32)
        s   = vol.sum()
        if s > 0:
            cx.append(float((XX * vol).sum() / s))
            cy.append(float((YY * vol).sum() / s))
            cz.append(float((ZZ * vol).sum() / s))
        else:
            cx.append(np.nan)
            cy.append(np.nan)
            cz.append(np.nan)

    cx, cy, cz = np.array(cx), np.array(cy), np.array(cz)

    for axis_label, arr in [("X", cx), ("Y", cy), ("Z", cz)]:
        valid = arr[~np.isnan(arr)]
        print(f"  Centroid {axis_label}  range: {valid.min():.2f} – {valid.max():.2f} voxels"
              f"   (shift = {valid.max()-valid.min():.2f} vx)")

    return dict(cx=cx, cy=cy, cz=cz)


def check_masked_epi(files: dict) -> dict:
    print("\n" + "=" * 60)
    print("CHECK 4 — MASKED EPI SIGNAL")
    print("=" * 60)

    bold_raw,    _ = load(files["bold_raw"])
    masked_bold, _ = load(files["masked_bold"])
    mask4d,      _ = load(files["mask_4d"])
    n_vols         = bold_raw.shape[3]

    mean_inside  = []
    mean_outside = []
    frac_masked  = []

    for t in range(n_vols):
        raw_vol = bold_raw[..., t]
        m       = mask4d[..., t] > 0
        inside  = raw_vol[m]
        outside = raw_vol[~m]
        mean_inside.append( float(inside.mean())  if inside.size  > 0 else 0.)
        mean_outside.append(float(outside.mean()) if outside.size > 0 else 0.)
        frac_masked.append( float(m.sum()) / float(m.size))

    mean_inside  = np.array(mean_inside)
    mean_outside = np.array(mean_outside)
    frac_masked  = np.array(frac_masked)
    snr          = float(mean_inside.mean() / (mean_inside.std() + 1e-9))

    print(f"  Within-mask mean signal  : {mean_inside.mean():>10.2f}")
    print(f"  Outside-mask mean signal : {mean_outside.mean():>10.2f}")
    print(f"  Signal ratio in/out      : {mean_inside.mean()/(mean_outside.mean()+1e-9):.2f}  (expect > 1)")
    print(f"  Temporal SNR (tSNR)      : {snr:.2f}")
    print(f"  Mean fraction of voxels masked : {frac_masked.mean()*100:.2f}%")

    # Sanity: masked BOLD should be zero exactly where 4D mask is zero
    t0_masked = masked_bold[..., 0]
    t0_mask   = mask4d[..., 0] > 0
    n_leak    = int((t0_masked[~t0_mask] != 0).sum())
    print(f"\n  Signal leaking outside mask at vol-0: "
          f"{n_leak}  {'<-- WARNING: masking incomplete' if n_leak > 0 else 'OK'}")

    return dict(mean_inside=mean_inside, mean_outside=mean_outside,
                frac_masked=frac_masked, snr=snr)


def fig_template_vs_native(files: dict, mean_bold_data, save_dir: Path, tag: str):
    print("\n  Plotting fig1: template vs native mask overlay ...")

    tmask, _ = load(files["template_mask"])
    tbg,   _ = load(files["template_bg"])
    nmask, _ = load(files["mask_3d"])
    nbg      = mean_bold_data   # already computed 3D array

    n_slices = 10
    t_slices = axial_slices_with_mask(tbg, tmask, n_slices)
    n_slices_native = axial_slices_with_mask(nbg, nmask, n_slices)

    fig, axes = plt.subplots(
        2, n_slices, figsize=(n_slices * 2.2, 5), facecolor="black"
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.02)

    overlay_mosaic(axes[0], tbg,  tmask, t_slices,
                   percentile_clim(tbg), title="Template space")
    overlay_mosaic(axes[1], nbg,  nmask, n_slices_native,
                   percentile_clim(nbg), title="Native space")

    fig.suptitle(f"{tag} — Mask: template (top) vs native (bottom)",
                 color="white", fontsize=11)
    out = save_dir / f"{tag}_fig1_template_vs_native_mask.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_motion_effect(motion: dict, vox_counts: dict,
                      save_dir: Path, tag: str):
    print("  Plotting fig2: motion effect on 4D mask ...")

    counts = vox_counts["counts"]
    cx, cy, cz = motion["cx"], motion["cy"], motion["cz"]
    t = np.arange(len(counts))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor="black")
    plt.subplots_adjust(hspace=0.45)

    def plot_ts(ax, y, ylabel, color):
        ax.plot(t, y, color=color, lw=0.8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors="white", labelsize=7)
        ax.axhline(np.nanmean(y), color="gray", lw=0.6, linestyle="--")

    plot_ts(axes[0], counts, "Voxel count\nin 4D mask", "#4ec9b0")
    plot_ts(axes[1], cx,     "Centroid X (vx)",         "#ce9178")
    plot_ts(axes[2], cy,     "Centroid Y (vx)",         "#dcdcaa")
    plot_ts(axes[3], cz,     "Centroid Z (vx)",         "#9cdcfe")
    axes[3].set_xlabel("Volume (timepoint)", color="white", fontsize=8)

    fig.suptitle(f"{tag} — Motion effect on 4D mask across time",
                 color="white", fontsize=11)
    out = save_dir / f"{tag}_fig2_4d_mask_motion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_masked_epi_signal(epi: dict, save_dir: Path, tag: str):
    print("  Plotting fig3: masked EPI signal ...")

    mi   = epi["mean_inside"]
    mo   = epi["mean_outside"]
    frac = epi["frac_masked"]
    t    = np.arange(len(mi))

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor="black")
    plt.subplots_adjust(hspace=0.45)

    def plot_ts(ax, y, ylabel, color):
        ax.plot(t, y, color=color, lw=0.8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
        ax.tick_params(colors="white", labelsize=7)

    axes[0].plot(t, mi, color="#4ec9b0", lw=0.8, label="inside mask")
    axes[0].plot(t, mo, color="#ce9178", lw=0.8, label="outside mask")
    axes[0].set_ylabel("Mean signal", color="white", fontsize=8)
    axes[0].set_facecolor("black")
    axes[0].legend(fontsize=7, facecolor="#1e1e1e", labelcolor="white",
                   edgecolor="gray")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("gray")
    axes[0].tick_params(colors="white", labelsize=7)

    ratio = mi / (mo + 1e-9)
    plot_ts(axes[1], ratio, "Signal ratio\n(in/out)", "#dcdcaa")
    axes[1].axhline(1.0, color="red", lw=0.6, linestyle="--")

    plot_ts(axes[2], frac * 100, "% voxels\nmasked", "#9cdcfe")
    axes[2].set_xlabel("Volume (timepoint)", color="white", fontsize=8)

    snr_str = f"tSNR={epi['snr']:.1f}"
    fig.suptitle(f"{tag} — Masked EPI signal  ({snr_str})",
                 color="white", fontsize=11)
    out = save_dir / f"{tag}_fig3_masked_epi_signal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out}")


def fig_mask_slices_native(files: dict, mean_bold_data, save_dir: Path, tag: str):
    print("  Plotting fig4: multi-plane 3D native mask on mean BOLD ...")

    nbg   = mean_bold_data
    nmask, _ = load(files["mask_3d"])

    n_slices = 8
    fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 2, 7),
                              facecolor="black")
    plt.subplots_adjust(hspace=0.05, wspace=0.02)
    clim = percentile_clim(nbg)

    # Axial (z)
    z_slices = axial_slices_with_mask(nbg, nmask, n_slices)
    for ax, z in zip(axes[0], z_slices):
        ax.imshow(nbg[:, :, z].T, origin="lower", cmap="gray",
                  vmin=clim[0], vmax=clim[1], aspect="auto")
        m = np.ma.masked_where(nmask[:, :, z] == 0, nmask[:, :, z])
        ax.imshow(m.T, origin="lower", cmap=_RED,
                  vmin=0, vmax=1, alpha=0.6, aspect="auto")
        ax.set_title(f"z={z}", color="white", fontsize=6)
        ax.axis("off")
    axes[0][0].set_ylabel("Axial", color="white", fontsize=8)

    # Coronal (y)
    y_idx = np.where(nmask.any(axis=(0, 2)))[0]
    y_slices = y_idx[np.linspace(0, len(y_idx) - 1, n_slices, dtype=int)]
    for ax, y in zip(axes[1], y_slices):
        ax.imshow(nbg[:, y, :].T, origin="lower", cmap="gray",
                  vmin=clim[0], vmax=clim[1], aspect="auto")
        m = np.ma.masked_where(nmask[:, y, :] == 0, nmask[:, y, :])
        ax.imshow(m.T, origin="lower", cmap=_RED,
                  vmin=0, vmax=1, alpha=0.6, aspect="auto")
        ax.set_title(f"y={y}", color="white", fontsize=6)
        ax.axis("off")
    axes[1][0].set_ylabel("Coronal", color="white", fontsize=8)

    # Sagittal (x)
    x_idx = np.where(nmask.any(axis=(1, 2)))[0]
    x_slices = x_idx[np.linspace(0, len(x_idx) - 1, n_slices, dtype=int)]
    for ax, x in zip(axes[2], x_slices):
        ax.imshow(nbg[x, :, :].T, origin="lower", cmap="gray",
                  vmin=clim[0], vmax=clim[1], aspect="auto")
        m = np.ma.masked_where(nmask[x, :, :] == 0, nmask[x, :, :])
        ax.imshow(m.T, origin="lower", cmap=_RED,
                  vmin=0, vmax=1, alpha=0.6, aspect="auto")
        ax.set_title(f"x={x}", color="white", fontsize=6)
        ax.axis("off")
    axes[2][0].set_ylabel("Sagittal", color="white", fontsize=8)

    fig.suptitle(f"{tag} — 3D native mask on mean BOLD (axial / coronal / sagittal)",
                 color="white", fontsize=11)
    out = save_dir / f"{tag}_fig4_mask_slices_native.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out}")



def fig_motion_mask_frames(files: dict, save_dir: Path, tag: str,
                           n_frames: int = 8):
    print("  Plotting fig5: 4D mask frames at selected timepoints ...")

    bold_raw,    _ = load(files["bold_raw"])
    masked_bold, _ = load(files["masked_bold"])
    mask4d,      _ = load(files["mask_4d"])
    nmask,       _ = load(files["mask_3d"])
    n_vols         = bold_raw.shape[3]

    frame_idx = np.linspace(0, n_vols - 1, n_frames, dtype=int)

    # Representative axial slice from the static 3D mask
    z_idx = np.where(nmask.any(axis=(0, 1)))[0]
    z_mid = int(np.median(z_idx)) if len(z_idx) > 0 else bold_raw.shape[2] // 2

    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 2.2, 5),
                              facecolor="black")
    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    for col, t in enumerate(frame_idx):
        raw_vol    = bold_raw[:, :, z_mid, t]
        mask_slice = mask4d[:, :, z_mid, t]     # actual 4D mask slice
        masked_vol = masked_bold[:, :, z_mid, t]
        clim       = percentile_clim(raw_vol)

        # Top row: raw BOLD with actual 4D mask overlay
        axes[0, col].imshow(raw_vol.T, origin="lower", cmap="gray",
                            vmin=clim[0], vmax=clim[1], aspect="auto")
        m = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[0, col].imshow(m.T, origin="lower", cmap=_RED,
                            vmin=0, vmax=1, alpha=0.5, aspect="auto")
        axes[0, col].set_title(f"t={t}", color="white", fontsize=7)
        axes[0, col].axis("off")

        # Bottom row: masked BOLD signal
        mclim = percentile_clim(masked_vol)
        axes[1, col].imshow(masked_vol.T, origin="lower", cmap="hot",
                            vmin=mclim[0], vmax=mclim[1], aspect="auto")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Raw + 4D mask", color="white", fontsize=7)
    axes[1, 0].set_ylabel("Masked BOLD",   color="white", fontsize=7)

    fig.suptitle(
        f"{tag} — 4D motion-aware mask at {n_frames} timepoints  (z={z_mid})\n"
        f"Top: raw BOLD + actual 4D mask overlay  |  Bottom: masked BOLD signal",
        color="white", fontsize=10,
    )
    out = save_dir / f"{tag}_fig5_motion_mask_frames.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out}")


def print_summary(dims: dict, vox: dict, epi: dict, tag: str):
    print("\n" + "=" * 60)
    print(f"SUMMARY  —  {tag}")
    print("=" * 60)

    all_dim_ok  = dims.get("all_ok", False)
    zero_frames = vox["n_zero"]
    ratio       = vox["ratio"]
    snr         = epi["snr"]
    sig_ratio   = float(epi["mean_inside"].mean() /
                        (epi["mean_outside"].mean() + 1e-9))

    def status(ok): return "PASS" if ok else "FAIL"

    print(f"  Dimension checks          : {status(all_dim_ok)}")
    print(f"  Volume ratio (tmpl/native): {ratio:.3f}  "
          f"{'PASS' if 0.5 < ratio < 2.0 else 'WARN  <- large discrepancy'}")
    print(f"  Zero frames in 4D mask    : {zero_frames}  "
          f"{'PASS' if zero_frames == 0 else 'FAIL  <- some timepoints empty'}")
    print(f"  Signal ratio (in/out mask): {sig_ratio:.2f}  "
          f"{'PASS' if sig_ratio > 1.0 else 'WARN  <- mask may be in wrong region'}")
    print(f"  Temporal SNR              : {snr:.1f}  "
          f"{'PASS' if snr > 10 else 'WARN  <- low tSNR'}")
    print("=" * 60)


def main():
    args     = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nValidating pipeline output for subject: {args.subject}")
    print(f"Figures will be saved to: {save_dir}\n")

    # -- Resolve all file paths -----------------------------------------------
    try:
        files, subject, session, run = resolve_paths(args)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    tag = f"sub-{subject}_ses-{session}_run-{run}"

    # -- Compute mean BOLD on the fly (not saved by pipeline) -----------------
    mean_bold_data, _ = load_mean_bold(files["bold_raw"])

    # -- Run all checks -------------------------------------------------------
    dims   = check_dimensions(files)
    vox    = check_voxel_counts(files)
    motion = check_motion_effect(files)
    epi    = check_masked_epi(files)

    # -- Generate all figures -------------------------------------------------
    print("\n" + "=" * 60)
    print("FIGURES")
    print("=" * 60)
    fig_template_vs_native(files, mean_bold_data, save_dir, tag)
    fig_motion_effect(motion, vox, save_dir, tag)
    fig_masked_epi_signal(epi, save_dir, tag)
    fig_mask_slices_native(files, mean_bold_data, save_dir, tag)
    fig_motion_mask_frames(files, save_dir, tag)

    # -- Final summary --------------------------------------------------------
    print_summary(dims, vox, epi, tag)
    print(f"\nAll figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
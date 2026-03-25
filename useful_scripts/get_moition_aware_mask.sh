#!/bin/bash
#SBATCH --job-name=motion_aware_mask
#SBATCH --output=logs/motion_aware_mask_%j.out
#SBATCH --error=logs/motion_aware_mask_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# =============================================================================
# create_motion_aware_mask.sh
#
# Creates a 4D motion-aware V1 mask by warping the static back-normalised
# V1 mask (in raw BOLD native space / average head position) to the exact
# head position at each of the 485 timepoints using per-volume mcflirt
# affine matrices.
#
# Pipeline
# --------
# For each volume t (0 to 484):
#   1. Load MAT_XXXX  (mcflirt matrix: raw vol t → reference/mean position)
#   2. Invert it      (mean position → raw vol t position)
#   3. Apply inverse to static 3D V1 mask → mask at timepoint t
#   4. Stack all 485 3D masks → 4D motion-aware mask
#
# Inputs
# ------
#   v1_binary_native.nii.gz          — static V1 mask in mean native space
#   bold_mcf.mat/MAT_0000 ... 0484   — per-volume mcflirt matrices
#   raw BOLD                          — for overlay visualisation
#
# Outputs
# -------
#   v1_mask_4d.nii.gz                — motion-aware 4D V1 mask
#   overlay frames as PNG            — selected timepoints showing mask on BOLD
# =============================================================================

# ── FSL setup  ────────────────────────────────────────────────────────────────
export FSLDIR=/opt/fsl/
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# ── Parameters  ───────────────────────────────────────────────────────────────
SUBJECT="IRN78"
SESSION="1"
RUN="001"

# ── Paths  ────────────────────────────────────────────────────────────────────
BIDS_DIR=/foundcog/bids

# Static 3D V1 mask in raw BOLD native space (back-normalised)
V1_STATIC=/home/muhammadfaizan/faizan_analysis/to_native_space_backnormalization/sub-${SUBJECT}/ses-${SESSION}/run-${RUN}/v1_binary_native.nii.gz

# Per-volume mcflirt matrices directory
MAT_DIR=/home/muhammadfaizan/faizan_analysis/mcflirt_mats/sub-${SUBJECT}/ses-${SESSION}/run-${RUN}/bold_mcf.mat

# Raw BOLD (reference grid + overlay target)
BOLD_RAW=${BIDS_DIR}/sub-${SUBJECT}/ses-${SESSION}/func/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold.nii.gz

# Output directory
OUT_DIR=/home/muhammadfaizan/faizan_analysis/motion_aware_mask/sub-${SUBJECT}/ses-${SESSION}/run-${RUN}
MASK_3D_DIR=${OUT_DIR}/masks_3d      # temporary per-volume masks
mkdir -p ${OUT_DIR}
mkdir -p ${MASK_3D_DIR}
mkdir -p logs

echo "══════════  Create Motion-Aware 4D V1 Mask  ══════════"
echo "Subject      : ${SUBJECT}"
echo "Session      : ${SESSION}"
echo "Run          : ${RUN}"
echo "Static mask  : ${V1_STATIC}"
echo "Mat dir      : ${MAT_DIR}"
echo "Raw BOLD     : ${BOLD_RAW}"
echo "Output dir   : ${OUT_DIR}"
echo ""

# ── Pre-flight checks  ────────────────────────────────────────────────────────
echo "── Pre-flight checks"

for f in "${V1_STATIC}" "${BOLD_RAW}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: not found: $f"
        exit 1
    fi
    echo "  OK: $(basename $f)"
done

if [ ! -d "${MAT_DIR}" ]; then
    echo "ERROR: mat directory not found: ${MAT_DIR}"
    echo "  Run run_mcflirt_with_mats.sh first"
    exit 1
fi

N_MATS=$(ls ${MAT_DIR}/MAT_* 2>/dev/null | wc -l)
N_VOLS=$(fslinfo ${BOLD_RAW} | grep "^dim4" | awk '{print $2}')

echo "  Mat files  : ${N_MATS}"
echo "  BOLD vols  : ${N_VOLS}"

if [ "${N_MATS}" -ne "${N_VOLS}" ]; then
    echo "ERROR: mat count (${N_MATS}) != BOLD volumes (${N_VOLS})"
    exit 1
fi

echo "  Static mask info:"
fslinfo ${V1_STATIC} | grep -E "^dim[1-3]|^pixdim[1-3]"
echo "  Raw BOLD info:"
fslinfo ${BOLD_RAW} | grep -E "^dim[1-4]|^pixdim[1-3]"

# ── Step 1: Extract single reference volume for flirt -ref  ──────────────────
echo ""
echo "── Step 1: Extract reference volume from raw BOLD"
REF_VOL=${OUT_DIR}/bold_ref_vol.nii.gz
fslroi ${BOLD_RAW} ${REF_VOL} 0 1
echo "  Reference volume: ${REF_VOL}"

# ── Step 2: Create per-volume masks  ─────────────────────────────────────────
echo ""
echo "── Step 2: Warp V1 mask to each timepoint (${N_VOLS} volumes)"
echo "   For each volume: invert mcflirt mat → apply to static mask"
echo "   This may take 10-20 minutes..."
echo ""

FAILED=0
for ((vol=0; vol<N_VOLS; vol++)); do

    # Zero-padded volume index
    VOL_PAD=$(printf '%04d' ${vol})
    MAT_FWD=${MAT_DIR}/MAT_${VOL_PAD}
    MAT_INV=${MASK_3D_DIR}/inv_${VOL_PAD}.mat
    MASK_OUT=${MASK_3D_DIR}/mask_${VOL_PAD}.nii.gz

    # Skip if already done (safe to re-run)
    if [ -f "${MASK_OUT}" ]; then
        continue
    fi

    # Invert the mcflirt matrix for this volume
    # mcflirt MAT: raw vol → mean (forward)
    # inverse:     mean → raw vol t (what we want)
    convert_xfm -inverse ${MAT_FWD} -omat ${MAT_INV}

    if [ $? -ne 0 ]; then
        echo "  ERROR inverting mat for vol ${vol}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Apply inverse to static V1 mask
    # -ref: raw BOLD ref vol (defines output voxel grid)
    # -interp: nearestneighbour preserves binary 0/1 values
    flirt \
        -in       ${V1_STATIC} \
        -ref      ${REF_VOL} \
        -applyxfm \
        -init     ${MAT_INV} \
        -interp   nearestneighbour \
        -out      ${MASK_OUT} \
        2>/dev/null

    if [ $? -ne 0 ]; then
        echo "  ERROR applying transform for vol ${vol}"
        FAILED=$((FAILED + 1))
    fi

    # Progress every 50 volumes
    if [ $(( (vol + 1) % 50 )) -eq 0 ]; then
        echo "  Progress: $((vol + 1)) / ${N_VOLS} volumes"
    fi

done

if [ ${FAILED} -gt 0 ]; then
    echo "WARNING: ${FAILED} volumes failed"
fi

N_DONE=$(ls ${MASK_3D_DIR}/mask_*.nii.gz 2>/dev/null | wc -l)
echo "  Completed: ${N_DONE} / ${N_VOLS} mask volumes"

# ── Step 3: Merge into 4D mask  ───────────────────────────────────────────────
echo ""
echo "── Step 3: Merge all 3D masks into 4D volume"

MASK_4D=${OUT_DIR}/v1_mask_4d.nii.gz

# Build sorted list of all mask files
MASK_LIST=$(ls ${MASK_3D_DIR}/mask_*.nii.gz | sort)

fslmerge -t ${MASK_4D} ${MASK_LIST}

if [ $? -ne 0 ]; then
    echo "ERROR: fslmerge failed"
    exit 1
fi

echo "  4D mask saved: ${MASK_4D}"
echo "  4D mask info:"
fslinfo ${MASK_4D} | grep -E "^dim[1-4]|^pixdim[1-4]"

# ── Step 4: Sanity checks  ────────────────────────────────────────────────────
echo ""
echo "── Step 4: Sanity checks"

# Voxel count should be similar at every timepoint (small variation is ok)
echo "  Voxel counts at selected timepoints:"
for t in 0 50 100 150 200 250 300 350 400 450 484; do
    if [ $t -lt ${N_VOLS} ]; then
        VOL_FILE=${MASK_3D_DIR}/mask_$(printf '%04d' ${t}).nii.gz
        COUNT=$(fslstats ${VOL_FILE} -V | awk '{print $1}')
        echo "    t=${t}: ${COUNT} voxels"
    fi
done

echo ""
echo "  Static mask voxels (reference):"
fslstats ${V1_STATIC} -V | awk '{print "    " $1 " voxels"}'

# ── Step 5: Generate overlay images  ─────────────────────────────────────────
echo ""
echo "── Step 5: Generate overlay images"

OVERLAY_SCRIPT=${OUT_DIR}/make_overlays.py
cat > ${OVERLAY_SCRIPT} << 'PYTHON'
import sys
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bold_path   = sys.argv[1]
mask4d_path = sys.argv[2]
out_dir     = sys.argv[3]
n_vols      = int(sys.argv[4])

bold = nib.load(bold_path).get_fdata(dtype=np.float32)
mask = nib.load(mask4d_path).get_fdata(dtype=np.float32)

# Find z-slices that contain V1 at timepoint 0
v1_z = np.where(mask[:, :, :, 0].any(axis=(0, 1)))[0]
if len(v1_z) == 0:
    v1_z = np.where(mask.any(axis=(0, 1, 3)))[0]

if len(v1_z) == 0:
    print("ERROR: mask is empty at all timepoints")
    sys.exit(1)

# Pick the z-slice with the most V1 voxels at t=0
v1_counts_per_z = [mask[:, :, z, 0].sum() for z in v1_z]
best_z = v1_z[np.argmax(v1_counts_per_z)]
print(f"Best z-slice for overlay: z={best_z}")

# Bold intensity range (non-zero)
bold_nz   = bold[bold != 0]
bold_vmin = np.percentile(bold_nz, 2)
bold_vmax = np.percentile(bold_nz, 98)

# ── Figure 1: Selected timepoints showing mask on BOLD  ──────────────────────
timepoints = np.linspace(0, n_vols - 1, 12, dtype=int)
fig, axes  = plt.subplots(3, 4, figsize=(16, 12), facecolor='black')
axes       = axes.flatten()

for i, t in enumerate(timepoints):
    ax   = axes[i]
    bold_slice = bold[:, :, best_z, t]
    mask_slice = mask[:, :, best_z, t]

    ax.imshow(bold_slice.T, origin='lower', cmap='gray',
              vmin=bold_vmin, vmax=bold_vmax)

    # Overlay mask in cyan where mask=1
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(masked.T, origin='lower', cmap='cool',
              vmin=0, vmax=1, alpha=0.6)

    n_v1 = int(mask[:, :, :, t].sum())
    ax.set_title(f't={t}  ({n_v1} V1 vox)', color='white', fontsize=8)
    ax.axis('off')

plt.suptitle(f'V1 mask (cyan) on raw BOLD (grey)  —  z={best_z}\n'
             f'sub-{os.path.basename(out_dir.split("/sub-")[1]).split("/")[0]}  '
             f'ses-{out_dir.split("/ses-")[1].split("/")[0]}  '
             f'run-{out_dir.split("/run-")[1].split("/")[0]}',
             color='white', fontsize=11)
plt.tight_layout()
out_png = os.path.join(out_dir, 'overlay_timepoints.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
plt.close()
print(f'Saved: {out_png}')

# ── Figure 2: Same timepoint, different z-slices  ────────────────────────────
t_mid  = n_vols // 2
z_plot = v1_z[np.linspace(0, len(v1_z)-1, 9, dtype=int)]

fig2, axes2 = plt.subplots(1, len(z_plot), figsize=(len(z_plot)*2.5, 3),
                            facecolor='black')

for i, z in enumerate(z_plot):
    ax = axes2[i]
    ax.imshow(bold[:, :, z, t_mid].T, origin='lower', cmap='gray',
              vmin=bold_vmin, vmax=bold_vmax)
    masked = np.ma.masked_where(mask[:, :, z, t_mid] == 0,
                                mask[:, :, z, t_mid])
    ax.imshow(masked.T, origin='lower', cmap='cool',
              vmin=0, vmax=1, alpha=0.6)
    n_v1 = int(mask[:, :, z, t_mid].sum())
    ax.set_title(f'z={z}\n({n_v1})', color='white', fontsize=7)
    ax.axis('off')

plt.suptitle(f'V1 mask across z-slices at t={t_mid}',
             color='white', fontsize=10)
plt.tight_layout()
out_png2 = os.path.join(out_dir, 'overlay_zslices.png')
plt.savefig(out_png2, dpi=150, bbox_inches='tight', facecolor='black')
plt.close()
print(f'Saved: {out_png2}')

# ── Figure 3: Voxel count over time  ─────────────────────────────────────────
vox_counts = [int(mask[:,:,:,t].sum()) for t in range(n_vols)]
fig3, ax3 = plt.subplots(figsize=(14, 3), facecolor='white')
ax3.plot(vox_counts, linewidth=0.8, color='steelblue')
ax3.axhline(np.mean(vox_counts), color='tomato', linestyle='--',
            label=f'mean={np.mean(vox_counts):.0f}')
ax3.fill_between(range(n_vols),
                 np.mean(vox_counts) - np.std(vox_counts),
                 np.mean(vox_counts) + np.std(vox_counts),
                 alpha=0.2, color='steelblue', label='±1 SD')
ax3.set_xlabel('Timepoint (volume)')
ax3.set_ylabel('V1 voxel count')
ax3.set_title('V1 voxel count over time — should be stable with small motion-driven variation')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
out_png3 = os.path.join(out_dir, 'voxel_count_over_time.png')
plt.savefig(out_png3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png3}')

print(f'\nV1 voxel count stats:')
print(f'  mean : {np.mean(vox_counts):.1f}')
print(f'  std  : {np.std(vox_counts):.1f}')
print(f'  min  : {min(vox_counts)}')
print(f'  max  : {max(vox_counts)}')
PYTHON

python3 ${OVERLAY_SCRIPT} \
    ${BOLD_RAW} \
    ${MASK_4D} \
    ${OUT_DIR} \
    ${N_VOLS}

# ── Cleanup temporary 3D masks to save disk space  ───────────────────────────
echo ""
echo "── Cleaning up temporary 3D mask files"
rm -f ${MASK_3D_DIR}/inv_*.mat
echo "  Removed inverse mat files (kept 3D masks for debugging)"
echo "  To remove 3D masks too: rm -rf ${MASK_3D_DIR}"

# ── Summary  ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════  Done  ══════════"
echo ""
echo "Outputs:"
echo "  4D mask     : ${MASK_4D}"
echo "  Overlay 1   : ${OUT_DIR}/overlay_timepoints.png  (mask on BOLD at 12 timepoints)"
echo "  Overlay 2   : ${OUT_DIR}/overlay_zslices.png     (mask across z-slices)"
echo "  Voxel plot  : ${OUT_DIR}/voxel_count_over_time.png"
echo ""
echo "What to check:"
echo "  1. V1 voxel counts should be similar across timepoints (±10%)"
echo "     Large swings = motion artefact or wrong matrices"
echo "  2. Cyan mask should sit in posterior occipital region on raw BOLD"
echo "  3. Mask should visibly shift slightly between timepoints"
echo "     if there was real head motion"
echo ""
echo "Download:"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/overlay_timepoints.png ~/Downloads/"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/overlay_zslices.png    ~/Downloads/"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/voxel_count_over_time.png ~/Downloads/"
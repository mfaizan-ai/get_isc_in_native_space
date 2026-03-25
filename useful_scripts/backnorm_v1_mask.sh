#!/bin/bash
#SBATCH --job-name=backnorm_v1
#SBATCH --output=logs/backnorm_v1_%j.out
#SBATCH --error=logs/backnorm_v1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# =============================================================================
# backnorm_v1_mask.sh
#
# Two-stage pipeline:
#   Stage 1 — Extract V1 from Julich atlas (grey value 91) in template space
#   Stage 2 — Back-normalize binary V1 mask to raw BOLD native EPI space
#             using the INVERSE of the combined run-to-template matrix
#
# Combined matrix does:
#   raw BOLD native → reference run → template space
#
# Its inverse does:
#   template space → reference run → raw BOLD native space
#
# The raw BOLD is used as the reference grid (defines native EPI voxel space).
#
# Sanity checks
# -------------
#   - V1 voxel count in template space
#   - V1 voxel count in native space (should be similar volume in mm3)
#   - fslinfo on both mask and native BOLD (dimensions should match)
#   - Mosaic overlay of V1 mask on mean native BOLD
#   - Mosaic overlay of V1 mask on template
#
# Outputs saved to OUT_DIR
# ------------------------
#   v1_atlas_label91.nii.gz         — raw atlas with only label 91
#   v1_binary_template.nii.gz       — binary V1 mask in template space
#   combined_matrix_inverse.mat     — inverted combined matrix
#   v1_binary_native.nii.gz         — binary V1 mask in native bold space
#   mean_bold_native.nii.gz         — mean of raw BOLD for reference
#   sanity_overlay_template.png     — V1 on template
#   sanity_overlay_native.png       — V1 on mean native BOLD
# =============================================================================

# ── FSL setup  ────────────────────────────────────────────────────────────────
export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# ── Parameters — edit before running  ────────────────────────────────────────
SUBJECT="IRN78"
SESSION="1"
RUN="001"
V1_LABEL=91        # grey value for V1 bilateral in Julich atlas

# ── Fixed paths  ──────────────────────────────────────────────────────────────
FD_DIR=/lustre/disk/home/shared/cusacklab/foundcog
BIDS_DIR=/lustre/disk/home/shared/cusacklab/foundcog/bids
WORKINGDIR=/lustre/disk/home/shared/cusacklab/foundcog/bids/workingdir

# Julich atlas in NIHPD 2mm template space
ATLAS=${BIDS_DIR}/derivatives/templates/rois/JulichBrainAtlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz

# NIHPD T2w template — for overlay sanity check
TEMPLATE=${BIDS_DIR}/derivatives/templates/mask/nihpd_asym_02-05_t2w_2mm.nii.gz

# Raw BOLD (native EPI space — defines the output voxel grid)
BOLD_RAW=${BIDS_DIR}/sub-${SUBJECT}/ses-${SESSION}/func/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold.nii.gz

# Combined matrix: run native EPI → reference run → template
# Using this matrix's INVERSE to go template → native EPI
MAT_FWD=${WORKINGDIR}/${SUBJECT}/derivatives/preproc/_subject_id_${SUBJECT}/_referencetype_standard/_run_${RUN}_session_${SESSION}_task_name_videos/combine_xfms_manual_selection/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold_mcf_corrected_mean_flirt_average_flirt.mat

# Output directory
OUT_DIR=/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_analysis/sub-${SUBJECT}/ses-${SESSION}/run-${RUN}
mkdir -p ${OUT_DIR}
mkdir -p logs

# Intermediate and output files
V1_LABEL_IMG=${OUT_DIR}/v1_atlas_label91.nii.gz
V1_BIN_TEMPLATE=${OUT_DIR}/v1_binary_template.nii.gz
MAT_INV=${OUT_DIR}/combined_matrix_inverse.mat
V1_BIN_NATIVE=${OUT_DIR}/v1_binary_native.nii.gz
MEAN_BOLD=${OUT_DIR}/mean_bold_native.nii.gz

echo "══════════  Back-Normalize V1 Mask to Native BOLD Space  ══════════"
echo "Subject      : ${SUBJECT}"
echo "Session      : ${SESSION}"
echo "Run          : ${RUN}"
echo "V1 label     : ${V1_LABEL}"
echo "Atlas        : ${ATLAS}"
echo "Combined mat : ${MAT_FWD}"
echo "Raw BOLD     : ${BOLD_RAW}"
echo "Output       : ${OUT_DIR}"
echo ""

# ── Pre-flight checks  ────────────────────────────────────────────────────────
echo "── Pre-flight: checking all input files"

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: not found: $1"
        exit 1
    fi
    echo "  OK: $(basename $1)"
}

check_file ${ATLAS}
check_file ${TEMPLATE}
check_file ${BOLD_RAW}
check_file ${MAT_FWD}

echo ""
echo "Atlas info:"
fslinfo ${ATLAS} | grep -E "^dim[1-4]|^pixdim[1-3]"

echo ""
echo "Raw BOLD info:"
fslinfo ${BOLD_RAW} | grep -E "^dim[1-4]|^pixdim[1-3]"

# ── Stage 1A: Extract V1 label from atlas  ────────────────────────────────────
echo ""
echo "══ Stage 1: Extract V1 from Julich atlas ══"
echo ""
echo "── Stage 1A: Isolate label ${V1_LABEL} (V1 bilateral)"

# Zero out everything except label 91
fslmaths ${ATLAS} -thr ${V1_LABEL} -uthr ${V1_LABEL} ${V1_LABEL_IMG}

if [ $? -ne 0 ]; then echo "ERROR: label extraction failed"; exit 1; fi

V1_RAW_VOXELS=$(fslstats ${V1_LABEL_IMG} -V | awk '{print $1}')
echo "  Non-zero voxels with label ${V1_LABEL}: ${V1_RAW_VOXELS}"

if [ "${V1_RAW_VOXELS}" -eq 0 ]; then
    echo "ERROR: No voxels found for label ${V1_LABEL}"
    echo "  Check available labels with: fslstats ${ATLAS} -R"
    fslstats ${ATLAS} -R
    exit 1
fi

# ── Stage 1B: Binarize to create binary V1 mask  ─────────────────────────────
echo ""
echo "── Stage 1B: Binarize V1 label → binary mask"

fslmaths ${V1_LABEL_IMG} -bin ${V1_BIN_TEMPLATE}

if [ $? -ne 0 ]; then echo "ERROR: binarization failed"; exit 1; fi

V1_TEMPLATE_VOXELS=$(fslstats ${V1_BIN_TEMPLATE} -V | awk '{print $1}')
V1_TEMPLATE_MM3=$(fslstats ${V1_BIN_TEMPLATE} -V | awk '{print $2}')
echo "  V1 binary mask in template space:"
echo "    Voxels : ${V1_TEMPLATE_VOXELS}"
echo "    Volume : ${V1_TEMPLATE_MM3} mm3"

fslinfo ${V1_BIN_TEMPLATE} | grep -E "^dim[1-4]|^pixdim[1-3]"

# ── Stage 2A: Invert combined matrix  ────────────────────────────────────────
echo ""
echo "══ Stage 2: Back-normalize to native BOLD space ══"
echo ""
echo "── Stage 2A: Invert combined matrix"
echo "  Forward matrix (run native → template):"
cat ${MAT_FWD}

convert_xfm -inverse ${MAT_FWD} -omat ${MAT_INV}

if [ $? -ne 0 ]; then echo "ERROR: matrix inversion failed"; exit 1; fi

echo ""
echo "  Inverse matrix (template → run native):"
cat ${MAT_INV}

# ── Stage 2B: Extract mean of raw BOLD as reference grid  ─────────────────────
echo ""
echo "── Stage 2B: Extract mean raw BOLD (defines native EPI grid)"
fslmaths ${BOLD_RAW} -Tmean ${MEAN_BOLD}
echo "  Mean BOLD info (this is the target native space):"
fslinfo ${MEAN_BOLD} | grep -E "^dim[1-4]|^pixdim[1-3]"

# ── Stage 2C: Apply inverse matrix to V1 binary mask  ─────────────────────────
echo ""
echo "── Stage 2C: Warp V1 mask from template → native BOLD space"
echo "  Input  : ${V1_BIN_TEMPLATE}  (template space)"
echo "  Ref    : ${MEAN_BOLD}         (native EPI space)"
echo "  Matrix : ${MAT_INV}           (template → native)"

flirt \
    -in       ${V1_BIN_TEMPLATE} \
    -ref      ${MEAN_BOLD} \
    -applyxfm \
    -init     ${MAT_INV} \
    -interp   nearestneighbour \
    -out      ${V1_BIN_NATIVE}

if [ $? -ne 0 ]; then echo "ERROR: back-normalization failed"; exit 1; fi

# ── Stage 3: Sanity checks  ───────────────────────────────────────────────────
echo ""
echo "══ Stage 3: Sanity checks ══"

echo ""
echo "── V1 mask dimensions in native space (should match raw BOLD):"
fslinfo ${V1_BIN_NATIVE} | grep -E "^dim[1-4]|^pixdim[1-3]"

echo ""
echo "── Raw BOLD dimensions (reference):"
fslinfo ${BOLD_RAW} | grep -E "^dim[1-4]|^pixdim[1-3]"

V1_NATIVE_VOXELS=$(fslstats ${V1_BIN_NATIVE} -V | awk '{print $1}')
V1_NATIVE_MM3=$(fslstats ${V1_BIN_NATIVE} -V | awk '{print $2}')

echo ""
echo "── V1 volume comparison:"
echo "  Template space : ${V1_TEMPLATE_VOXELS} voxels  (${V1_TEMPLATE_MM3} mm3)"
echo "  Native space   : ${V1_NATIVE_VOXELS} voxels   (${V1_NATIVE_MM3} mm3)"
echo "  Note: voxel counts differ (2mm vs 3mm), but mm3 should be similar"

if [ "${V1_NATIVE_VOXELS}" -eq 0 ]; then
    echo "ERROR: V1 mask in native space is empty — back-norm failed"
    echo "  Check that the inverse matrix is correct"
    exit 1
fi

# Volume ratio check — should be close to (3mm/2mm)^3 = 3.375
echo ""
RATIO=$(echo "scale=2; ${V1_TEMPLATE_MM3} / ${V1_NATIVE_MM3}" | bc 2>/dev/null || echo "bc not available")
echo "  Template/Native mm3 ratio: ${RATIO}  (expect ~1.0 if registration correct)"

# ── Stage 4: Visual sanity check mosaics  ────────────────────────────────────
echo ""
echo "══ Stage 4: Generate sanity check mosaics ══"

MOSAIC_SCRIPT=${OUT_DIR}/make_sanity_mosaic.py
cat > ${MOSAIC_SCRIPT} << 'PYTHON'
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

template_path    = sys.argv[1]
v1_template_path = sys.argv[2]
mean_bold_path   = sys.argv[3]
v1_native_path   = sys.argv[4]
out_template_png = sys.argv[5]
out_native_png   = sys.argv[6]

def make_overlay_mosaic(bg_path, mask_path, title, out_path, n_slices=9):
    bg   = nib.load(bg_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    if bg.ndim == 4:
        bg = bg.mean(axis=3)

    # Find z-slices that contain V1 mask voxels
    mask_z = np.where(mask.any(axis=(0, 1)))[0]
    if len(mask_z) == 0:
        print(f"WARNING: mask is empty in {mask_path}")
        return

    z_slices = mask_z[np.linspace(0, len(mask_z)-1, n_slices, dtype=int)]

    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices * 2.5, 3),
                              facecolor='black')

    bg_vmin = np.percentile(bg[bg != 0], 2)  if bg.any() else 0
    bg_vmax = np.percentile(bg[bg != 0], 98) if bg.any() else 1

    for i, z in enumerate(z_slices):
        ax = axes[i]

        # Background anatomy
        ax.imshow(bg[:, :, z].T, origin='lower', cmap='gray',
                  vmin=bg_vmin, vmax=bg_vmax)

        # V1 mask overlay in cyan
        v1_slice = np.ma.masked_where(mask[:, :, z] == 0, mask[:, :, z])
        ax.imshow(v1_slice.T, origin='lower', cmap='cool',
                  vmin=0, vmax=1, alpha=0.7)

        ax.set_title(f'z={z}', color='white', fontsize=8)
        ax.axis('off')

    plt.suptitle(title, color='white', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f'Saved: {out_path}')

    # Print quick stats
    v1_voxels = int((nib.load(mask_path).get_fdata() > 0).sum())
    print(f'  V1 voxels in {out_path}: {v1_voxels}')

# Overlay 1: V1 mask on T2w template
make_overlay_mosaic(
    template_path, v1_template_path,
    'V1 mask (cyan) on NIHPD T2w template — should sit in occipital cortex',
    out_template_png,
)

# Overlay 2: V1 mask on mean raw BOLD
make_overlay_mosaic(
    mean_bold_path, v1_native_path,
    'V1 mask (cyan) on mean raw BOLD — should sit in occipital cortex',
    out_native_png,
)
PYTHON

python3 ${MOSAIC_SCRIPT} \
    ${TEMPLATE} \
    ${V1_BIN_TEMPLATE} \
    ${MEAN_BOLD} \
    ${V1_BIN_NATIVE} \
    ${OUT_DIR}/sanity_overlay_template.png \
    ${OUT_DIR}/sanity_overlay_native.png

# ── Final summary  ────────────────────────────────────────────────────────────
echo ""
echo "══════════  Done  ══════════"
echo ""
echo "What to check:"
echo "  1. V1 voxel counts printed above — native mm3 should be close to template mm3"
echo "  2. sanity_overlay_template.png — V1 should sit in posterior occipital region"
echo "  3. sanity_overlay_native.png   — V1 should sit in same region on raw BOLD"
echo "  4. Native mask dims must match raw BOLD dims (64x64x36)"
echo ""
echo "Outputs:"
echo "  ${V1_BIN_TEMPLATE}  — V1 binary mask in template space"
echo "  ${V1_BIN_NATIVE}    — V1 binary mask in raw BOLD space"
echo "  ${MAT_INV}          — inverse combined matrix"
echo "  ${OUT_DIR}/sanity_overlay_template.png"
echo "  ${OUT_DIR}/sanity_overlay_native.png"
echo ""
echo "Next step — motion-aware 4D mask:"
echo "  Use the per-volume mcflirt affines (.par file) to warp"
echo "  v1_binary_native.nii.gz at each timepoint"
echo ""
echo "Download mosaics:"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/sanity_overlay_template.png ~/Downloads/"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/sanity_overlay_native.png   ~/Downloads/"
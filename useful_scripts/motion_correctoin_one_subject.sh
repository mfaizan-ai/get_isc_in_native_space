#!/bin/bash
#SBATCH --job-name=mcflirt_mats
#SBATCH --output=logs/mcflirt_mats_%j.out
#SBATCH --error=logs/mcflirt_mats_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# =============================================================================
# run_mcflirt_with_mats.sh
#
# Re-runs mcflirt on the raw BOLD using the SAME reference volume that was
# used in the original pipeline (read from reference.txt).
#
# Uses -mats flag so FSL saves one 4x4 affine matrix per volume:
#   out_mcf.mat/MAT_0000  ... MAT_0484
#
# These matrices are in correct FSL format with proper centre-of-rotation
# convention handled internally — safe to use directly with flirt -applyxfm.
#
# Why re-run rather than use .par file?
# --------------------------------------
# Converting .par rows to 4x4 matrices manually requires knowing FSL's
# exact centre-of-rotation convention. Getting this wrong produces a
# subtly incorrect mask alignment. Re-running mcflirt with -mats lets
# FSL handle the convention internally — zero risk of error.
#
# The output motion-corrected BOLD will be identical to the original
# bold_mcf.nii.gz (same reference volume, same input) — just with the
# per-volume .mat files now saved.
#
# Outputs
# -------
#   {OUT_DIR}/bold_mcf.nii.gz          — motion corrected BOLD (verify = original)
#   {OUT_DIR}/bold_mcf.mat/MAT_0000    — per-volume affine matrices
#   {OUT_DIR}/bold_mcf.mat/MAT_0001
#   ...
#   {OUT_DIR}/bold_mcf.mat/MAT_0484
#   {OUT_DIR}/bold_mcf.nii.par         — 6-parameter motion file (same as original)
#   sanity_check.png                   — motion plots + voxel comparison
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
WORKINGDIR=/foundcog/bids/workingdir

# Raw BOLD — input to mcflirt
BOLD_RAW=${BIDS_DIR}/sub-${SUBJECT}/ses-${SESSION}/func/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold.nii.gz

# Reference volume index saved by the original pipeline
REF_TXT=${WORKINGDIR}/${SUBJECT}/derivatives/preproc/bold_preproc/_subject_id_${SUBJECT}/_run_${RUN}_session_${SESSION}_task_name_videos/_referencetype_standard/save_ref_vol/reference.txt

# Original motion-corrected BOLD for comparison
BOLD_MCF_ORIG=${WORKINGDIR}/${SUBJECT}/derivatives/preproc/bold_preproc/_subject_id_${SUBJECT}/_run_${RUN}_session_${SESSION}_task_name_videos/_referencetype_standard/mcflirt/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold_mcf.nii.gz

# Original .par file for comparison
PAR_ORIG=${WORKINGDIR}/${SUBJECT}/derivatives/preproc/bold_preproc/_subject_id_${SUBJECT}/_run_${RUN}_session_${SESSION}_task_name_videos/_referencetype_standard/mcflirt/sub-${SUBJECT}_ses-${SESSION}_task-videos_dir-AP_run-${RUN}_bold_mcf.nii.par

# Output directory
OUT_DIR=/home/muhammadfaizan/faizan_analysis/mcflirt_mats/sub-${SUBJECT}/ses-${SESSION}/run-${RUN}
mkdir -p ${OUT_DIR}
mkdir -p logs

echo "══════════  mcflirt with -mats  ══════════"
echo "Subject  : ${SUBJECT}"
echo "Session  : ${SESSION}"
echo "Run      : ${RUN}"
echo ""

# ── Pre-flight checks  ────────────────────────────────────────────────────────
echo "── Pre-flight: checking inputs"

for f in "${BOLD_RAW}" "${REF_TXT}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: not found: $f"
        exit 1
    fi
    echo "  OK: $(basename $f)"
done

# Read reference volume index from file
REF_VOL=$(cat ${REF_TXT})
echo ""
echo "Reference volume index : ${REF_VOL}  (from reference.txt)"
echo "Raw BOLD info:"
fslinfo ${BOLD_RAW} | grep -E "^dim[1-4]|^pixdim[1-3]"

# ── Run mcflirt with -mats  ───────────────────────────────────────────────────
echo ""
echo "── Running mcflirt with -mats flag"
echo "   Input  : ${BOLD_RAW}"
echo "   Refvol : ${REF_VOL}"
echo "   Output : ${OUT_DIR}/bold_mcf"
echo ""
echo "   This will take ~10-15 minutes for 485 volumes..."

mcflirt \
    -in     ${BOLD_RAW} \
    -refvol ${REF_VOL} \
    -mats \
    -plots \
    -report \
    -out    ${OUT_DIR}/bold_mcf

if [ $? -ne 0 ]; then
    echo "ERROR: mcflirt failed"
    exit 1
fi

echo ""
echo "mcflirt complete."

# ── Verify mat files were created  ────────────────────────────────────────────
echo ""
echo "── Verifying per-volume matrix files"
MAT_DIR=${OUT_DIR}/bold_mcf.mat

if [ ! -d "${MAT_DIR}" ]; then
    echo "ERROR: mat directory not created: ${MAT_DIR}"
    exit 1
fi

N_MATS=$(ls ${MAT_DIR}/MAT_* 2>/dev/null | wc -l)
echo "  Mat files created : ${N_MATS}"

N_VOLS=$(fslinfo ${BOLD_RAW} | grep "^dim4" | awk '{print $2}')
echo "  BOLD volumes      : ${N_VOLS}"

if [ "${N_MATS}" -ne "${N_VOLS}" ]; then
    echo "WARNING: number of mat files (${N_MATS}) != number of volumes (${N_VOLS})"
else
    echo "  OK: mat count matches volume count"
fi

echo ""
echo "First matrix (MAT_0000) — should be close to identity if vol 0 ≈ ref:"
cat ${MAT_DIR}/MAT_0000

echo ""
echo "Matrix at reference volume (MAT_$(printf '%04d' ${REF_VOL})) — should be identity:"
cat ${MAT_DIR}/MAT_$(printf '%04d' ${REF_VOL})

# ── Sanity check: compare new mcf with original  ──────────────────────────────
echo ""
echo "── Sanity check: compare new motion-corrected BOLD with original"

if [ -f "${BOLD_MCF_ORIG}" ]; then
    # Compute difference between original and new mcf mean volumes
    MEAN_NEW=${OUT_DIR}/bold_mcf_mean.nii.gz
    MEAN_ORIG=${OUT_DIR}/bold_mcf_orig_mean.nii.gz
    DIFF=${OUT_DIR}/mcf_diff.nii.gz

    fslmaths ${OUT_DIR}/bold_mcf.nii.gz -Tmean ${MEAN_NEW}
    fslmaths ${BOLD_MCF_ORIG} -Tmean ${MEAN_ORIG}
    fslmaths ${MEAN_NEW} -sub ${MEAN_ORIG} ${DIFF}

    echo "  Difference between new and original mcf mean:"
    echo "  (should be near 0 0 — same input, same refvol)"
    fslstats ${DIFF} -R
    fslstats ${DIFF} -M
else
    echo "  Original mcf BOLD not found for comparison — skipping diff check"
    fslmaths ${OUT_DIR}/bold_mcf.nii.gz -Tmean ${OUT_DIR}/bold_mcf_mean.nii.gz
fi

# ── Compare .par files  ───────────────────────────────────────────────────────
echo ""
echo "── Compare new vs original .par file (first 5 rows)"
echo "  Original .par:"
head -5 ${PAR_ORIG} 2>/dev/null || echo "  (not found)"
echo ""
echo "  New .par:"
head -5 ${OUT_DIR}/bold_mcf.nii.par 2>/dev/null || echo "  (not found)"
echo ""
echo "  Note: values should be identical — same input, same reference volume"

# ── Generate motion plot  ─────────────────────────────────────────────────────
echo ""
echo "── Generating motion parameter plots"

PLOT_SCRIPT=${OUT_DIR}/plot_motion.py
cat > ${PLOT_SCRIPT} << 'PYTHON'
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

par_new  = sys.argv[1]
par_orig = sys.argv[2] if len(sys.argv) > 2 else None
out_png  = sys.argv[3]

new_par  = np.loadtxt(par_new)
n_vols   = new_par.shape[0]
t        = np.arange(n_vols)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='white')

# Rotations (convert radians to degrees for readability)
ax = axes[0]
labels = ['Rx', 'Ry', 'Rz']
for i, label in enumerate(labels):
    ax.plot(t, np.degrees(new_par[:, i]), label=f'new {label}', linewidth=0.8)
if par_orig:
    orig_par = np.loadtxt(par_orig)
    for i, label in enumerate(labels):
        ax.plot(t, np.degrees(orig_par[:, i]), '--', label=f'orig {label}',
                linewidth=0.8, alpha=0.5)
ax.set_ylabel('Rotation (degrees)')
ax.set_title('Rotations — new (solid) vs original (dashed)')
ax.legend(fontsize=7, ncol=6)
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)

# Translations
ax = axes[1]
labels = ['Tx', 'Ty', 'Tz']
for i, label in enumerate(labels):
    ax.plot(t, new_par[:, i+3], label=f'new {label}', linewidth=0.8)
if par_orig:
    for i, label in enumerate(labels):
        ax.plot(t, orig_par[:, i+3], '--', label=f'orig {label}',
                linewidth=0.8, alpha=0.5)
ax.set_ylabel('Translation (mm)')
ax.set_xlabel('Volume')
ax.set_title('Translations — new (solid) vs original (dashed)')
ax.legend(fontsize=7, ncol=6)
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Motion parameters ({n_vols} volumes)\n'
             'New and original should overlap perfectly',
             fontsize=11)
plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_png}')
PYTHON

python3 ${PLOT_SCRIPT} \
    ${OUT_DIR}/bold_mcf.nii.par \
    ${PAR_ORIG} \
    ${OUT_DIR}/motion_comparison.png

# ── Summary  ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════  Done  ══════════"
echo ""
echo "Per-volume matrices : ${MAT_DIR}/MAT_0000 ... MAT_$(printf '%04d' $((N_MATS-1)))"
echo "Motion plot         : ${OUT_DIR}/motion_comparison.png"
echo ""
echo "Key checks:"
echo "  1. MAT_$(printf '%04d' ${REF_VOL}) should be identity (reference volume = no motion)"
echo "  2. Motion plot new vs original should overlap perfectly"
echo "  3. Difference map between mcf means should be near 0"
echo ""
echo "Next step — motion-aware 4D V1 mask:"
echo "  Use MAT_0000...MAT_$(printf '%04d' $((N_MATS-1))) to warp the static V1 mask"
echo "  at each timepoint, then fslmerge -t into a 4D mask"
echo ""
echo "Download:"
echo "  scp \${USER}@\${HOSTNAME}:${OUT_DIR}/motion_comparison.png ~/Downloads/"
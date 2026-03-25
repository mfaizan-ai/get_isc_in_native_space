#!/bin/bash
#SBATCH --job-name=julich_v1_mask
#SBATCH --output=logs/julich_v1_mask_%j.out
#SBATCH --error=logs/julich_v1_mask_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# ==============================================================================
# Create a binary V1 mask from the Julich Brain Atlas (3.1)
#
# Atlas:   JulichBrainAtlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz
# Space:   NIH-PD 02-05 months template (infant MNI152, 2mm)
# V1 label: gray value 91 → hOc1 (primary visual cortex, V1), bilateral
#
# Output:  binary_mask_from_julichbrainatlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz
# ==============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────────
ATLAS_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/rois"
ATLAS_FILE="JulichBrainAtlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz"
ATLAS_PATH="${ATLAS_DIR}/${ATLAS_FILE}"

MASK_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/mask"
MASK_FILE="binary_mask_from_julichbrainatlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz"
MASK_PATH="${MASK_DIR}/${MASK_FILE}"
V1_LABEL=91   # hOc1 / V1 gray value in the Julich MPM atlas

# ── Environment ────────────────────────────────────────────────────────────────
# Load FSL — adjust module name to match your cluster's module system
module load fsl 2>/dev/null || true

# Fall back: if 'fslmaths' is not on PATH after module load, try common locations
if ! command -v fslmaths &>/dev/null; then
    for fsl_path in /usr/local/fsl /opt/fsl; do
        if [ -f "${fsl_path}/bin/fslmaths" ]; then
            export FSLDIR="${fsl_path}"
            export PATH="${FSLDIR}/bin:${PATH}"
            source "${FSLDIR}/etc/fslconf/fsl.sh"
            break
        fi
    done
fi

if ! command -v fslmaths &>/dev/null; then
    echo "ERROR: fslmaths not found. Please load FSL before running this script." >&2
    exit 1
fi

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "${ATLAS_PATH}" ]; then
    echo "ERROR: Atlas file not found: ${ATLAS_PATH}" >&2
    exit 1
fi

mkdir -p "${MASK_DIR}" || { echo "ERROR: Cannot create mask directory: ${MASK_DIR}" >&2; exit 1; }

# ── Create binary mask ─────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Atlas      : ${ATLAS_PATH}"
echo "V1 label   : ${V1_LABEL}"
echo "Output     : ${MASK_PATH}"
echo "Start time : $(date)"
echo "============================================================"

# Step 1 — threshold to isolate voxels == 91  (thr 90.5, uthr 91.5)
# Step 2 — binarise so output is 0/1
fslmaths "${ATLAS_PATH}" \
    -thr  90.5  \
    -uthr 91.5  \
    -bin          \
    "${MASK_PATH}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Binary V1 mask written to ${MASK_PATH}"
    echo "Voxel count (V1): $(fslstats "${MASK_PATH}" -V | awk '{print $1}')"
else
    echo "ERROR: fslmaths failed with exit code ${EXIT_CODE}" >&2
    exit ${EXIT_CODE}
fi

echo "End time: $(date)"
#!/bin/bash
#SBATCH --job-name=isc_bootstrap
#SBATCH --output=logs/isc_bootstrap_%j.out
#SBATCH --error=logs/isc_bootstrap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=compute

# ── Tuneable parameters ─────────────────────────────────────────────────────
# Change these to control the bootstrap without editing the Python script.
N_BOOT=1000
SEED=42
CI_ALPHA=0.05
BOOTSTRAP_PER_ROI=0          # set to 1 to also bootstrap per-ROI ISC (slow)

# Paths
SCRIPT_DIR="/lustre/disk/home/users/mfaizan/isc_analysis/isc_analysis_native_space/get_isc_in_native_space"
PIPELINE="${SCRIPT_DIR}/isc_schaefer_bootstrap.py"
CSV="${SCRIPT_DIR}/per_order_alignment/segments_mapping_each_sub_usable.csv"
OUT_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_analysis/isc_schaefer/across_brain_networks_analysis_and_boostrapping"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "======================================================================"
echo "Job        : ${SLURM_JOB_NAME}  (ID: ${SLURM_JOB_ID})"
echo "Node       : ${SLURMD_NODENAME}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Memory     : ${SLURM_MEM_PER_NODE} MB  ($(( SLURM_MEM_PER_NODE / 1024 )) GB)"
echo "Started    : $(date)"
echo "----------------------------------------------------------------------"
echo "N_BOOT     : ${N_BOOT}"
echo "SEED       : ${SEED}"
echo "CI_ALPHA   : ${CI_ALPHA}"
echo "PER_ROI    : $([ ${BOOTSTRAP_PER_ROI} -eq 1 ] && echo 'YES' || echo 'no')"
echo "SCRIPT     : ${PIPELINE}"
echo "CSV        : ${CSV}"
echo "OUT_DIR    : ${OUT_DIR}"
echo "======================================================================"

# ── NumPy / MKL threading ───────────────────────────────────────────────────
# Let numpy/einsum use all allocated CPUs for the vectorised ISC + bootstrap.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

# ── Conda ────────────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate isc_analysis

echo "--- Tool check ---"
which python && python --version
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import nibabel; print('nibabel', nibabel.__version__)"
python -c "import nilearn; print('nilearn', nilearn.__version__)"
echo "------------------"

# ── Build command ────────────────────────────────────────────────────────────
CMD=(
    python -u "${PIPELINE}"
    --csv       "${CSV}"
    --out_dir   "${OUT_DIR}"
    --bootstrap
    --n_boot    "${N_BOOT}"
    --seed      "${SEED}"
    --ci_alpha  "${CI_ALPHA}"
)

if [ ${BOOTSTRAP_PER_ROI} -eq 1 ]; then
    CMD+=(--bootstrap_per_roi)
fi

echo "Command    : ${CMD[*]}"
echo "======================================================================"

# ── Run ──────────────────────────────────────────────────────────────────────
"${CMD[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "Finished   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
echo "Outputs    : ${OUT_DIR}"
echo "Bootstrap  : ${OUT_DIR}/bootstrap/"
echo "======================================================================"
exit ${EXIT_CODE}
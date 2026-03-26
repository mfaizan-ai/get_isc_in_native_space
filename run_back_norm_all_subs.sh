#!/bin/bash
#SBATCH --job-name=backnorm_pipeline
#SBATCH --output=logs/backnorm_%j.out
#SBATCH --error=logs/backnorm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# run_backnorm.sh
#
# Submits backnorm_pipeline.py to SLURM for all 2-month subjects.
#
# Usage:
#   sbatch run_backnorm.sh
#
# To process specific subjects only:
#   sbatch run_backnorm.sh IRN78 IRC13
#
# The --cpus-per-task value is passed directly to --n_procs so nipype
# uses all allocated cores.  Adjust --cpus-per-task and --mem together
# if you change the core count (rough guide: ~4G RAM per core).
# =============================================================================

# -- Guard: create log directory before SLURM tries to write to it -----------
mkdir -p logs

# -- FSL setup ----------------------------------------------------------------
export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# -- Python / conda environment -----------------------------------------------
# Activate whichever environment has nipype + nibabel installed
source activate isc_analysis    # replace 'base' with your env name if different

# -- Pipeline script location -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/back_norm_all_sub.py"

# -- Number of parallel nipype processes = CPUs allocated by SLURM ------------
N_PROCS=${SLURM_CPUS_PER_TASK:-4}

# -- Optional: specific subjects passed as positional args to sbatch ----------
# e.g.  sbatch run_backnorm.sh IRN78 IRC13
# If none given, pipeline auto-discovers all 2-month subjects.
SUBJECTS=("$@")

echo "======================================================================"
echo "Job        : ${SLURM_JOB_NAME}  (ID: ${SLURM_JOB_ID})"
echo "Node       : ${SLURMD_NODENAME}"
echo "CPUs       : ${N_PROCS}"
echo "Memory     : ${SLURM_MEM_PER_NODE} MB"
echo "Started    : $(date)"
echo "Pipeline   : ${PIPELINE}"
if [ ${#SUBJECTS[@]} -gt 0 ]; then
    echo "Subjects   : ${SUBJECTS[*]}"
else
    echo "Subjects   : all 2-month subjects (auto-discovered)"
fi
echo "======================================================================"

# -- Run ----------------------------------------------------------------------
if [ ${#SUBJECTS[@]} -gt 0 ]; then
    python "${PIPELINE}" \
        --n_procs "${N_PROCS}" \
        --subjects "${SUBJECTS[@]}"
else
    python "${PIPELINE}" \
        --n_procs "${N_PROCS}"
fi

EXIT_CODE=$?

echo "======================================================================"
echo "Finished   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
echo "======================================================================"

exit ${EXIT_CODE}
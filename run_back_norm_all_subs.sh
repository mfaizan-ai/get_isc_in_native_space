#!/bin/bash
#SBATCH --job-name=backnorm_pipeline
#SBATCH --output=logs/backnorm_%j.out
#SBATCH --error=logs/backnorm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80          # 80 of 96 available cores -- leaves headroom for OS/other users
#SBATCH --mem=400G                  # ~5 GB/core; nodes have 2 TB so this is conservative
#SBATCH --time=16:00:00             # was 30h @ 16 cores; 80 cores should finish in ~3-5h
#SBATCH --partition=gpu             # only partition available on this cluster
# No --gres=gpu line intentionally -- FSL is CPU-only, leave H200s free for GPU users

# =============================================================================
# Usage:
#   sbatch run_backnorm.sh                        # process all subjects
#   sbatch run_backnorm.sh IRN78 IRC13            # process specific subjects
#
# Resource guide for this cluster (96-core / 2 TB nodes):
#   --cpus-per-task : keep at 80 (leaves 16 for OS + other users)
#   --mem           : ~5 GB per core is safe; 400G for 80 cores
#   --time          : 12h is generous; expect ~3-5h at 80 cores
# =============================================================================

# -- Guard: create log directory before SLURM tries to write to it -----------
mkdir -p logs

# -- Print node diagnostics at job start -------------------------------------
echo "======================================================================"
echo "Job        : ${SLURM_JOB_NAME}  (ID: ${SLURM_JOB_ID})"
echo "Node       : ${SLURMD_NODENAME}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Memory     : ${SLURM_MEM_PER_NODE} MB  ($(( SLURM_MEM_PER_NODE / 1024 )) GB)"
echo "Started    : $(date)"
echo "======================================================================"

# -- Verify /tmp is available and has space (NVMe local scratch on these nodes)
echo "--- /tmp (node-local NVMe scratch) ---"
df -h /tmp
echo "TMPDIR     : ${TMPDIR:-not set, will default to /tmp}"
echo "--------------------------------------"

# -- FSL setup ----------------------------------------------------------------
export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# -- Python / conda environment -----------------------------------------------
source ~/.bashrc
conda activate isc_analysis

# -- Verify key tools are available before spending cluster time --------------
echo "--- Tool check ---"
which python  && python  --version
which flirt   && flirt   -version 2>&1 | head -1
which fslmaths
echo "------------------"

# -- Pipeline script ----------------------------------------------------------
PIPELINE="back_norm_all_sub.py"   # the new HPC-optimized version

# n_procs and memory_gb are now auto-detected inside the Python script from
# $SLURM_CPUS_PER_TASK and $SLURM_MEM_PER_NODE -- no need to pass them
# explicitly.  You can still override if needed:
#   python -u "${PIPELINE}" --n_procs 40 --memory_gb 200

# -- Optional subject list from positional args -------------------------------
SUBJECTS=("$@")
# SUBJECTS=("IRN78")

if [ ${#SUBJECTS[@]} -gt 0 ]; then
    echo "Subjects   : ${SUBJECTS[*]}"
else
    echo "Subjects   : all 2-month subjects (auto-discovered)"
fi
echo "======================================================================"

# -- Run the pipeline ---------------------------------------------------------
if [ ${#SUBJECTS[@]} -gt 0 ]; then
    python -u "${PIPELINE}" --subjects "${SUBJECTS[@]}"
else
    python -u "${PIPELINE}"
fi

EXIT_CODE=$?

# -- Final summary ------------------------------------------------------------
echo "======================================================================"
echo "Finished   : $(date)"
echo "Exit code  : ${EXIT_CODE}"

# Show how much scratch was used / confirm it was cleaned up
echo "--- /tmp after pipeline ---"
df -h /tmp
echo "---------------------------"
echo "======================================================================"

exit ${EXIT_CODE}
#!/bin/bash
#SBATCH --job-name=backnorm_schaefer_atlas 
#SBATCH --output=logs/backnorm_schaefer_testcase%j.out
#SBATCH --error=logs/backnorm_schaefer_testcase%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80         
#SBATCH --mem=400G                 
#SBATCH --time=16:00:00            
#SBATCH --partition=gpu             



# -- Set to 1 to apply FSL topup distortion correction before masking 
#    Requires AP + PA fieldmaps in each subject's fmap/ directory.
#    Set to 0 to skip topup (original behaviour).
USE_TOPUP=1

# logs saved here
mkdir -p logs

#  Print node diagnostics at job start
echo "======================================================================"
echo "Job        : ${SLURM_JOB_NAME}  (ID: ${SLURM_JOB_ID})"
echo "Node       : ${SLURMD_NODENAME}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Memory     : ${SLURM_MEM_PER_NODE} MB  ($(( SLURM_MEM_PER_NODE / 1024 )) GB)"
echo "Started    : $(date)"
echo "Topup      : $([ $USE_TOPUP -eq 1 ] && echo 'ENABLED' || echo 'disabled')"
echo "======================================================================"

# -- Verify /tmp is available and has space (NVMe local scratch on these nodes)
echo "--- /tmp (node-local NVMe scratch) ---"
df -h /tmp
echo "TMPDIR     : ${TMPDIR:-not set, will default to /tmp}"
echo "--------------------------------------"

# FSL setup 
export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Python / conda environment 
source ~/.bashrc
conda activate isc_analysis

#  Verify key tools are available 
echo "--- Tool check ---"
which python  && python  --version
which flirt   && flirt   -version 2>&1 | head -1
which fslmaths
which topup                          # needed when USE_TOPUP=1
which applytopup
echo "------------------"

# Pipeline script
PIPELINE="back_norm_all_sub.py"

# n_procs and memory_gb are auto-detected from $SLURM_CPUS_PER_TASK and
# $SLURM_MEM_PER_NODE inside the Python script -- no need to pass explicitly.
# Override if needed:
#   python -u "${PIPELINE}" --n_procs 40 --memory_gb 200
SUBJECTS=("$@")
# SUBJECTS=("IRN78")           # uncomment to hardcode specific subjects for sanity check before running pipelien whchic take long time

#  Build the python command 
CMD=(python -u "${PIPELINE}")

if [ ${#SUBJECTS[@]} -gt 0 ]; then
    CMD+=(--subjects "${SUBJECTS[@]}")
    echo "Subjects   : ${SUBJECTS[*]}"
else
    echo "Subjects   : all subjects (auto-discovered)"
fi

if [ $USE_TOPUP -eq 1 ]; then
    CMD+=(--use_topup)
fi

echo "Command    : ${CMD[*]}"
echo "======================================================================"

#  Run the pipeline
"${CMD[@]}"
EXIT_CODE=$?

# Final summary 
echo "======================================================================"
echo "Finished   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
echo "Outputs    : derivatives/faizan_analysis/schaefer_backnorm/test_case"
echo "--- /tmp after pipeline ---"
df -h /tmp
echo "---------------------------"
echo "======================================================================"
exit ${EXIT_CODE}
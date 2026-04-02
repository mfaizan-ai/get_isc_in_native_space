#!/usr/bin/env python3

import argparse
import re
import atexit
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from typing import List, Optional, Tuple

import nipype.pipeline.engine as pe
from nipype import Node, MapNode, Workflow, config as nipype_config
from nipype.interfaces import fsl
from nipype.interfaces.io import DataSink


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backnorm")


# =============================================================================
# Default paths  
# =============================================================================
_BIDS_DIR = "/lustre/disk/home/shared/cusacklab/foundcog/bids"
DEFAULTS = dict(
    bids_dir         = _BIDS_DIR,
    workingdir       = f"{_BIDS_DIR}/workingdir",
    mcflirt_mats_dir = f"{_BIDS_DIR}/derivatives/motion_affines/mcflirt_mats_output",
    template_schaefer_atlast = (
        f"{_BIDS_DIR}/derivatives/templates/rois/"
        "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    ),
    template_mask    = (
        f"{_BIDS_DIR}/derivatives/templates/mask/"
        "binary_mask_from_julichbrainatlas_3.1_207areas_MPM_MNI152"
        "_space-nihpd-02-05_2mm.nii.gz"
    ),
    output_dir      = f"{_BIDS_DIR}/derivatives/faizan_analysis",
    # nipype_work_dir intentionally left None -- resolved at runtime from scratch
    n_procs         = None,   # None = auto-detect from SLURM
    memory_gb       = None,   # None = auto-detect from SLURM
    batch_size      = 200,    # max runs per meta-workflow to keep graph lean
)

EXCLUDE_SUBS = ["ICC89", "ICC103", "ICN50", "ICC57"]

# =============================================================================
# Scratch / TMPDIR management
# =============================================================================
_SCRATCH_DIR: Optional[Path] = None   # set once in setup_scratch()

def setup_scratch() -> Path:
    """
    Create a job-specific scratch directory on node-local storage.

    Priority order:
      1.  /scratch/$USER/$SLURM_JOB_ID   (preferred on most HPC clusters)
      2.  $TMPDIR                         (set by some schedulers)
      3.  /tmp/$USER_backnorm_<pid>        (fallback)

    The directory is registered for automatic cleanup via atexit and SIGTERM.
    """
    global _SCRATCH_DIR

    user    = os.environ.get("USER", "user")
    job_id  = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    scratch_root = Path(os.environ.get("TMPDIR", f"/scratch/{user}"))
    scratch = scratch_root / job_id / "backnorm"

    scratch.mkdir(parents=True, exist_ok=True)
    _SCRATCH_DIR = scratch
    log.info(f"Scratch directory : {scratch}")

    # Register cleanup
    atexit.register(_cleanup_scratch)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    return scratch


def _cleanup_scratch():
    if _SCRATCH_DIR and _SCRATCH_DIR.exists():
        log.info(f"Cleaning up scratch: {_SCRATCH_DIR}")
        try:
            shutil.rmtree(_SCRATCH_DIR)
        except Exception as e:
            log.warning(f"Could not remove scratch dir: {e}")


def _sigterm_handler(signum, frame):
    log.info("Received SIGTERM -- cleaning up and exiting.")
    _cleanup_scratch()
    sys.exit(0)


# detect auto slurm resoures 
def detect_slurm_resources() -> Tuple[int, float]:
    """
    Return (n_procs, memory_gb) from SLURM environment variables.
    Falls back to conservative defaults if not running under SLURM.
    """
    # ---- CPUs ----------------------------------------------------------------
    cpus_str = (
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_NPROCS")
        or os.environ.get("SLURM_NTASKS")
    )
    try:
        n_procs = int(cpus_str)
    except (TypeError, ValueError):
        n_procs = 4
        log.warning(
            "Could not detect SLURM CPU count from environment; "
            f"defaulting to {n_procs}. Pass --n_procs to override."
        )
    else:
        log.info(f"SLURM CPUs detected  : {n_procs}")

    # ---- Memory --------------------------------------------------------------
    # SLURM_MEM_PER_NODE is in MB; SLURM_MEM_PER_CPU is also in MB
    mem_node = os.environ.get("SLURM_MEM_PER_NODE")
    mem_cpu  = os.environ.get("SLURM_MEM_PER_CPU")
    try:
        if mem_node:
            memory_gb = int(mem_node) / 1024.0
        elif mem_cpu:
            memory_gb = int(mem_cpu) * n_procs / 1024.0
        else:
            raise ValueError("no SLURM mem env var")
    except (TypeError, ValueError):
        memory_gb = 8.0
        log.warning(
            "Could not detect SLURM memory from environment; "
            f"defaulting to {memory_gb} GB. Pass --memory_gb to override."
        )
    else:
        log.info(f"SLURM memory detected: {memory_gb:.1f} GB")

    return n_procs, memory_gb

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bids_dir",         default=DEFAULTS["bids_dir"])
    p.add_argument("--workingdir",       default=DEFAULTS["workingdir"],
                   help="Directory containing per-subject normalization matrices")
    p.add_argument("--mcflirt_mats_dir", default=DEFAULTS["mcflirt_mats_dir"])
    p.add_argument("--template_mask",    default=DEFAULTS["template_mask"])
    p.add_argument("--output_dir",       default=DEFAULTS["output_dir"])
    p.add_argument("--nipype_work_dir",  default=None,
                   help="Override Nipype scratch dir (default: node-local scratch)")
    p.add_argument("--subjects",         nargs="+", default=None)
    p.add_argument("--n_procs",          type=int, default=None,
                   help="Parallel processes (default: $SLURM_CPUS_PER_TASK)")
    p.add_argument("--memory_gb",        type=float, default=None,
                   help="Total memory in GB (default: from SLURM env)")
    p.add_argument("--batch_size",       type=int, default=DEFAULTS["batch_size"],
                   help="Max runs per meta-workflow graph (default: 200)")
    return p.parse_args()


def bold_path(bids_dir, subject, session, run) -> Path:
    return (
        Path(bids_dir) / f"sub-{subject}" / f"ses-{session}" / "func"
        / f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}_bold.nii.gz"
    )

def norm_mat_path(workingdir, subject, session, run) -> Path:
    return (
        Path(workingdir) / subject / "derivatives" / "preproc"
        / f"_subject_id_{subject}" / "_referencetype_standard"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "combine_xfms_manual_selection"
        / (
            f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}"
            "_bold_mcf_corrected_mean_flirt_average_flirt.mat"
        )
    )

def mcflirt_mats_path(mcflirt_mats_dir, subject, session, run) -> Path:
    return (
        Path(mcflirt_mats_dir)
        / f"_subject_id_{subject}"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "mats"
    )

def output_func_dir(output_dir, subject, session) -> Path:
    return Path(output_dir) / f"sub-{subject}" / f"ses-{session}" / "func"

def output_prefix(subject, session, run) -> str:
    return f"sub-{subject}_ses-{session}_task-videos_run-{run}"


def find_subjects(bids_dir: str) -> List[str]:
    return sorted(
        d.name[4:]
        for d in Path(bids_dir).iterdir()
        if d.is_dir()
        and d.name.startswith("sub-")
        and not d.name.endswith("A")
        and d.name[4:] not in EXCLUDE_SUBS
    )

def find_sessions_and_runs(bids_dir: str, subject: str) -> List[Tuple[str, str]]:
    pairs = []
    sub_dir = Path(bids_dir) / f"sub-{subject}"
    for ses_dir in sorted(sub_dir.iterdir()):
        if not (ses_dir.is_dir() and ses_dir.name.startswith("ses-")):
            continue
        session  = ses_dir.name[4:]
        func_dir = ses_dir / "func"
        if not func_dir.exists():
            continue
        pattern = f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-*_bold.nii.gz"
        for f in sorted(func_dir.glob(pattern)):
            run = next(
                (part[4:] for part in f.stem.split("_") if part.startswith("run-")),
                None,
            )
            if run:
                pairs.append((session, run))
    return pairs



_G = "\033[32m"; _R = "\033[31m"; _Y = "\033[33m"; _E = "\033[0m"
def _ok(m):   return f"{_G}  OK   {_E}{m}"
def _miss(m): return f"{_R}  MISS {_E}{m}"
def _warn(m): return f"{_Y}  WARN {_E}{m}"


def check_all_paths(subjects, args) -> Tuple[List[Tuple[str,str,str]], List[str]]:
    import nibabel as nib

    ready, skipped = [], []
    total_issues   = 0

    print("\n" + "=" * 70)
    print("PATH VALIDATION")
    print("=" * 70)

    tmask = Path(args.template_mask)
    print(_ok(f"template_mask   {tmask}") if tmask.exists()
          else _miss(f"template_mask   {tmask}"))
    if not tmask.exists():
        total_issues += 1
    print()

    for subject in subjects:
        pairs = find_sessions_and_runs(args.bids_dir, subject)
        print("-" * 70)
        if not pairs:
            print(_miss(f"sub-{subject}  -- no videos-task BOLD found in BIDS"))
            skipped.append(f"sub-{subject} (no BOLD found)")
            total_issues += 1
            continue

        for session, run in pairs:
            tag    = f"sub-{subject}  ses-{session}  run-{run}"
            issues = 0
            print(tag)

            b = bold_path(args.bids_dir, subject, session, run)
            print(_ok(f"  BOLD         {b}") if b.exists()
                  else _miss(f"  BOLD         {b}"))
            if not b.exists(): issues += 1

            m = norm_mat_path(args.workingdir, subject, session, run)
            print(_ok(f"  norm_mat     {m}") if m.exists()
                  else _miss(f"  norm_mat     {m}"))
            if not m.exists(): issues += 1

            md        = mcflirt_mats_path(args.mcflirt_mats_dir, subject, session, run)
            mat_files = sorted(md.glob("MAT_*")) if md.is_dir() else []
            if md.is_dir() and mat_files:
                print(_ok(f"  mcflirt_dir  {md}  ({len(mat_files)} MAT files)"))
                if b.exists():
                    try:
                        n_vols = nib.load(str(b)).shape[3]
                        if len(mat_files) == n_vols:
                            print(_ok(f"  vol_count    {len(mat_files)} MAT == {n_vols} BOLD volumes"))
                        else:
                            print(_warn(f"  vol_count    {len(mat_files)} MAT != {n_vols} BOLD volumes  <- mismatch!"))
                            issues += 1
                    except Exception as e:
                        print(_warn(f"  vol_count    could not read BOLD: {e}"))
            elif md.is_dir():
                print(_warn(f"  mcflirt_dir  {md}  (exists but NO MAT files)"))
                issues += 1
            else:
                print(_miss(f"  mcflirt_dir  {md}"))
                issues += 1

            if issues == 0:
                print(f"  -> {_G}ALL OK{_E}")
                ready.append((subject, session, run))
            else:
                print(f"  -> {_R}{issues} issue(s) -- will be skipped{_E}")
                skipped.append(tag.strip())
                total_issues += issues
            print()

    print("=" * 70)
    print(f"  Ready to run : {len(ready)}")
    print(f"  Skipped      : {len(skipped)}")
    if total_issues == 0:
        print(f"{_G}All paths OK.{_E}")
    else:
        print(f"{_R}{total_issues} issue(s) found. Fix MISS entries before re-running.{_E}")
    print("=" * 70 + "\n")
    return ready, skipped


class ProgressTracker:
    """
    Progress callback for Nipype's MultiProc plugin.

    NOTE: Nipype deepcopies nodes (and anything reachable from plugin_args)
    before sending them to worker processes.  threading.Lock is not picklable
    and will cause a TypeError crash.  This class deliberately avoids any
    threading primitives so it can be safely deepcopied/pickled.
    Minor counter inaccuracy under heavy parallelism is acceptable for logging.
    """
    def __init__(self, total_nodes: int):
        self._done   = 0
        self._failed = 0
        self._total  = total_nodes
        self._start  = time.time()

    def __call__(self, node, status):
        if status == "end":
            self._done += 1
        elif status == "exception":
            self._failed += 1

        elapsed = time.time() - self._start
        rate    = self._done / elapsed if elapsed > 0 else 0
        eta_s   = (self._total - self._done) / rate if rate > 0 else float("inf")
        eta_str = f"{eta_s/60:.0f} min" if eta_s < 3600 else f"{eta_s/3600:.1f} h"

        log.info(
            f"[{status.upper():9s}] {node.name}  |  "
            f"{self._done}/{self._total} done  "
            f"({self._failed} failed)  |  "
            f"elapsed {elapsed/60:.1f} min  ETA ~{eta_str}"
        )


def configure_nipype(scratch: Path):
    """
    Enable hash-based caching so re-runs skip already-completed nodes.
    Write crash files to scratch (fast local disk, human-readable).
    """
    nipype_config.set("execution", "stop_on_first_crash",  "false")
    nipype_config.set("execution", "crashdump_dir",        str(scratch / "crashdumps"))
    nipype_config.set("execution", "hash_method",          "content")
    # Keep all node working dirs after completion so the hash cache persists
    # between re-runs (nodes whose outputs haven't changed will be skipped).
    nipype_config.set("execution", "remove_unnecessary_outputs", "false")
    nipype_config.set("logging",   "workflow_level",       "INFO")
    nipype_config.set("logging",   "interface_level",      "WARNING")

    (scratch / "crashdumps").mkdir(parents=True, exist_ok=True)


def build_run_workflow(
    subject, session, run,
    bold, fwd_mat, mats,
    template_mask, output_dir,
    nipype_work_dir,
) -> Workflow:
    """
    One Nipype workflow per subject/session/run.

    All processing nodes run entirely in nipype_work_dir (node-local scratch).
    A DataSink node at the end explicitly copies the three final outputs to
    Lustre output_dir with correct BIDS filenames:
      - {pfx}_space-native_mask.nii.gz              (3D back-normalised mask)
      - {pfx}_space-native_desc-mask4d.nii.gz       (4D motion-aware mask)
      - {pfx}_space-native_desc-maskedbold.nii.gz   (final masked BOLD)

    Using DataSink (rather than hardcoded out_file paths) is the correct
    Nipype pattern when the work directory and output directory are on
    different filesystems (scratch vs Lustre).  Hardcoded out_file paths
    on a different filesystem cause silent failures where the file lands
    in the scratch work dir and is then deleted at job end.
    """

    out_dir = output_func_dir(output_dir, subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = output_prefix(subject, session, run)

    wf = Workflow(
        name     = f"run_sub{subject}_ses{session}_run{run}",
        base_dir = nipype_work_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1A -- mean BOLD  (native EPI reference grid)
    # ------------------------------------------------------------------
    mean_bold = Node(
        fsl.ImageMaths(in_file=bold, op_string="-Tmean"),
        name="mean_bold",
    )

    # ------------------------------------------------------------------
    # Stage 1B -- invert normalization matrix  (template -> native EPI)
    # ------------------------------------------------------------------
    invert_mat = Node(
        fsl.ConvertXFM(in_file=fwd_mat, invert_xfm=True),
        name="invert_mat",
    )

    # ------------------------------------------------------------------
    # Stage 1C -- back-normalize mask to native space (3D)
    # No out_file here -- DataSink handles the final copy to Lustre
    # ------------------------------------------------------------------
    backnorm_3d = Node(
        fsl.FLIRT(
            in_file   = template_mask,
            interp    = "nearestneighbour",
            apply_xfm = True,
        ),
        name="backnorm_3d",
    )
    wf.connect([
        (mean_bold,  backnorm_3d, [("out_file", "reference")]),
        (invert_mat, backnorm_3d, [("out_file", "in_matrix_file")]),
    ])

    # ------------------------------------------------------------------
    # Stage 2A -- invert each per-volume MCFlirt affine  (MapNode)
    # ------------------------------------------------------------------
    invert_motion_mats = MapNode(
        fsl.ConvertXFM(invert_xfm=True),
        iterfield = ["in_file"],
        name      = "invert_motion_mats",
    )
    invert_motion_mats.inputs.in_file = mats

    # ------------------------------------------------------------------
    # Stage 2B -- extract individual volumes from BOLD  (MapNode)
    # ------------------------------------------------------------------
    extract_vols = MapNode(
        fsl.ExtractROI(in_file=bold, t_size=1),
        iterfield = ["t_min"],
        name      = "extract_vols",
    )
    extract_vols.inputs.t_min = list(range(len(mats)))

    # ------------------------------------------------------------------
    # Stage 2C -- apply inv(MAT_t) to 3D mask, ref = vol_t  (MapNode)
    # All ~400 intermediate 3D masks live entirely in scratch
    # ------------------------------------------------------------------
    per_vol_mask = MapNode(
        fsl.FLIRT(interp="nearestneighbour", apply_xfm=True),
        iterfield = ["in_matrix_file", "reference"],
        name      = "per_vol_mask",
    )
    wf.connect([
        (invert_motion_mats, per_vol_mask, [("out_file", "in_matrix_file")]),
        (extract_vols,       per_vol_mask, [("roi_file", "reference")]),
        (backnorm_3d,        per_vol_mask, [("out_file", "in_file")]),
    ])

    # ------------------------------------------------------------------
    # Stage 2D -- merge per-volume masks into 4D motion-aware mask
    # ------------------------------------------------------------------
    merge_4d = Node(fsl.Merge(dimension="t"), name="merge_4d")
    wf.connect([(per_vol_mask, merge_4d, [("out_file", "in_files")])])

    # ------------------------------------------------------------------
    # Stage 3 -- apply 4D mask to raw BOLD (element-wise multiply)
    # No out_file -- DataSink handles the copy to Lustre
    # ------------------------------------------------------------------
    apply_mask = Node(
        fsl.ApplyMask(in_file=bold),
        name="apply_mask",
    )
    wf.connect([(merge_4d, apply_mask, [("merged_file", "mask_file")])])

    # ------------------------------------------------------------------
    # Stage 4 -- Rename nodes + DataSink
    #
    # FSL and Nipype auto-generate filenames from their inputs, producing
    # long unpredictable names.  We insert a lightweight Function node
    # that copies each output to an explicit BIDS filename before handing
    # it to DataSink.  DataSink then just moves the already-named file
    # into the correct subdirectory -- no substitution magic needed.
    #
    # Output layout under func/:
    #   mask_native/   {pfx}_space-native_mask.nii.gz
    #   mask_4d/       {pfx}_space-native_desc-mask4d.nii.gz
    #   masked_bold/   {pfx}_space-native_desc-maskedbold.nii.gz
    # ------------------------------------------------------------------
    from nipype.interfaces.utility import Function

    def _rename(in_file: str, out_name: str) -> str:
        """Copy in_file to a sibling path with out_name as filename."""
        import shutil, os
        out_path = os.path.join(os.path.dirname(in_file), out_name)
        shutil.copy(in_file, out_path)
        return out_path

    rename_mask3d = Node(
        Function(input_names=["in_file", "out_name"],
                 output_names=["out_file"],
                 function=_rename),
        name="rename_mask3d",
    )
    rename_mask3d.inputs.out_name = f"{pfx}_space-native_mask.nii.gz"

    rename_mask4d = Node(
        Function(input_names=["in_file", "out_name"],
                 output_names=["out_file"],
                 function=_rename),
        name="rename_mask4d",
    )
    rename_mask4d.inputs.out_name = f"{pfx}_space-native_desc-mask4d.nii.gz"

    rename_bold = Node(
        Function(input_names=["in_file", "out_name"],
                 output_names=["out_file"],
                 function=_rename),
        name="rename_bold",
    )
    rename_bold.inputs.out_name = f"{pfx}_space-native_desc-maskedbold.nii.gz"

    wf.connect([
        (backnorm_3d, rename_mask3d, [("out_file",    "in_file")]),
        (merge_4d,    rename_mask4d, [("merged_file", "in_file")]),
        (apply_mask,  rename_bold,   [("out_file",    "in_file")]),
    ])

    # DataSink: dot notation "subdir.@tag" routes file into subdir/
    # No substitutions needed -- files are already correctly named.
    datasink = Node(DataSink(), name="datasink")
    datasink.inputs.base_directory = str(out_dir)
    datasink.inputs.container      = ""
    datasink.inputs.substitutions  = []   # no renaming needed

    wf.connect([
        (rename_mask3d, datasink, [("out_file", "mask_native.@out")]),
        (rename_mask4d, datasink, [("out_file", "mask_4d.@out")]),
        (rename_bold,   datasink, [("out_file", "masked_bold.@out")]),
    ])

    return wf


def run_batch(
    batch:           List[Tuple[str, str, str]],
    batch_idx:       int,
    n_batches:       int,
    args:            argparse.Namespace,
    scratch:         Path,
    n_procs:         int,
    memory_gb:       float,
):
    """Build and execute a meta-workflow for one batch of runs."""
    nipype_work_dir = str(scratch / "nipype_work")

    log.info(
        f"Batch {batch_idx}/{n_batches}: "
        f"{len(batch)} run(s), {n_procs} procs, {memory_gb:.1f} GB RAM"
    )

    meta_wf = Workflow(
        name     = f"backnorm_batch{batch_idx:03d}",
        base_dir = nipype_work_dir,
    )

    total_nodes = 0
    for subject, session, run in batch:
        mats = sorted(
            str(m) for m in mcflirt_mats_path(
                args.mcflirt_mats_dir, subject, session, run
            ).glob("MAT_*")
        )
        run_wf = build_run_workflow(
            subject         = subject,
            session         = session,
            run             = run,
            bold            = str(bold_path(args.bids_dir, subject, session, run)),
            fwd_mat         = str(norm_mat_path(args.workingdir, subject, session, run)),
            mats            = mats,
            template_mask   = args.template_mask,
            output_dir      = args.output_dir,
            nipype_work_dir = nipype_work_dir,
        )
        meta_wf.add_nodes([run_wf])
        # Each run has roughly: 3 Stage-1 nodes + 3 MapNodes (each ~n_vols items) + 2 more
        total_nodes += 8 + 3 * len(mats)

    tracker  = ProgressTracker(total_nodes)
    mem_per_proc = memory_gb / n_procs

    log.info(f"Estimated total nodes in graph: ~{total_nodes:,}")
    log.info(f"Memory per process: {mem_per_proc:.2f} GB")

    meta_wf.run(
        plugin      = "MultiProc",
        plugin_args = {
            "n_procs":      n_procs,
            "memory_gb":    memory_gb,
            "status_callback": tracker,
        },
    )



def main():
    args = parse_args()

    # -- Scratch setup --------------------------------------------------------
    scratch = setup_scratch()

    # -- Override nipype_work_dir with scratch unless user specified one -------
    if args.nipype_work_dir is None:
        args.nipype_work_dir = str(scratch / "nipype_work")
    log.info(f"Nipype work dir   : {args.nipype_work_dir}")

    # -- Resolve n_procs and memory_gb ----------------------------------------
    slurm_procs, slurm_mem = detect_slurm_resources()
    n_procs   = args.n_procs   if args.n_procs   is not None else slurm_procs
    memory_gb = args.memory_gb if args.memory_gb is not None else slurm_mem
    log.info(f"Using n_procs={n_procs}, memory_gb={memory_gb:.1f}")

    # -- Configure nipype globally --------------------------------------------
    configure_nipype(scratch)

    # -- Discover subjects ----------------------------------------------------
    if args.subjects:
        excluded = [s for s in args.subjects if s in EXCLUDE_SUBS]
        subjects = [s for s in args.subjects if s not in EXCLUDE_SUBS]
        if excluded:
            log.warning(f"Excluded (in EXCLUDE_SUBS): {excluded}")
    else:
        subjects = find_subjects(args.bids_dir)

    log.info(f"Always-excluded: {EXCLUDE_SUBS}")
    log.info(f"Subjects to process: {len(subjects)}")

    # -- Validate all inputs --------------------------------------------------
    ready, skipped = check_all_paths(subjects, args)
    if not ready:
        log.error("No valid subject/session/run combinations found. Exiting.")
        sys.exit(1)

    # -- Split into batches to keep the graph manageable ----------------------
    batch_size = args.batch_size
    batches    = [ready[i:i+batch_size] for i in range(0, len(ready), batch_size)]
    n_batches  = len(batches)
    log.info(
        f"Processing {len(ready)} run(s) in {n_batches} batch(es) "
        f"of up to {batch_size} run(s) each."
    )

    t0 = time.time()
    for idx, batch in enumerate(batches, start=1):
        run_batch(
            batch      = batch,
            batch_idx  = idx,
            n_batches  = n_batches,
            args       = args,
            scratch    = scratch,
            n_procs    = n_procs,
            memory_gb  = memory_gb,
        )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Ran      : {len(ready)} run(s) in {elapsed/3600:.2f} h")
    log.info(f"  Skipped  : {len(skipped)}")
    for s in skipped:
        log.info(f"    - {s}")
    log.info(f"  Outputs  : {args.output_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
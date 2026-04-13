#!/usr/bin/env python3
import argparse
import json
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
from nipype.interfaces.utility import Function


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
    bids_dir              = _BIDS_DIR,
    workingdir            = f"{_BIDS_DIR}/workingdir",
    mcflirt_mats_dir      = f"{_BIDS_DIR}/derivatives/motion_affines/mcflirt_mats_output",
    template_schaefer_atlas = (
        f"{_BIDS_DIR}/derivatives/templates/rois/"
        "Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
    ),
    # Outputs go into faizan_analysis/schaefer_backnorm/ to keep Schaefer
    # results separate from any other back-normalisation runs
    output_dir            = f"{_BIDS_DIR}/derivatives/faizan_analysis/schaefer_backnorm",
    # nipype_work_dir intentionally left None -- resolved at runtime from scratch
    n_procs         = None,   # None = auto-detect from SLURM
    memory_gb       = None,   # None = auto-detect from SLURM
    batch_size      = 200,    # max runs per meta-workflow to keep graph lean
    # -------------------------------------------------------------------------
    # [TOPUP] defaults -- only used when --use_topup is passed
    # -------------------------------------------------------------------------
    use_topup       = True,  # off by default; existing runs unaffected
)

EXCLUDE_SUBS = ["ICC89", "ICC103", "ICN50", "ICC57"]

# =============================================================================
# Scratch / TMPDIR management
# =============================================================================
_SCRATCH_DIR: Optional[Path] = None   # set once in setup_scratch()

def setup_scratch() -> Path:
    global _SCRATCH_DIR

    user    = os.environ.get("USER", "user")
    job_id  = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    scratch_root = Path(os.environ.get("TMPDIR", f"/scratch/{user}"))
    scratch = scratch_root / job_id / "backnorm"

    scratch.mkdir(parents=True, exist_ok=True)
    _SCRATCH_DIR = scratch
    log.info(f"Scratch directory : {scratch}")

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


def detect_slurm_resources() -> Tuple[int, float]:
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
    p.add_argument("--template_schaefer_atlas", default=DEFAULTS["template_schaefer_atlas"],
                   help="Schaefer atlas in template space (back-normalised to native)")
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
    # -------------------------------------------------------------------------
    # [TOPUP] new flag -- all other args unchanged
    # -------------------------------------------------------------------------
    p.add_argument("--use_topup",        action="store_true", default=False,
                   help=(
                       "Apply FSL topup distortion correction to the BOLD "
                       "before back-normalizing the mask. Requires AP and PA "
                       "fieldmap files in the BIDS fmap/ directory. "
                       "Without this flag the pipeline behaves exactly as before."
                   ))
    return p.parse_args()


# =============================================================================
# Path helpers  (originals unchanged)
# =============================================================================
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


# =============================================================================
# [TOPUP] path helpers -- new, mirrors main pipeline style
# =============================================================================
def fmap_paths(bids_dir: str, subject: str, session: str) -> Tuple[List[Path], List[Path]]:
    """
    Return (ap_fmaps, pa_fmaps) for a given subject/session.

    AP and PA are allowed to have different run numbers (e.g. IRN78 has
    dir-AP_run-001 and dir-PA_run-002) -- the wildcard run-* handles this.
    Files are sorted so ordering is deterministic across runs.
    """
    fmap_dir = Path(bids_dir) / f"sub-{subject}" / f"ses-{session}" / "fmap"
    if not fmap_dir.exists():
        log.warning(f"  [{subject} ses-{session}] No fmap/ directory found at {fmap_dir}")
        return [], []

    ap = sorted(fmap_dir.glob(
        f"sub-{subject}_ses-{session}_dir-AP_run-*_epi.nii.gz"))
    pa = sorted(fmap_dir.glob(
        f"sub-{subject}_ses-{session}_dir-PA_run-*_epi.nii.gz"))

    # Log what was found so it's easy to verify correct files are picked
    for f in ap:
        log.info(f"  [{subject} ses-{session}] Found AP fmap: {f.name}")
    for f in pa:
        log.info(f"  [{subject} ses-{session}] Found PA fmap: {f.name}")

    if not ap:
        log.warning(f"  [{subject} ses-{session}] No AP fieldmaps found in {fmap_dir}")
    if not pa:
        log.warning(f"  [{subject} ses-{session}] No PA fieldmaps found in {fmap_dir}")

    return ap, pa


def _read_fmap_params(fmap_files: List[Path]) -> Tuple[List[str], List[float]]:
    """
    Read PhaseEncodingDirection and TotalReadoutTime from JSON sidecars.
    Returns (encoding_directions, readout_times) parallel to fmap_files.

    FSL topup requires TotalReadoutTime (seconds, range 0.01-0.2 s) --
    NOT EffectiveEchoSpacing. EffectiveEchoSpacing is the dwell time per
    k-space line (~0.5 ms); TotalReadoutTime is that multiplied by
    (PE_steps - 1) and is what topup's --datain file expects.

    Priority:
      1. TotalReadoutTime  from the JSON sidecar  (preferred, always present
         in well-formed BIDS datasets)
      2. Computed fallback: EffectiveEchoSpacing x (n_PE_lines - 1)
         using the image second dimension as an approximation of PE steps.
    """
    import nibabel as nib

    remap = {"i": "x", "i-": "x-", "j": "y", "j-": "y-", "k": "z", "k-": "z-"}
    enc_dirs      = []
    readout_times = []

    for fmap in fmap_files:
        stem = fmap.with_suffix("") if fmap.suffix == ".gz" else fmap
        stem = stem.with_suffix("")
        json_path = stem.with_suffix(".json")
        with open(json_path) as f:
            meta = json.load(f)

        enc_dirs.append(remap[meta["PhaseEncodingDirection"]])

        if "TotalReadoutTime" in meta:
            rt = float(meta["TotalReadoutTime"])
            log.info(f"    {fmap.name}: TotalReadoutTime = {rt:.5f} s  (from JSON)")
        elif "EffectiveEchoSpacing" in meta:
            pe_steps = int(nib.load(str(fmap)).shape[1])
            rt = float(meta["EffectiveEchoSpacing"]) * (pe_steps - 1)
            log.warning(
                f"    {fmap.name}: TotalReadoutTime missing -- "
                f"computed as EffectiveEchoSpacing ({meta['EffectiveEchoSpacing']}) "
                f"x (PE_steps-1) ({pe_steps-1}) = {rt:.5f} s"
            )
        else:
            raise KeyError(
                f"Neither TotalReadoutTime nor EffectiveEchoSpacing found "
                f"in {json_path}. Cannot run topup."
            )

        if not (0.01 <= rt <= 0.2):
            log.warning(
                f"    {fmap.name}: readout time {rt:.5f} s is outside FSL's "
                f"expected range (0.01-0.2 s). Check JSON sidecar."
            )

        readout_times.append(rt)

    return enc_dirs, readout_times


# =============================================================================
# Discovery helpers  (unchanged)
# =============================================================================
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


# =============================================================================
# Validation
# =============================================================================
_G = "\033[32m"; _R = "\033[31m"; _Y = "\033[33m"; _E = "\033[0m"
def _ok(m):   return f"{_G}  OK   {_E}{m}"
def _miss(m): return f"{_R}  MISS {_E}{m}"
def _warn(m): return f"{_Y}  WARN {_E}{m}"


def check_all_paths(subjects, args) -> Tuple[List[Tuple[str,str,str]], List[str]]:
    """
    Validates all required input files.
    When --use_topup is active, also checks for AP and PA fieldmap files.
    Original checks are completely unchanged.
    """
    import nibabel as nib

    ready, skipped = [], []
    total_issues   = 0

    print("\n" + "=" * 70)
    print("PATH VALIDATION" + ("  [topup mode]" if args.use_topup else ""))
    print("=" * 70)

    tmask = Path(args.template_schaefer_atlas)
    print(_ok(f"template_schaefer_atlas  {tmask}") if tmask.exists()
          else _miss(f"template_schaefer_atlas  {tmask}"))
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

            # -----------------------------------------------------------------
            # [TOPUP] fieldmap validation -- only runs when --use_topup is set
            # -----------------------------------------------------------------
            if args.use_topup:
                ap_fmaps, pa_fmaps = fmap_paths(args.bids_dir, subject, session)
                if ap_fmaps:
                    print(_ok(f"  fmap AP      {ap_fmaps[0].parent}  ({len(ap_fmaps)} file(s))"))
                else:
                    print(_miss(f"  fmap AP      no AP fieldmaps found for ses-{session}"))
                    issues += 1
                if pa_fmaps:
                    print(_ok(f"  fmap PA      {pa_fmaps[0].parent}  ({len(pa_fmaps)} file(s))"))
                else:
                    print(_miss(f"  fmap PA      no PA fieldmaps found for ses-{session}"))
                    issues += 1

                # Check JSON sidecars exist alongside each fmap
                all_fmaps = ap_fmaps + pa_fmaps
                missing_json = [
                    f for f in all_fmaps
                    if not f.with_suffix("").with_suffix(".json").exists()
                    and not f.parent.joinpath(
                        f.name.replace(".nii.gz", ".json")).exists()
                ]
                if missing_json:
                    for mj in missing_json:
                        print(_miss(f"  fmap JSON    missing sidecar for {mj.name}"))
                    issues += len(missing_json)
                else:
                    print(_ok(f"  fmap JSON    all sidecars present"))
            # -----------------------------------------------------------------

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


# =============================================================================
# Progress tracker  (unchanged)
# =============================================================================
class ProgressTracker:
    """
    Progress callback for Nipype's MultiProc plugin.
    Deliberately avoids threading primitives so it is safely picklable.
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


# =============================================================================
# Nipype configuration  (unchanged)
# =============================================================================
def configure_nipype(scratch: Path):
    nipype_config.set("execution", "stop_on_first_crash",  "false")
    nipype_config.set("execution", "crashdump_dir",        str(scratch / "crashdumps"))
    nipype_config.set("execution", "hash_method",          "content")
    nipype_config.set("execution", "remove_unnecessary_outputs", "false")
    nipype_config.set("logging",   "workflow_level",       "INFO")
    nipype_config.set("logging",   "interface_level",      "WARNING")

    (scratch / "crashdumps").mkdir(parents=True, exist_ok=True)


# =============================================================================
# [TOPUP] helper nodes -- mirrors the main pipeline's select_fmaps approach
# =============================================================================
def _build_topup_nodes(bold: str, bids_dir: str, subject: str, session: str) -> Tuple:
    """
    Build the topup-related nodes and return them ready to be wired into
    the per-run workflow.

    Chain:  hmc_fmaps -> mean_fmaps -> merge_fmaps -> topup -> applytopup

    AP fieldmaps are always listed first in all_fmaps so they occupy row 1
    of the topup encoding file.  The BOLD was acquired with AP phase-encoding,
    so applytopup receives index=[1] (1-based FSL convention) to tell it
    which encoding row corresponds to the input BOLD.

    AP and PA are allowed to have different run numbers (e.g. IRN78 uses
    dir-AP_run-001 / dir-PA_run-002) -- fmap_paths() handles this with
    wildcards and logs exactly which files were found.
    """
    ap_fmaps, pa_fmaps = fmap_paths(bids_dir, subject, session)

    if not ap_fmaps or not pa_fmaps:
        raise FileNotFoundError(
            f"[{subject} ses-{session}] Cannot build topup nodes: "
            f"AP fmaps found={len(ap_fmaps)}, PA fmaps found={len(pa_fmaps)}. "
            f"Check fmap/ directory."
        )

    # AP first, PA second -- this ordering sets which row index maps to AP
    all_fmaps = [str(f) for f in (ap_fmaps + pa_fmaps)]
    enc_dirs, readout_times = _read_fmap_params(ap_fmaps + pa_fmaps)

    log.info(
        f"  [{subject} ses-{session}] Topup encoding directions : {enc_dirs}"
    )
    log.info(
        f"  [{subject} ses-{session}] Topup readout times       : {readout_times}"
    )

    # ------------------------------------------------------------------
    # Step 1 -- motion-correct each fieldmap (AP and PA separately)
    # Mirrors hmc_fmaps MapNode in the main pipeline
    # ------------------------------------------------------------------
    hmc_fmaps = MapNode(
        fsl.MCFLIRT(),
        iterfield=["in_file"],
        name="topup_hmc_fmaps",
    )
    hmc_fmaps.inputs.in_file = all_fmaps

    # ------------------------------------------------------------------
    # Step 2 -- take mean of each motion-corrected fieldmap
    # Produces one clean 3D volume per fieldmap file
    # ------------------------------------------------------------------
    mean_fmaps = MapNode(
        fsl.maths.MeanImage(),
        iterfield=["in_file"],
        name="topup_mean_fmaps",
    )

    # ------------------------------------------------------------------
    # Step 3 -- concatenate mean AP + mean PA into single 4D file
    # Order: AP (index 1) then PA (index 2) in the encoding file
    # ------------------------------------------------------------------
    merge_fmaps = Node(
        fsl.Merge(dimension="t"),
        name="topup_merge_fmaps",
    )

    # ------------------------------------------------------------------
    # Step 4 -- estimate the field inhomogeneity map
    # Parameters read directly from JSON sidecars (same as main pipeline)
    # ------------------------------------------------------------------
    topup_node = Node(
        fsl.TOPUP(
            encoding_direction=enc_dirs,
            readout_times=readout_times,
        ),
        name="topup_estimate",
    )

    # ------------------------------------------------------------------
    # Step 5 -- apply the field correction to the raw BOLD
    #
    # index=[1]: the BOLD was acquired with AP phase-encoding, which is
    #   row 1 of the encoding file (AP fieldmaps were listed first above).
    #   This is a required parameter -- without it FSL does not know which
    #   correction to apply and will raise an error.
    #
    # method='jac': Jacobian modulation -- corrects both geometry and the
    #   signal intensity changes caused by voxel compression/stretching.
    #   Mirrors the main pipeline's fixed choice.
    # ------------------------------------------------------------------
    applytopup_node = Node(
        fsl.ApplyTOPUP(
            in_files=[bold],
            method="jac",
            in_index=[1],       # BOLD = AP = row 1 of encoding file
        ),
        name="topup_apply",
    )

    return hmc_fmaps, mean_fmaps, merge_fmaps, topup_node, applytopup_node


# =============================================================================
# Core per-run workflow  
# One small new parameter: use_topup (False by default, original unchanged)
# =============================================================================
def build_run_workflow(
    subject, session, run,
    bold, fwd_mat, mats,
    template_schaefer_atlas, output_dir,
    nipype_work_dir,
    bids_dir,           # [TOPUP] needed to locate fieldmaps; harmless when unused
    use_topup = False,  # [TOPUP] off by default -- original behaviour preserved
) -> Workflow:
    """
    One Nipype workflow per subject/session/run.

    When use_topup=False (default): identical to the original pipeline.
    When use_topup=True: an FSL topup + applytopup stage is inserted before
    the mean_bold node.  The undistorted BOLD then flows into every downstream
    step that previously received the raw BOLD (mean_bold and apply_mask).
    Everything else -- backnorm_3d, 4D motion-aware mask, DataSink layout --
    is completely unchanged.
    """

    out_dir = output_func_dir(output_dir, subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = output_prefix(subject, session, run)

    wf = Workflow(
        name     = f"run_sub{subject}_ses{session}_run{run}",
        base_dir = nipype_work_dir,
    )

    # ------------------------------------------------------------------
    # [TOPUP]  Optional distortion correction  
    # Builds the topup chain and returns the node whose output replaces
    # the raw bold path everywhere downstream.
    # When use_topup=False this block is skipped entirely.
    # ------------------------------------------------------------------
    if use_topup:
        log.info(f"  [{subject} ses-{session} run-{run}] Building topup nodes")
        (hmc_fmaps,
         mean_fmaps,
         merge_fmaps,
         topup_node,
         applytopup_node) = _build_topup_nodes(bold, bids_dir, subject, session)

        # Wire the topup chain
        wf.connect([
            (hmc_fmaps,      mean_fmaps,      [("out_file",    "in_file")]),
            (mean_fmaps,     merge_fmaps,     [("out_file",    "in_files")]),
            (merge_fmaps,    topup_node,      [("merged_file", "in_file")]),
            (topup_node,     applytopup_node, [
                ("out_fieldcoef", "in_topup_fieldcoef"),
                ("out_movpar",    "in_topup_movpar"),
                ("out_enc_file",  "encoding_file"),
            ]),
        ])
        # Convenience alias: downstream nodes connect from this output
        bold_source      = applytopup_node
        bold_source_port = "out_corrected"
    else:
        bold_source      = None   # signals: use hardcoded bold path below
        bold_source_port = None

    # ------------------------------------------------------------------
    # Stage 1A -- mean BOLD  (native EPI reference grid)
    # If topup is on:  mean of the corrected BOLD
    # If topup is off: mean of the raw BOLD  (original behaviour)
    # ------------------------------------------------------------------
    mean_bold = Node(
        fsl.ImageMaths(
            # only used as the static input when topup is OFF
            **({"in_file": bold} if not use_topup else {}),
            op_string="-Tmean",
        ),
        name="mean_bold",
    )
    if use_topup:
        wf.connect([(bold_source, mean_bold, [(bold_source_port, "in_file")])])

    # ------------------------------------------------------------------
    # Stage 1B -- invert normalization matrix  (template -> native EPI)
    # ------------------------------------------------------------------
    invert_mat = Node(
        fsl.ConvertXFM(in_file=fwd_mat, invert_xfm=True),
        name="invert_mat",
    )

    # ------------------------------------------------------------------
    # Stage 1C -- back-normalize mask to native space (3D)
    # ------------------------------------------------------------------
    backnorm_3d = Node(
        fsl.FLIRT(
            in_file   = template_schaefer_atlas,
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
    # When topup is on: extract from the corrected BOLD via a connection
    # When topup is off: use the hardcoded raw bold path  (original)
    # ------------------------------------------------------------------
    extract_vols = MapNode(
        fsl.ExtractROI(
            **({"in_file": bold} if not use_topup else {}),
            t_size=1,
        ),
        iterfield = ["t_min"],
        name      = "extract_vols",
    )
    extract_vols.inputs.t_min = list(range(len(mats)))
    if use_topup:
        wf.connect([(bold_source, extract_vols, [(bold_source_port, "in_file")])])

    # ------------------------------------------------------------------
    # Stage 2C -- apply inv(MAT_t) to 3D mask, ref = vol_t  (MapNode)
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
    # Stage 3 -- apply 4D mask to BOLD (element-wise multiply)
    # When topup is on: apply to corrected BOLD
    # When topup is off: apply to raw BOLD  (original behaviour)
    # ------------------------------------------------------------------
    apply_mask = Node(
        fsl.ApplyMask(
            **({"in_file": bold} if not use_topup else {}),
        ),
        name="apply_mask",
    )
    if use_topup:
        wf.connect([(bold_source, apply_mask, [(bold_source_port, "in_file")])])
    wf.connect([(merge_4d, apply_mask, [("merged_file", "mask_file")])])

    # ------------------------------------------------------------------
    # Stage 4 -- Rename + DataSink  (unchanged)
    # ------------------------------------------------------------------
    def _rename(in_file: str, out_name: str) -> str:
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

    datasink = Node(DataSink(), name="datasink")
    datasink.inputs.base_directory = str(out_dir)
    datasink.inputs.container      = ""
    datasink.inputs.substitutions  = []

    wf.connect([
        (rename_mask3d, datasink, [("out_file", "mask_native.@out")]),
        (rename_mask4d, datasink, [("out_file", "mask_4d.@out")]),
        (rename_bold,   datasink, [("out_file", "masked_bold.@out")]),
    ])

    return wf


# =============================================================================
# Batch runner  (one new kwarg: use_topup, passed straight through)
# =============================================================================

def run_batch(
    batch:           List[Tuple[str, str, str]],
    batch_idx:       int,
    n_batches:       int,
    args:            argparse.Namespace,
    scratch:         Path,
    n_procs:         int,
    memory_gb:       float,
):
    nipype_work_dir = str(scratch / "nipype_work")

    log.info(
        f"Batch {batch_idx}/{n_batches}: "
        f"{len(batch)} run(s), {n_procs} procs, {memory_gb:.1f} GB RAM"
        + (f"  [topup ON]" if args.use_topup else "")
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
            template_schaefer_atlas = args.template_schaefer_atlas,
            output_dir      = args.output_dir,
            nipype_work_dir = nipype_work_dir,
            bids_dir        = args.bids_dir,   # [TOPUP] needed for fmap lookup
            use_topup       = args.use_topup,  # [TOPUP] passed through
        )
        meta_wf.add_nodes([run_wf])
        # [TOPUP] adds ~5 extra nodes per run (hmc, mean, merge, topup, apply)
        extra = 5 if args.use_topup else 0
        total_nodes += 8 + 3 * len(mats) + extra

    tracker  = ProgressTracker(total_nodes)
    mem_per_proc = memory_gb / n_procs

    log.info(f"Estimated total nodes in graph: ~{total_nodes:,}")
    log.info(f"Memory per process: {mem_per_proc:.2f} GB")

    meta_wf.run(
        plugin      = "MultiProc",
        plugin_args = {
            "n_procs":         n_procs,
            "memory_gb":       memory_gb,
            "status_callback": tracker,
        },
    )


# =============================================================================
# Main  (unchanged except passing use_topup through)
# =============================================================================
def main():
    args = parse_args()

    scratch = setup_scratch()
    # scratch = Path(args.scratch_dir) if args.scratch_dir else setup_scratch()
    # scratch = Path("scratch/") # for debugging

    if args.nipype_work_dir is None:
        args.nipype_work_dir = str(scratch / "nipype_work")
    log.info(f"Nipype work dir   : {args.nipype_work_dir}")

    slurm_procs, slurm_mem = detect_slurm_resources()
    n_procs   = args.n_procs   if args.n_procs   is not None else slurm_procs
    memory_gb = args.memory_gb if args.memory_gb is not None else slurm_mem
    log.info(f"Using n_procs={n_procs}, memory_gb={memory_gb:.1f}")

    # [TOPUP] log the mode so it's obvious in SLURM stdout
    if args.use_topup:
        log.info("Topup distortion correction: ENABLED")
    else:
        log.info("Topup distortion correction: disabled (pass --use_topup to enable)")

    configure_nipype(scratch)

    if args.subjects:
        excluded = [s for s in args.subjects if s in EXCLUDE_SUBS]
        subjects = [s for s in args.subjects if s not in EXCLUDE_SUBS]
        if excluded:
            log.warning(f"Excluded (in EXCLUDE_SUBS): {excluded}")
    else:
        subjects = find_subjects(args.bids_dir)

    log.info(f"Always-excluded: {EXCLUDE_SUBS}")
    log.info(f"Subjects to process: {len(subjects)}")

    ready, skipped = check_all_paths(subjects, args)
    if not ready:
        log.error("No valid subject/session/run combinations found. Exiting.")
        sys.exit(1)

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
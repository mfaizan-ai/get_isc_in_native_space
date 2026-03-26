#!/usr/bin/env python3
"""
backnorm_pipeline.py
====================
Nipype pipeline -- three stages:

  Stage 1 -- Back-normalize a binary mask from template space to native BOLD
             space (3D) using the INVERSE of the combined run-to-template
             normalization matrix.

  Stage 2 -- Build a motion-aware 4D mask by applying each per-volume MCFlirt
             affine to the 3D native mask and merging into a 4D volume whose
             temporal length matches the raw BOLD.

  Stage 3 -- Mask the raw BOLD element-wise with the 4D motion-aware mask.

Subjects  : only 2-month-old subjects (IDs that do NOT end with 'A').
Task      : videos only.
Sessions  : ALL sessions found for each subject (auto-discovered from BIDS).
Runs      : ALL runs found per session that also have a matching norm matrix
            and MCFlirt mats directory (mismatches are skipped with a warning).

Parallelism
-----------
All per-run workflows are embedded in a single top-level meta-workflow so
nipype's scheduler sees every node across every subject/session/run at once.
Independent runs execute concurrently up to --n_procs cores, instead of
one run at a time as in a sequential for-loop.

Usage (all defaults)::

    python backnorm_pipeline.py

Selective run::

    python backnorm_pipeline.py --subjects IRN78 IRN64 --n_procs 16

Output layout (BIDS-derivative)::

    <output_dir>/sub-{sub}/ses-{ses}/func/
        sub-{sub}_ses-{ses}_task-videos_run-{run}_space-native_mask.nii.gz
        sub-{sub}_ses-{ses}_task-videos_run-{run}_space-native_desc-4dmotionmask.nii.gz
        sub-{sub}_ses-{ses}_task-videos_run-{run}_space-native_desc-maskedbold.nii.gz
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from nipype import Node, MapNode, Workflow
from nipype.interfaces import fsl


# =============================================================================
# Default paths  -- edit here or override via CLI
# =============================================================================

_BIDS_DIR = "/lustre/disk/home/shared/cusacklab/foundcog/bids"

DEFAULTS = dict(
    bids_dir         = _BIDS_DIR,
    workingdir       = f"{_BIDS_DIR}/workingdir",
    mcflirt_mats_dir = f"{_BIDS_DIR}/derivatives/motion_affines/mcflirt_mats_output",
    template_mask    = (
        f"{_BIDS_DIR}/derivatives/templates/mask/"
        "binary_mask_from_julichbrainatlas_3.1_207areas_MPM_MNI152"
        "_space-nihpd-02-05_2mm.nii.gz"
    ),
    output_dir      = f"{_BIDS_DIR}/derivatives/faizan_analysis",
    nipype_work_dir = f"{_BIDS_DIR}/derivatives/faizan_analysis/.nipype_work",
    n_procs         = 4,
)


# Subjects excluded from all processing regardless of what BIDS contains
EXCLUDE_SUBS = ["ICC89", "ICC103", "ICN50", "ICC57"]

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bids_dir",         default=DEFAULTS["bids_dir"],
                   help="BIDS root directory")
    p.add_argument("--workingdir",       default=DEFAULTS["workingdir"],
                   help="Directory containing per-subject normalization matrices")
    p.add_argument("--mcflirt_mats_dir", default=DEFAULTS["mcflirt_mats_dir"],
                   help="Root of the MCFlirt per-volume affine matrices tree")
    p.add_argument("--template_mask",    default=DEFAULTS["template_mask"],
                   help="Binary mask in template (NIHPD) space (.nii.gz)")
    p.add_argument("--output_dir",       default=DEFAULTS["output_dir"],
                   help="Pipeline output root (BIDS-derivative)")
    p.add_argument("--nipype_work_dir",  default=DEFAULTS["nipype_work_dir"],
                   help="Nipype scratch directory for intermediate files")
    p.add_argument("--subjects",         nargs="+", default=None,
                   help="Subject IDs to process (default: all 2-month subjects)")
    p.add_argument("--n_procs",          type=int, default=DEFAULTS["n_procs"],
                   help="Total parallel processes shared across ALL runs")
    return p.parse_args()


# =============================================================================
# Path construction helpers
# =============================================================================

def bold_path(bids_dir: str, subject: str, session: str, run: str) -> Path:
    """Raw BOLD NIfTI for the videos task."""
    return (
        Path(bids_dir)
        / f"sub-{subject}" / f"ses-{session}" / "func"
        / f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}_bold.nii.gz"
    )


def norm_mat_path(workingdir: str, subject: str, session: str, run: str) -> Path:
    """
    Forward normalization matrix  (native EPI -> reference run -> template).
    We invert this to go  template -> native EPI.
    """
    return (
        Path(workingdir)
        / subject / "derivatives" / "preproc"
        / f"_subject_id_{subject}"
        / "_referencetype_standard"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "combine_xfms_manual_selection"
        / (
            f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}"
            "_bold_mcf_corrected_mean_flirt_average_flirt.mat"
        )
    )


def mcflirt_mats_path(mcflirt_mats_dir: str, subject: str, session: str, run: str) -> Path:
    """Directory that contains MAT_0000 ... MAT_N for this run."""
    return (
        Path(mcflirt_mats_dir)
        / f"_subject_id_{subject}"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "mats"
    )


def output_func_dir(output_dir: str, subject: str, session: str) -> Path:
    """BIDS-derivative func folder for a subject/session."""
    return Path(output_dir) / f"sub-{subject}" / f"ses-{session}" / "func"


def output_prefix(subject: str, session: str, run: str) -> str:
    """Shared BIDS filename prefix for all outputs of this run."""
    return f"sub-{subject}_ses-{session}_task-videos_run-{run}"


# =============================================================================
# Discovery  -- subjects, sessions, runs
# =============================================================================

def find_subjects(bids_dir: str) -> List[str]:
    """
    All 2-month-old subject IDs in the BIDS directory.
    9-month re-scan subjects (ending with 'A') and any subject in
    EXCLUDE_SUBS are filtered out.
    """
    return sorted(
        d.name[4:]
        for d in Path(bids_dir).iterdir()
        if d.is_dir()
        and d.name.startswith("sub-")
        and not d.name.endswith("A")
        and d.name[4:] not in EXCLUDE_SUBS
    )


def find_sessions_and_runs(bids_dir: str, subject: str) -> List[Tuple[str, str]]:
    """
    Return ALL (session, run) pairs that have a videos-task BOLD file for
    this subject, across every session directory found under sub-{subject}/.
    No sessions or runs are assumed -- everything is discovered from BIDS.
    """
    pairs: List[Tuple[str, str]] = []
    sub_dir = Path(bids_dir) / f"sub-{subject}"

    for ses_dir in sorted(sub_dir.iterdir()):
        if not (ses_dir.is_dir() and ses_dir.name.startswith("ses-")):
            continue
        session  = ses_dir.name[4:]     # e.g. "1", "2"
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
# Path validation  -- runs before any workflow is built
# =============================================================================

_G = "\033[32m"
_R = "\033[31m"
_Y = "\033[33m"
_E = "\033[0m"

def _ok(msg):   return f"{_G}  OK   {_E}{msg}"
def _miss(msg): return f"{_R}  MISS {_E}{msg}"
def _warn(msg): return f"{_Y}  WARN {_E}{msg}"


def check_all_paths(
    subjects: List[str],
    args:     argparse.Namespace,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Validate every input for all subjects x sessions x runs and print a
    colour-coded report.

    Checks
    ------
    1. Template mask exists (global, once)
    2. Raw BOLD NIfTI exists
    3. Forward normalisation matrix exists
    4. MCFlirt mats directory exists and contains MAT_* files
    5. MAT file count == BOLD volume count

    Only combinations that pass ALL checks are added to 'ready'.

    Returns
    -------
    ready   : (subject, session, run) tuples that passed all checks
    skipped : human-readable labels of combinations that failed
    """
    import nibabel as nib

    ready:   List[Tuple[str, str, str]] = []
    skipped: List[str]                  = []
    total_issues = 0

    print("\n" + "=" * 70)
    print("PATH VALIDATION")
    print("=" * 70)

    # 1. Global: template mask
    tmask = Path(args.template_mask)
    if tmask.exists():
        print(_ok(f"template_mask   {tmask}"))
    else:
        print(_miss(f"template_mask   {tmask}"))
        total_issues += 1
    print()

    # 2. Per subject / session / run
    for subject in subjects:
        pairs = find_sessions_and_runs(args.bids_dir, subject)
        print("-" * 70)

        if not pairs:
            print(_miss(f"sub-{subject}  -- no videos-task BOLD found in BIDS"))
            skipped.append(f"sub-{subject} (no BOLD found)")
            total_issues += 1
            continue

        for session, run in pairs:
            tag = f"sub-{subject}  ses-{session}  run-{run}"
            print(tag)
            issues = 0

            # BOLD
            b = bold_path(args.bids_dir, subject, session, run)
            if b.exists():
                print(_ok(f"  BOLD         {b}"))
            else:
                print(_miss(f"  BOLD         {b}"))
                issues += 1

            # Norm matrix
            m = norm_mat_path(args.workingdir, subject, session, run)
            if m.exists():
                print(_ok(f"  norm_mat     {m}"))
            else:
                print(_miss(f"  norm_mat     {m}"))
                issues += 1

            # MCFlirt mats directory + file count
            md        = mcflirt_mats_path(args.mcflirt_mats_dir, subject, session, run)
            mat_files = sorted(md.glob("MAT_*")) if md.is_dir() else []

            if md.is_dir() and mat_files:
                print(_ok(f"  mcflirt_dir  {md}  ({len(mat_files)} MAT files)"))

                # Volume count match
                if b.exists():
                    try:
                        n_vols = nib.load(str(b)).shape[3]
                        if len(mat_files) == n_vols:
                            print(_ok(
                                f"  vol_count    {len(mat_files)} MAT == {n_vols} BOLD volumes"
                            ))
                        else:
                            print(_warn(
                                f"  vol_count    {len(mat_files)} MAT != {n_vols} BOLD volumes  <- mismatch!"
                            ))
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

    # Summary
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
# Per-run Nipype workflow
# =============================================================================

def build_run_workflow(
    subject:         str,
    session:         str,
    run:             str,
    bold:            str,
    fwd_mat:         str,
    mats:            List[str],
    template_mask:   str,
    output_dir:      str,
    nipype_work_dir: str,
) -> Workflow:
    """
    Self-contained Nipype workflow for one subject / session / run.

    Workflow graph
    --------------
    mean_bold -------------------------------------------------------+
                                                                     v
    invert_mat ----------------------------------------> backnorm_3d (FLIRT -applyxfm NN)
                                                                     |
                                               +---------------------+
                                               |  (3D native mask)
                                               v
                                  per_vol_mask  (MapNode: one FLIRT per MAT file)
                                               |  (list of N 3D volumes)
                                               v
                                         merge_4d  (fslmerge -t)
                                               |  (4D motion-aware mask)
                                               v
                                       apply_mask  (fslmaths -mas)
                                               |
                                               v
                                        masked BOLD (4D)
    """
    out_dir = output_func_dir(output_dir, subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = output_prefix(subject, session, run)

    wf = Workflow(
        name     = f"run_sub{subject}_ses{session}_run{run}",
        base_dir = nipype_work_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1A  --  mean BOLD  (defines the native EPI reference grid)
    # Command   :  fslmaths <bold> -Tmean <mean_bold>
    # ------------------------------------------------------------------
    mean_bold = Node(
        fsl.ImageMaths(
            in_file   = bold,
            op_string = "-Tmean",
            out_file  = str(out_dir / f"{pfx}_meanbold.nii.gz"),
        ),
        name = "mean_bold",
    )

    # ------------------------------------------------------------------
    # Stage 1B  --  invert forward norm matrix
    # Forward   :  native EPI -> reference run -> template
    # Inverse   :  template -> native EPI
    # Command   :  convert_xfm -inverse <fwd_mat> -omat <inv_mat>
    # ------------------------------------------------------------------
    invert_mat = Node(
        fsl.ConvertXFM(
            in_file    = fwd_mat,
            invert_xfm = True,
            out_file   = str(out_dir / f"{pfx}_norm_matrix_inverse.mat"),
        ),
        name = "invert_mat",
    )

    # ------------------------------------------------------------------
    # Stage 1C  --  back-normalize mask to native space (3D output)
    # Command   :  flirt -in <template_mask> -ref <mean_bold>
    #                    -applyxfm -init <inv_mat> -interp nearestneighbour
    # ------------------------------------------------------------------
    backnorm_3d = Node(
        fsl.FLIRT(
            in_file   = template_mask,
            interp    = "nearestneighbour",
            apply_xfm = True,
            out_file  = str(out_dir / f"{pfx}_space-native_mask.nii.gz"),
        ),
        name = "backnorm_3d",
    )
    wf.connect([
        (mean_bold,  backnorm_3d, [("out_file", "reference")]),
        (invert_mat, backnorm_3d, [("out_file", "in_matrix_file")]),
    ])

    # ------------------------------------------------------------------
    # Stage 2A  --  invert each per-volume MCFlirt affine
    #
    # MCFlirt produces MAT_t  :  vol_t -> mean  (aligns each vol to mean)
    # We need  inv(MAT_t)     :  mean  -> vol_t (to move mask into vol_t space)
    #
    # Command   :  convert_xfm -inverse MAT_t -omat inv_MAT_t
    # ------------------------------------------------------------------
    invert_motion_mats = MapNode(
        fsl.ConvertXFM(invert_xfm=True),
        iterfield = ["in_file"],
        name      = "invert_motion_mats",
    )
    invert_motion_mats.inputs.in_file = mats

    # ------------------------------------------------------------------
    # Stage 2B  --  extract each individual volume from BOLD
    #
    # inv(MAT_t) maps mean -> vol_t so vol_t must be the reference grid.
    # We extract each volume so flirt can use it as the per-timepoint ref.
    # Command   :  fslroi <bold> <vol_t> t 1
    # ------------------------------------------------------------------
    extract_vols = MapNode(
        fsl.ExtractROI(
            in_file = bold,
            t_size  = 1,
        ),
        iterfield = ["t_min"],
        name      = "extract_vols",
    )
    extract_vols.inputs.t_min = list(range(len(mats)))

    # ------------------------------------------------------------------
    # Stage 2C  --  apply inv(MAT_t) to 3D native mask, ref = vol_t
    #
    # This places the mask at the exact brain position for timepoint t.
    # Command   :  flirt -in <3d_native_mask> -ref <vol_t>
    #                    -applyxfm -init <inv_MAT_t> -interp nearestneighbour
    # Note      :  fsl.Merge does not accept out_file; nipype auto-names it
    # ------------------------------------------------------------------
    per_vol_mask = MapNode(
        fsl.FLIRT(
            interp    = "nearestneighbour",
            apply_xfm = True,
        ),
        iterfield = ["in_matrix_file", "reference"],
        name      = "per_vol_mask",
    )

    wf.connect([
        (invert_motion_mats, per_vol_mask, [("out_file",  "in_matrix_file")]),
        (extract_vols,       per_vol_mask, [("roi_file",  "reference")]),
        (backnorm_3d,        per_vol_mask, [("out_file",  "in_file")]),
    ])

    # ------------------------------------------------------------------
    # Stage 2B  --  merge per-volume masks into a 4D motion-aware mask
    # Temporal length == number of MAT files == BOLD n_volumes
    # Command   :  fslmerge -t <out_4d> <vol_0> ... <vol_N>
    # ------------------------------------------------------------------
    merge_4d = Node(
        fsl.Merge(dimension="t"),
        name = "merge_4d",
    )
    wf.connect([
        (per_vol_mask, merge_4d, [("out_file", "in_files")]),
    ])

    # ------------------------------------------------------------------
    # Stage 3   --  apply 4D mask to raw BOLD (element-wise multiply)
    # masked_bold[x,y,z,t] = bold[x,y,z,t] * mask[x,y,z,t]
    # Command   :  fslmaths <bold> -mas <4d_mask> <masked_bold>
    # ------------------------------------------------------------------
    apply_mask = Node(
        fsl.ApplyMask(
            in_file  = bold,
            out_file = str(out_dir / f"{pfx}_space-native_desc-maskedbold.nii.gz"),
        ),
        name = "apply_mask",
    )
    wf.connect([
        (merge_4d, apply_mask, [("merged_file", "mask_file")]),
    ])

    return wf


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # -- Discover subjects (always enforce EXCLUDE_SUBS) ----------------------
    if args.subjects:
        excluded = [s for s in args.subjects if s in EXCLUDE_SUBS]
        subjects = [s for s in args.subjects if s not in EXCLUDE_SUBS]
        if excluded:
            print(f"\nExcluded (in EXCLUDE_SUBS): {excluded}")
    else:
        subjects = find_subjects(args.bids_dir)   # EXCLUDE_SUBS filtered inside

    print(f"\nAlways-excluded subjects: {EXCLUDE_SUBS}")
    print(f"Subjects to process ({len(subjects)} total):")
    print("  " + "  ".join(subjects))

    # -- Validate all inputs up front, print full colour-coded report ---------
    ready, skipped = check_all_paths(subjects, args)

    if not ready:
        print("No valid subject/session/run combinations found. Exiting.")
        sys.exit(1)

    # -- Build a single meta-workflow containing all per-run sub-workflows ----
    #
    #   WHY: a for-loop of wf.run() calls is sequential -- MultiProc only
    #   parallelises nodes *within* one run at a time, so every other subject
    #   sits idle while one run's 400-500 FLIRT MapNode calls execute.
    #
    #   By embedding every per-run workflow as a child of one meta-workflow,
    #   nipype's scheduler sees all nodes across all subjects/sessions/runs
    #   simultaneously and fills n_procs cores globally.  Stage-1 of run-002
    #   can run while Stage-2 of run-001 is still in progress, etc.
    #
    meta_wf = Workflow(
        name     = "backnorm_all_subjects",
        base_dir = args.nipype_work_dir,
    )

    for subject, session, run in ready:
        mats = sorted(str(m) for m in mcflirt_mats_path(
            args.mcflirt_mats_dir, subject, session, run
        ).glob("MAT_*"))

        run_wf = build_run_workflow(
            subject         = subject,
            session         = session,
            run             = run,
            bold            = str(bold_path(args.bids_dir, subject, session, run)),
            fwd_mat         = str(norm_mat_path(args.workingdir, subject, session, run)),
            mats            = mats,
            template_mask   = args.template_mask,
            output_dir      = args.output_dir,
            nipype_work_dir = args.nipype_work_dir,
        )
        meta_wf.add_nodes([run_wf])

    # -- Execute the full graph in one shot -----------------------------------
    print(
        f"Submitting {len(ready)} run(s) across {len(subjects)} subject(s) "
        f"to a single scheduler using {args.n_procs} parallel process(es).\n"
        f"All independent nodes across subjects/runs execute concurrently.\n"
    )
    meta_wf.run(plugin="MultiProc", plugin_args={"n_procs": args.n_procs})

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Ran      : {len(ready)} run(s)")
    print(f"  Skipped  : {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    - {s}")
    print(f"  Outputs  : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
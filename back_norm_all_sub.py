#!/usr/bin/env python3
"""
backnorm_pipeline.py
====================
Nipype pipeline for three stages:

  Stage 1 — Back-normalize a binary mask from template space → native BOLD space (3D)
             Uses the INVERSE of the combined run-to-template normalization matrix.

  Stage 2 — Build a motion-aware 4D mask
             Applies each per-volume MCFlirt affine to the 3D native mask,
             then merges into a 4D volume matching the raw BOLD temporal length.

  Stage 3 — Mask the raw BOLD
             Multiplies the 4D BOLD by the 4D motion-aware mask (element-wise).

Subjects  : only 2-month-old subjects (IDs that do NOT end with 'A').
Task      : videos only.
Sessions  : all available sessions discovered automatically per subject.
Runs      : all runs discovered automatically per subject/session.

Usage (all defaults)::

    python backnorm_pipeline.py

Selective run::

    python backnorm_pipeline.py --subjects IRN78 IRN64 --n_procs 8

Output layout (BIDS-derivative)::

    <output_dir>/
      sub-{subject}/ses-{session}/func/
        sub-{subject}_ses-{session}_task-videos_run-{run}_space-native_mask.nii.gz
        sub-{subject}_ses-{session}_task-videos_run-{run}_space-native_desc-4dmotionmask.nii.gz
        sub-{subject}_ses-{session}_task-videos_run-{run}_space-native_desc-maskedbold.nii.gz
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from nipype import Node, MapNode, Workflow
from nipype.interfaces import fsl


# =============================================================================
# Default paths — edit here or override via CLI arguments
# =============================================================================

_BIDS_DIR = "/lustre/disk/home/shared/cusacklab/foundcog/bids"

DEFAULTS = dict(
    bids_dir         = _BIDS_DIR,
    workingdir       = f"{_BIDS_DIR}/workingdir",
    # !! Update this to the actual location of your mcflirt_mats_output !!
    mcflirt_mats_dir = "/lustre/disk/home/shared/cusacklab/foundcog/mcflirt_mats_output",
    template_mask    = (
        f"{_BIDS_DIR}/derivatives/templates/mask/"
        "binary_mask_from_julichbrainatlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz"
    ),
    output_dir       = f"{_BIDS_DIR}/derivatives/faizan_analysis",
    nipype_work_dir  = f"{_BIDS_DIR}/derivatives/faizan_analysis/.nipype_work",
    n_procs          = 4,
)


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
                   help="Root directory of MCFlirt per-volume affine matrices")
    p.add_argument("--template_mask",    default=DEFAULTS["template_mask"],
                   help="Binary mask in template (MNI / NIHPD) space (.nii.gz)")
    p.add_argument("--output_dir",       default=DEFAULTS["output_dir"],
                   help="Pipeline output root (BIDS-derivative)")
    p.add_argument("--nipype_work_dir",  default=DEFAULTS["nipype_work_dir"],
                   help="Nipype scratch directory for intermediate files")
    p.add_argument("--subjects",         nargs="+", default=None,
                   help="Subject IDs to process (default: all 2-month subjects in BIDS)")
    p.add_argument("--n_procs",          type=int, default=DEFAULTS["n_procs"],
                   help="Number of parallel processes for MultiProc runner")
    return p.parse_args()


# =============================================================================
# Path construction helpers
# =============================================================================

def bold_path(bids_dir: str, subject: str, session: str, run: str) -> Path:
    """Raw BOLD NIfTI for the videos task."""
    return (
        Path(bids_dir)
        / f"sub-{subject}"
        / f"ses-{session}"
        / "func"
        / f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}_bold.nii.gz"
    )


def norm_mat_path(workingdir: str, subject: str, session: str, run: str) -> Path:
    """
    Forward normalization matrix (native EPI -> reference run -> template space).
    Invert this to go template -> native.
    """
    return (
        Path(workingdir)
        / subject
        / "derivatives" / "preproc"
        / f"_subject_id_{subject}"
        / "_referencetype_standard"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "combine_xfms_manual_selection"
        / (
            f"sub-{subject}_ses-{session}_task-videos_dir-AP_run-{run}"
            "_bold_mcf_corrected_mean_flirt_average_flirt.mat"
        )
    )


def get_mcflirt_mats(mcflirt_mats_dir: str, subject: str, session: str, run: str) -> List[str]:
    """
    Sorted list of per-volume MCFlirt affine matrices.
    Each MAT_XXXX file is the affine for one BOLD timepoint.
    """
    mats_dir = (
        Path(mcflirt_mats_dir)
        / f"_subject_id_{subject}"
        / f"_run_{run}_session_{session}_task_name_videos"
        / "mats"
    )
    return sorted(str(m) for m in mats_dir.glob("MAT_*"))


def output_func_dir(output_dir: str, subject: str, session: str) -> Path:
    """BIDS-derivative func folder for all outputs of this subject/session."""
    return Path(output_dir) / f"sub-{subject}" / f"ses-{session}" / "func"


def output_prefix(subject: str, session: str, run: str) -> str:
    """Shared filename prefix following BIDS convention."""
    return f"sub-{subject}_ses-{session}_task-videos_run-{run}"


# =============================================================================
# Subject / session / run discovery
# =============================================================================

def is_2mo(subject_id: str) -> bool:
    """2-month-old subjects do NOT end with 'A'."""
    return not subject_id.endswith("A")


def find_subjects(bids_dir: str) -> List[str]:
    """Return sorted list of all 2-month-old subject IDs present in the BIDS dir."""
    return sorted(
        d.name[4:]                            # strip leading 'sub-'
        for d in Path(bids_dir).iterdir()
        if d.is_dir()
        and d.name.startswith("sub-")
        and is_2mo(d.name[4:])
    )


def find_sessions_and_runs(bids_dir: str, subject: str) -> List[Tuple[str, str]]:
    """
    Return all (session, run) pairs that have a videos-task BOLD file
    for the given subject.  E.g. [('1', '001'), ('1', '002'), ('1', '003')].
    """
    pairs: List[Tuple[str, str]] = []
    sub_dir = Path(bids_dir) / f"sub-{subject}"

    for ses_dir in sorted(sub_dir.iterdir()):
        if not (ses_dir.is_dir() and ses_dir.name.startswith("ses-")):
            continue
        session  = ses_dir.name[4:]           # strip 'ses-'
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
# Input validation
# =============================================================================

def validate_inputs(
    bold: Path,
    fwd_mat: Path,
    mats: List[str],
    template_mask: str,
) -> List[str]:
    """
    Return a list of problem descriptions.
    An empty list means all inputs are present and ready.
    """
    issues: List[str] = []
    for p in [bold, fwd_mat, Path(template_mask)]:
        if not p.exists():
            issues.append(f"Missing file: {p}")
    if not mats:
        issues.append("No MCFlirt per-volume matrices found (check --mcflirt_mats_dir)")
    return issues


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
    Build a self-contained Nipype workflow for one subject / session / run.

    Workflow graph
    --------------
    mean_bold ──────────────────────────────────────────────────────────┐
                                                                         ▼
    invert_mat ──────────────────────────────────────► backnorm_3d  (FLIRT -applyxfm NN)
                                                                         │
                                                    ┌────────────────────┘
                                                    │  (3D native mask)
                                                    ▼
                                       per_vol_mask  (MapNode — one FLIRT per MAT file)
                                                    │  (list of 3D volumes)
                                                    ▼
                                           merge_4d  (fslmerge -t)
                                                    │  (4D motion-aware mask)
                                                    ▼
                                        apply_mask  (fslmaths -mas)
                                                    │
                                                    ▼
                                          masked BOLD (4D)
    """
    out_dir = output_func_dir(output_dir, subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = output_prefix(subject, session, run)

    wf = Workflow(
        name     = f"backnorm_sub{subject}_ses{session}_run{run}",
        base_dir = nipype_work_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1A — Mean BOLD
    #   Purpose : defines the native EPI voxel grid used as flirt reference
    #   Command : fslmaths <bold> -Tmean <mean_bold>
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
    # Stage 1B — Invert forward normalization matrix
    #   Forward : native EPI -> reference run -> template space
    #   Inverse : template space -> native EPI  (what we need)
    #   Command : convert_xfm -inverse <fwd_mat> -omat <inv_mat>
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
    # Stage 1C — Back-normalize mask to native BOLD space (3D output)
    #   Command : flirt -in <template_mask> -ref <mean_bold>
    #                   -applyxfm -init <inv_mat> -interp nearestneighbour
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
    # Stage 2A — Per-volume mask warping (motion-aware)
    #   MapNode fans out over every MCFlirt MAT file.
    #   Each node applies one affine to the 3D native mask -> 1 warped vol.
    #   The single in_file (3D mask) is broadcast to all map nodes.
    #   Command : flirt -in <3d_native_mask> -ref <mean_bold>
    #                   -applyxfm -init <MAT_t> -interp nearestneighbour
    # ------------------------------------------------------------------
    per_vol_mask = MapNode(
        fsl.FLIRT(
            interp    = "nearestneighbour",
            apply_xfm = True,
        ),
        iterfield = ["in_matrix_file"],     # one node per timepoint
        name      = "per_vol_mask",
    )
    per_vol_mask.inputs.in_matrix_file = mats   # MAT_0000 ... MAT_NNNN

    wf.connect([
        (backnorm_3d, per_vol_mask, [("out_file", "in_file")]),
        (mean_bold,   per_vol_mask, [("out_file", "reference")]),
    ])

    # ------------------------------------------------------------------
    # Stage 2B — Merge per-volume masks into a 4D motion-aware mask
    #   Temporal length equals the number of MAT files = BOLD n_volumes.
    #   Command : fslmerge -t <out_4d> <vol_0> <vol_1> ... <vol_N>
    # ------------------------------------------------------------------
    merge_4d = Node(
        fsl.Merge(
            dimension = "t",
            out_file  = str(out_dir / f"{pfx}_space-native_desc-4dmotionmask.nii.gz"),
        ),
        name = "merge_4d",
    )
    wf.connect([
        (per_vol_mask, merge_4d, [("out_file", "in_files")]),
    ])

    # ------------------------------------------------------------------
    # Stage 3 — Apply 4D mask to raw BOLD (element-wise)
    #   masked_bold[x,y,z,t] = bold[x,y,z,t] * mask[x,y,z,t]
    #   Command : fslmaths <bold> -mas <4d_mask> <masked_bold>
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

    # ── Discover subjects ─────────────────────────────────────────────────────
    subjects = args.subjects if args.subjects else find_subjects(args.bids_dir)
    print(f"\nSubjects to process ({len(subjects)} total):")
    print("  " + "  ".join(subjects))
    print()

    # ── Build one workflow per subject / session / run ────────────────────────
    workflows: List[Workflow] = []
    skipped:   List[str]      = []

    for subject in subjects:
        pairs = find_sessions_and_runs(args.bids_dir, subject)
        if not pairs:
            print(f"  [{subject}] No videos-task BOLD files found — skipping.")
            skipped.append(subject)
            continue

        for session, run in pairs:
            b    = bold_path(args.bids_dir, subject, session, run)
            mat  = norm_mat_path(args.workingdir, subject, session, run)
            mats = get_mcflirt_mats(args.mcflirt_mats_dir, subject, session, run)

            issues = validate_inputs(b, mat, mats, args.template_mask)
            tag    = f"sub-{subject} ses-{session} run-{run}"
            if issues:
                print(f"  [{tag}] Skipping — input problems:")
                for issue in issues:
                    print(f"    x {issue}")
                skipped.append(tag)
                continue

            print(f"  [{tag}] ready  ({len(mats)} volumes)")
            wf = build_run_workflow(
                subject         = subject,
                session         = session,
                run             = run,
                bold            = str(b),
                fwd_mat         = str(mat),
                mats            = mats,
                template_mask   = args.template_mask,
                output_dir      = args.output_dir,
                nipype_work_dir = args.nipype_work_dir,
            )
            workflows.append(wf)

    # ── Run ───────────────────────────────────────────────────────────────────
    if not workflows:
        print("\nNo workflows to run. Check paths and input problems above.")
        sys.exit(1)

    print(f"\nRunning {len(workflows)} workflow(s) with {args.n_procs} parallel processes ...\n")
    for wf in workflows:
        wf.run(plugin="MultiProc", plugin_args={"n_procs": args.n_procs})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Ran      : {len(workflows)} workflow(s)")
    print(f"  Skipped  : {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    - {s}")
    print(f"  Outputs  : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
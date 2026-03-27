# Inter-Subject Correlation in fMRI Native Space

Analysis of neural responses across subjects watching identical video stimuli, focusing on the visual cortex (V1 area) in native space.Subjects watch the same video stimuli in specific orders. The visual stimuli are expected to strongly drive responses in the primary visual cortex. Analysis is performed in **native space**, where the BOLD signal retains motion and other artefacts.

## Visual Cortex (V1) Mask

The V1 mask is defined in **MNI template space** at **2mm resolution**, derived from the Julich Brain Atlas which assigns integer labels to brain regions. Grayvalue `91` corresponds to the primary visual cortex.
```
/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/
```

Atlas files can be found at:

| Age group | File |
|-----------|------|
| 2 months  | `JulichBrainAtlas_3.1_207areas_MPM_MNI152_space-nihpd-02-05_2mm.nii.gz` |
| 9 months  | `JulichBrainAtlas_3.1_207areas_MPM_MNI152_space-nihpd-08-11_2mm.nii.gz` |

> The atlas is available on both **maguire** and **foundcog** servers.

### Extracting the binary V1 mask

1. Set the correct input/output paths inside:
   ```
   useful_scripts/get_v1_binary_mask_from_julich_bilateral_atlast.sh
   ```

The binary mask lives in 3D template space, so it has to go back to the native space. Therefore, there is affine matrix lives in maguire with the following directory structure. 

```bash
/lustre/disk/home/shared/cusacklab/foundcog/bids/workingdir/
```

### Subjects (Top-Level)

```bash
workingdir/
├── ICC103/   ├── ICC103A/  ├── ICC105/   ├── ICC105A/
├── ICC107/   ├── ICC107A/  ├── ICC111/   ├── ICC111A/
├── ICC113/   ├── ICC113A/  ├── ICC115/   ├── ICC117/
...
```

Example Subject: IRN78 (Detailed)

```bash
IRN78/
└── derivatives/
    └── preproc/
        └── _subject_id_IRN78/
            └── _referencetype_standard/
                ├── flirt_manualselection/
                ├── _run_001_session_1_task_name_videos/
                │   └── combine_xfms_manual_selection/
                │       └── sub-IRN78_ses-1_task-videos_dir-AP_run-001_bold_mcf_corrected_mean_flirt_average_flirt.mat
                ├── _run_002_session_1_task_name_videos/
                └── _run_003_session_1_task_name_videos/
```

so here the ```sub-IRN78_ses-1_task-videos_dir-AP_run-001_bold_mcf_corrected_mean_flirt_average_flirt.mat``` is the affine transformation that normalize from the invidual 
run space to reference run space within each subject and from the reference run space to template space. Therefore, inverse of that would project it back to individual run space (3mm) native bold grid. 
This is where the mask in in native space, however, the mask does not know the exact location of the visual cortex region in the raw bold data which non-motion correction. This is where the motion affine matrix during the 
motion correction comes into play, they project the mask from the mean bold EPI data to invidiual repetition where the motion is present. Therefore, inverse of those motion matrices would take the mask to non-motion corrected bold space so it can align with the visual cortext (V1) region. To locate those affine on maguire:

```bash id="h7x3kq"
/lustre/disk/home/shared/cusacklab/foundcog/bids/workingdir/derivatives/motion_affines/
```
subject level structure:

```bash id="r8k1zp"
motion_affines/
├── logs/
├── mcflirt_mats_output/
│   ├── _subject_id_ICC103/
│   ├── _subject_id_ICC105/
│   ├── _subject_id_ICC107/
│   ├── ...
│   ├── _subject_id_IRN78/
│   └── ...
└── mcflirt_mats_workdir/
```
Example Subject: IRN78

```bash id="v9p2ds"
mcflirt_mats_output/
└── _subject_id_IRN78/
    ├── _run_001_session_1_task_name_videos/
    │   ├── chosen_reference/
    │   ├── mcf_epi/
    │   └── mats/
    │       ├── MAT_0000
    │       ├── MAT_0001
    │       ├── MAT_0002
    │       ├── MAT_0003
    │       ├── MAT_0004
    │       ├── ...
    │       ├── MAT_0479
    │       └── MAT_0480
    │
    └── _run_002_session_1_task_name_videos/
        ├── chosen_reference/
        ├── mcf_epi/
        └── mats/
            ├── MAT_0000
            ├── MAT_0001
            ├── ...
```

## Usage
after preparing the necessary inputs and configuring the paths we run the slurm job to backnormalize and get masked bold data for visual cortext region, run this script:

```bash
sbatch run_back_norm_all_subs.sh
```
Nipype pipeline -- three stages:

```Stage 1```:   Back-normalize a binary mask from template space to native BOLD
             space (3D) using the INVERSE of the combined run-to-template
             normalization matrix.

```Stage 2``` :   Build a motion-aware 4D mask by applying each per-volume MCFlirt
             affine to the 3D native mask and merging into a 4D volume whose
             temporal length matches the raw BOLD.

```Stage 3``` :  Mask the raw BOLD element-wise with the 4D motion-aware mask.

**Subjects**  : only 2-month-old subjects (IDs that do NOT end with 'A').

**Task**      : videos only.

**Sessions**  : ALL sessions found for each subject (auto-discovered from BIDS).

**Runs**      : ALL runs found per session that also have a matching norm matrix
            and MCFlirt mats directory (mismatches are skipped with a warning).

this will save the output as follows:

```bash id="b0x9kt"
/lustre/disk/home/users/mfaizan/faizan_analysis/
```
Top level structure. 
```bash id="h3k2pd"
faizan_analysis/
├── nipype_work/
└── sub-IRN78/
```
Subject: IRN78

```bash id="k9w4lm"
sub-IRN78/
└── ses-1/
    └── func/
        ├── sub-IRN78_ses-1_task-videos_run-001_meanbold.nii.gz
        ├── sub-IRN78_ses-1_task-videos_run-001_norm_matrix_inverse.mat
        ├── sub-IRN78_ses-1_task-videos_run-001_space-native_desc-maskedbold.nii.gz
        ├── sub-IRN78_ses-1_task-videos_run-001_space-native_mask.nii.gz
        │
        ├── sub-IRN78_ses-1_task-videos_run-002_meanbold.nii.gz
        ├── sub-IRN78_ses-1_task-videos_run-002_norm_matrix_inverse.mat
        ├── sub-IRN78_ses-1_task-videos_run-002_space-native_desc-maskedbold.nii.gz
        └── sub-IRN78_ses-1_task-videos_run-002_space-native_mask.nii.gz
```
This should have successufly saved the each subject output for masked epi, inverse affine that backnormalize the bold from template to native space, mask in native space etc. 

### Sanity check backnormalized mask 
Sanity checks and visualisations for one subject's backnorm pipeline output.

Checks performed

1.  DIMENSION CHECK
      - 3D native mask ```spatial dims  == raw BOLD spatial dims (x, y, z)```
      - 4D motion mask ```temporal dim  == raw BOLD temporal dim (n_vols)```
      - 4D motion mask ```spatial dims  == raw BOLD spatial dims```

2.  VOXEL COUNT / VOLUME CHECK
      - Template mask voxel count and mm³ volume
      - 3D native mask voxel count and mm³ volume
      - Ratio should be ~1.0 (same brain region, different voxel sizes)
      - Each timepoint's slice of the 4D mask should have non-zero voxels

3.  MOTION EFFECT CHECK
      - Per-timepoint voxel count in 4D mask across the run
        (should fluctuate slightly due to head motion, not be constant or zero)
      - Centroid shift of the mask across timepoints (x, y, z)

4.  MASKED EPI SIGNAL CHECK
      - Mean signal inside mask vs outside mask across time
      - SNR estimate: mean / std of the within-mask signal timeseries
      - Check for zero or near-zero frames (failed masking)

Usage:
```bash 
python sanity_check_backnorm.py --subject IRN78

python sanity_check_backnorm.py \\
   --subject IRN78 \\
   --session 1 \\
   --run 001 \\
   --output_dir /lustre/.../faizan_analysis \\
   --bids_dir   /lustre/.../bids \\
   --template_mask /lustre/.../binary_mask_...2mm.nii.gz \\
   --template_bg   /lustre/.../nihpd_asym_02-05_t2w_2mm.nii.gz \\
   --save_dir  ./sanity_checks
```

The following image visualize how the mask are overlayed over the raw bold signal these mask are motion aware as we do ```inv(affine)```

![alt text](https://github.com/mfaizan-ai/get_isc_in_native_space/blob/main/images/sub-IRN78_ses-1_run-001_fig5_motion_mask_frames.png)

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
├── ICC123/   ├── ICC125/   ├── ICC127/   ├── ICC127A/
├── ICC131/   ├── ICC133/   ├── ICC133A/  ├── ICC135/
├── ICC139/   ├── ICC139A/  ├── ICC141/   ├── ICC145/
├── ICC145A/  ├── ICC147/   ├── ICC147A/  ├── ICC149/
├── ICC151/   ├── ICC153/   ├── ICC153A/  ├── ICC155/
├── ICC155A/  ├── ICC157/   ├── ICC159/   ├── ICC161/
├── ICC163/   ├── ICC163A/  ├── ICC165/   ├── ICC167/
├── ICC167A/  ├── ICC169/   ├── ICC177/   ├── ICC179/
├── ICC179A/  ├── ICC183/   ├── ICC183A/  ├── ICC185/
├── ICC187/   ├── ICC189/   ├── ICC191/   ├── ICC191A/
├── ICC193/   ├── ICC193A/  ├── ICC195/   ├── ICC197/
├── ICC199/   ├── ICC201/   ├── ICC201A/  ├── ICC203/
├── ICC205/   ├── ICC205A/  ├── ICC207/   ├── ICC207A/
├── ICC211/   ├── ICC213/   ├── ICC217/   ├── ICC219/
├── ICC221/   ├── ICC221A/  ├── ICC223/   ├── ICC225/
├── ICC227/   ├── ICC229/   ├── ICC233/   ├── ICC235/
├── ICC237/   ├── ICC237A/  ├── ICC239/   ├── ICC239A/
├── ICC241/   ├── ICC241A/  ├── ICC243/   ├── ICC243A/
├── ICC247/   ├── ICC253/   ├── ICC255/   ├── ICC257/
├── ICC257A/
├── ICN2/     ├── ICN2A/    ├── ICN8/     ├── ICN8A/
├── ICN14/    ├── ICN14A/   ├── ICN18/    ├── ICN18A/
├── ICN46/    ├── ICN48/    ├── ICN50/
├── IRC1A/    ├── IRC3/     ├── IRC3A/    ├── IRC9/
├── IRC9A/    ├── IRC13/    ├── IRC17/    ├── IRC17A/
├── IRC19/    ├── IRC19A/   ├── IRC21/    ├── IRC25/
├── IRC27/    ├── IRC27A/   ├── IRC29/    ├── IRC29A/
├── IRC31/    ├── IRC31A/   ├── IRC41/    ├── IRC41A/
├── IRC45/    ├── IRC49/    ├── IRC61/    ├── IRC61A/
├── IRC67/    ├── IRC67A/   ├── IRC69/    ├── IRC69A/
├── IRC99/    ├── IRC109/   ├── IRC119/   ├── IRC119A/
├── IRN1/     ├── IRN1A/    ├── IRN12/    ├── IRN20/
├── IRN20A/   ├── IRN22/    ├── IRN26/    ├── IRN26A/
├── IRN30/    ├── IRN30A/   ├── IRN32/    ├── IRN34/
├── IRN36/    ├── IRN38/    ├── IRN42/    ├── IRN42A/
├── IRN44/    ├── IRN52/    ├── IRN60/    ├── IRN62/
├── IRN62A/   ├── IRN64/    ├── IRN66/    ├── IRN68/
├── IRN68A/   ├── IRN72/    ├── IRN74/    ├── IRN76/
├── IRN78/
└── ITT1/
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


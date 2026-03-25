# Inter-Subject Correlation in fMRI Native Space

Analysis of neural responses across subjects watching identical video stimuli, focusing on the visual cortex (V1 area) in native space.

---

## Overview

Subjects watch the same video stimuli in specific orders. The visual stimuli are expected to strongly drive responses in the primary visual cortex. Analysis is performed in **native space**, where the BOLD signal retains motion and other artefacts.

---

## Visual Cortex (V1) Mask

The V1 mask is defined in **MNI template space** at **2mm resolution**, derived from the Julich Brain Atlas which assigns integer labels to brain regions. Grayvalue `91` corresponds to the primary visual cortex.

### Atlas location

```
/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/
```

### Atlas files

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
2. Submit as a job — the script writes the binary mask to the specified destination path.


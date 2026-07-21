# AutoRAPNO Pipeline

An end-to-end pipeline for computing RAPNO (Response Assessment in Pediatric Neuro-Oncology) tumor measurements from segmentation masks, tracking longitudinal treatment response, and visualizing results.

## What it does

Given a folder of tumor segmentation masks (NIfTI) and clinical metadata (radiotherapy dates, scan dates), this pipeline automates the full RAPNO measurement and response-assessment workflow:

- **3D volumetric analysis** — computes tumor volume (cm³) directly from segmentation masks using voxel counting and spacing.
- **2D cross-sectional analysis** — identifies the two largest perpendicular diameters and the corresponding cross-sectional area for each slice, in each plane (axial, sagittal, coronal) or a single plane you specify, per RAPNO criteria. From there it can report the largest measurable area per plane across all slices, or the single largest measurable area across all planes for each scan.
- **Longitudinal response classification** — compares each follow-up scan to baseline (the first scan) or to the smallest tumor burden observed to date, and classifies each timepoint as Partial Response, Progressive Disease, Complete Response, or Stable Disease, based on RAPNO criteria for DIPG and pHGG cases.
- **Image Creation** — overlays segmentation masks on the underlying MRI (modality is configurable, e.g. T2 or FLAIR), producing pairwise (earlier-vs-later scan) comparison images at the radiologist- or pipeline-selected slice, useful for QC and reporting. It also generates .png of each mask annotated with its two computed cross-sectional diameters per slice, per scan — useful for tracking pipeline behavior and seeing how diameters evolve over time.
- **Clinical dataset normalization** — a flexible column-matching utility (`normalize_columns`) that maps inconsistent clinical spreadsheet headers (site-specific naming, mixed date formats) onto a standardized schema before merging with imaging-derived measurements.
- **Visualization** — a summary dashboard giving an overview across all patients.

## Pipeline overview

1. `tumor_measurements_2d_all_slices` — computes diameters/areas for every slice of every mask, across all three planes.
2. `compute_largest_diam_and_area` / `compute_largest_area_per_plane` — reduces per-slice results to the single largest (RAPNO-defining) measurement per scan/plane.
3. `tumor_3D_volume` — computes 3D tumor volume from the same masks and merges it with clinical metadata.
4. `overlap_masks_img` — generates side-by-side / overlaid visualizations of consecutive scans.
5. RAPNO progression logic — merges 2D and 3D outputs with clinical dates (RT start/end, scan dates) and derives per-timepoint and overall response classifications.

## Input requirements

- **Segmentation masks**: NIfTI files named `{pat_id}_{scandate}_mask.nii.gz` (scan date in `YYYYMMDD` format).
- **Clinical CSV**: patient ID, RT start/end dates, and scan dates. Column names don't need to match exactly — `normalize_columns` maps common variants onto the expected schema

## Examples
The csv/ folder contains a sample clinical CSV and an overlapped_img/ folder showing example output — use these as a reference for the expected input format and the resulting overlay visualizations before running the pipeline on your own data.

## Notes

This pipeline was developed for pediatric brain tumor (DMG, DIPG and phGG) segmentation and treatment-response analysis. It's designed to be reasonably dataset-agnostic, but assumes RAPNO-style bidimensional measurement conventions and pediatric neuro-oncology response criteria specifically — adapt the response-classification thresholds if applying to a different tumor type or response framework (e.g., RANO).

## Requirements

- Python 3.11+
- See [`requirements.txt`](requirements.txt) for dependencies

Install with:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

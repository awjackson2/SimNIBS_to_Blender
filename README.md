# SimNIBS Cortical Mesh → PLY (Regions + Whole GM)

This repository provides a single tool to export subject‑specific cortical regions and the whole gray‑matter (GM) surface from SimNIBS `.msh` files to PLY for Blender (or other tools). It uses the subject’s atlas from the `m2m_*` directory to ensure region accuracy (default atlas: `DK40`).

## Overview

The single script `scripts/cortical_regions_to_ply.py`:
- **Exports individual cortical regions to PLY** using the chosen atlas (default `DK40`)
- **Exports the whole GM surface to PLY**
- **Optionally samples a NIfTI field onto mesh nodes** and maps it to vertex colors or stores as scalars
- **Supports global colormap normalization** from a NIfTI file so colors are comparable across regions/meshes

## Requirements

### Software Dependencies
- **SimNIBS 4.0+** (with Python API)
- **Python 3.7+**
- **NumPy**
- **nibabel** (for NIfTI field data handling)
- **matplotlib** (optional, for colormaps; otherwise a simple blue↔red map is used)

### Input Requirements
- EITHER a cortical surface mesh (`.msh`) produced by `msh2cortex` OR a tetrahedral GM `.msh` (the tool can run `msh2cortex` for you)
- Subject’s `m2m_*` directory (for the subject atlas)
- Supported atlases: `DK40` (default), `DKTatlas40`, `HCP_MMP1`, `aparc.a2009s`
- Optional: NIfTI field file (e.g., `*_TI_max.nii.gz`) for field coloring

## Installation

1. **Install SimNIBS** following the official instructions: https://simnibs.github.io/simnibs/
2. **Download the scripts** to your working directory:
   ```bash
   # Download the scripts
   git clone <repository-url>
   cd simnibs-cortical-regions
   ```
3. **Make scripts executable** (optional):
   ```bash
   chmod +x simnibs_cortical_regions.py
   chmod +x simnibs_region_utils.py
   ```

## Usage

### Basic Usage

Export atlas‑accurate region PLYs and the whole GM PLY (default atlas: `DK40`):

```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --field-file subject_overlays/subject_TI_max.nii.gz
```

### Command Line Options

```bash
simnibs_python scripts/cortical_regions_to_ply.py [OPTIONS]

Required (one of):
  --mesh           Cortical surface .msh (from msh2cortex)
  --gm-mesh        Tetrahedral GM .msh (the tool will run msh2cortex)
  --m2m            Subject m2m directory
  --output-dir     Output directory

Optional:
  --atlas          Atlas name (default: DK40)
  --surface        Surface when using --gm-mesh: central|pial|white (default: central)
  --msh2cortex     Path to msh2cortex executable (if not on PATH)
  --field-file     NIfTI file to sample onto nodes (e.g., TI_max)
  --field          Field name to use/store (default: TI_max)
  --scalars        Store scalars instead of vertex colors
  --colormap       Colormap name (default: viridis)
  --field-range    MIN MAX explicit range for mapping
  --global-from-nifti  Use global min/max from the given NIfTI for color scaling
  --skip-regions   Do not export individual region PLYs
  --skip-whole-gm  Do not export the whole GM PLY
```

### Examples

1) Regions + whole GM with default atlas and NIfTI colors (surface mesh input):
```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --field-file subject_overlays/subject_TI_max.nii.gz
```

2) Start from tetrahedral GM mesh (auto-runs msh2cortex):
```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --gm-mesh m2m_subject/subject.msh \
  --surface central \
  --m2m m2m_subject \
  --output-dir out
```

3) Without field data (gray colors) and custom atlas:
```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --atlas HCP_MMP1 \
  --output-dir out
```

4) Global color normalization from NIfTI (comparable colors across regions):
```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --field-file subject_overlays/subject_TI_max.nii.gz \
  --global-from-nifti subject_overlays/subject_TI_max.nii.gz
```

## Complete Workflow: From Mesh to Blender

### Step-by-Step Guide

This section covers the complete pipeline from SimNIBS mesh files to visualization-ready PLY files for Blender.

#### Convert to PLY Format for Blender

```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --gm-mesh m2m_subject/subject.msh \
  --surface central \
  --m2m m2m_subject \
  --output-dir blender_meshes \
  --field-file subject_overlays/subject_TI_max.nii.gz
```

#### Step 3: Import into Blender

1. **Open Blender**
2. **Import PLY files**: File → Import → Stanford PLY (.ply)
3. **Select your PLY files** (individual or combined)
4. **View field data**: 
   - Switch viewport shading to "Vertex Paint" mode, OR
   - Use Material Preview/Rendered view with vertex color nodes

**For batch import:**
- Use Blender's File → Import → Stanford PLY (.ply) and select multiple files
- Or use Blender's command line: `blender --python your_import_script.py`

### Advanced Options

```bash
# Different colormap
--colormap plasma

# Explicit field range
--field-range 0.0 1.5

# Store as scalars instead of vertex colors
--scalars
```

#### Without Field Data

If you don't have field data, you can still visualize the region geometry (gray):

```bash
simnibs_python scripts/cortical_regions_to_ply.py \
  --gm-mesh m2m_subject/subject.msh \
  --m2m m2m_subject \
  --output-dir blender_meshes \
  --scalars
```

### Output Structure

```
project_directory/
├── out/
│   ├── regions/                     # Individual region PLYs
│   │   ├── lh.bankssts_region.ply
│   │   ├── lh.superiorfrontal_region.ply
│   │   ├── rh.bankssts_region.ply
│   │   └── ...
│   └── whole_gm.ply                 # Whole GM surface
```

## Supported Atlases

### HCP_MMP1 (Human Connectome Project Multi-Modal Parcellation)
- **Regions**: ~180 cortical areas per hemisphere
- **Based on**: Multi-modal MRI features
- **Reference**: Glasser et al. (2016) Nature

### DK40 (Desikan-Killiany 40) / DKTatlas40 (Desikan-Killiany-Tourville)
- **Regions**: ~40 cortical areas per hemisphere
- **Based on**: Structural MRI
- **Reference**: Desikan et al. (2006) NeuroImage

### aparc.a2009s (Destrieux Atlas)
- **Regions**: ~75 cortical areas per hemisphere
- **Based on**: Sulco-gyral anatomy
- **Reference**: Destrieux et al. (2010) NeuroImage

## Output Files

- **Region PLYs**: `<region_name>_region.ply` for each atlas region
- **Whole GM PLY**: `whole_gm.ply`


## Workflow Integration

1. Create head model (SimNIBS):
   ```bash
   headreco -i T1.nii.gz -o m2m_subject
   ```
2. Map to cortical surface (optional if using --gm-mesh in this tool):
   ```bash
   msh2cortex -i simulation_result.msh -m m2m_subject -o subject_overlays
   ```
3. Export regions + whole GM to PLY:
   ```bash
   simnibs_python scripts/cortical_regions_to_ply.py \
     --mesh subject_overlays/simulation_result_central.msh \
     --m2m m2m_subject \
     --output-dir blender_visualization \
     --field-file subject_overlays/simulation_result_TI_max.nii.gz
   ```

## Troubleshooting

### Common Issues

1. **"SimNIBS not found" error**
   - Ensure SimNIBS is properly installed and in your Python path
   - Try: `python -c "import simnibs; print('SimNIBS found')"`

2. **"Mesh file does not contain cortical surface triangles"**
   - Make sure your mesh file was created with `msh2cortex`
   - The input should be a surface mesh, not a volume mesh

3. **"Atlas not found" error**
   - Check that your m2m directory contains the necessary atlas files
   - Try regenerating the head model with the latest SimNIBS version

4. **Empty regions or missing vertices**
   - This is normal for some atlases - not all regions may have vertices
   - Check the summary report for details

### Performance Tips

- **Large meshes**: Processing may take several minutes for high-resolution meshes
- **Memory usage**: Each region file contains a copy of the full mesh structure
- **Parallel processing**: Use the batch processing utility for multiple subjects

## File Format Details

### Input Files
- **Mesh file**: SimNIBS .msh format (gmsh-compatible)
- **m2m directory**: Contains FreeSurfer-to-subject transformations

### Output Files
- **Region meshes**: SimNIBS .msh format with region-specific node data
- **Can be loaded** in SimNIBS GUI, ParaView, or custom analysis scripts

## Citation

If you use these scripts in your research, please cite:
- **SimNIBS**: Thielscher, A., Antunes, A., & Saturnino, G. B. (2015)
- **Relevant atlas papers** (see atlas descriptions above)

## Support

For SimNIBS-related questions, consult the official documentation:
- **SimNIBS Documentation**: https://simnibs.github.io/simnibs/
- **SimNIBS Forum**: https://simnibs.discourse.group/

For script-specific issues, please create an issue in this repository.

## Quick Reference

```bash
# Default (DK40), regions + whole GM, with field colors
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --field-file subject_overlays/subject_TI_max.nii.gz

# No field file (gray), only whole GM
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --skip-regions
```

⚠️ Always use `simnibs_python` instead of regular `python` to ensure the SimNIBS environment is active.

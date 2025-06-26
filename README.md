# SimNIBS Cortical Region Generator & Blender Converter

This repository contains Python scripts for generating cortical regions from SimNIBS mesh files using various brain atlases, and converting them to PLY format for visualization in Blender. The scripts are designed to work with mesh files produced by SimNIBS's `msh2cortex` tool.

## Overview

The main functionality includes:
- **Extract cortical regions** from SimNIBS mesh files using brain atlases
- **Convert to PLY format** for Blender visualization with field data as vertex colors
- **Combine individual regions** into collective visualization files

## Requirements

### Software Dependencies
- **SimNIBS 4.0+** (with Python API)
- **Python 3.7+**
- **NumPy**
- **nibabel** (for NIfTI field data handling)
- **matplotlib** (optional, for advanced colormaps)
- **argparse** (standard library)
- **pathlib** (standard library)

### Input Requirements
- SimNIBS mesh file (`.msh`) produced by `msh2cortex`
- Subject's `m2m_*` directory from SimNIBS head model creation
- Brain atlas (supported: HCP_MMP1, DKTatlas40, aparc.a2009s)
- Optional: NIfTI field files (e.g., `*_TI_max.nii.gz`) for field visualization

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

Generate cortical regions from a single subject:

```bash
python simnibs_cortical_regions.py -i subject_central.msh -m m2m_subject -o output_regions
```

### Command Line Options

#### Main Script (`simnibs_cortical_regions.py`)

```bash
python simnibs_cortical_regions.py [OPTIONS]

Required Arguments:
  -i, --input     Input SimNIBS mesh file (.msh) produced by msh2cortex
  -m, --m2m       Path to the m2m directory for the subject

Optional Arguments:
  -a, --atlas     Atlas name (default: HCP_MMP1)
                  Options: HCP_MMP1, DKTatlas40, aparc.a2009s
  -o, --output    Output directory (default: current directory)
  -v, --verbose   Enable verbose output
  --list-atlases  List available atlases and exit
```

### Examples

#### 1. Basic region generation with HCP_MMP1 atlas
```bash
python simnibs_cortical_regions.py \
    -i /path/to/subject_overlays/subject_TDCS_1_scalar_central.msh \
    -m /path/to/m2m_subject \
    -o cortical_regions_output
```

#### 2. Use a different atlas
```bash
python simnibs_cortical_regions.py \
    -i subject_central.msh \
    -m m2m_subject \
    -a DKTatlas40 \
    -o regions_dk40
```

#### 3. List available atlases
```bash
python simnibs_cortical_regions.py --list-atlases
```

## Complete Workflow: From Mesh to Blender

### Step-by-Step Guide

This section covers the complete pipeline from SimNIBS mesh files to visualization-ready PLY files for Blender.

#### Step 1: Generate Cortical Regions

Extract individual cortical regions from your SimNIBS mesh:

```bash
# Basic region extraction with DKTatlas40 (recommended for visualization)
simnibs_python simnibs_cortical_regions.py \
    -i your_mesh.msh \
    -m m2m_subject \
    -a DKTatlas40 \
    -o output_regions_simple
```

This creates individual `.msh` files for each cortical region in the `output_regions_simple/` directory.

#### Step 2: Convert to PLY Format for Blender

**Option A: Convert single region with field data**
```bash
# Convert one region with TI_max field visualization
simnibs_python regions_to_blender.py \
    --region output_regions_simple/lh.superiorfrontal_region.msh \
    --field-file your_TI_max_field.nii.gz \
    --output my_region.ply
```

**Option B: Batch convert all regions**
```bash
# Convert all regions to individual PLY files
simnibs_python regions_to_blender.py \
    --regions-dir output_regions_simple \
    --output-dir blender_meshes \
    --field-file your_TI_max_field.nii.gz \
    --create-blender-script
```

**Option C: Create collective PLY file**
```bash
# Create both individual PLYs and one combined file
simnibs_python combine_regions_to_ply.py \
    --regions-dir output_regions_simple \
    --field-file your_TI_max_field.nii.gz \
    --output-individual blender_meshes \
    --output-combined all_regions_combined.ply
```

#### Step 3: Import into Blender

1. **Open Blender**
2. **Import PLY files**: File → Import → Stanford PLY (.ply)
3. **Select your PLY files** (individual or combined)
4. **View field data**: 
   - Switch viewport shading to "Vertex Paint" mode, OR
   - Use Material Preview/Rendered view with vertex color nodes

**For batch import:**
- Run the generated `import_regions.py` script in Blender's text editor
- Or use: `blender --python blender_meshes/import_regions.py`

### Advanced Options

#### Custom Field Visualization

**Different colormaps:**
```bash
simnibs_python regions_to_blender.py \
    --regions-dir output_regions_simple \
    --output-dir blender_meshes \
    --field-file your_field.nii.gz \
    --colormap plasma  # Options: viridis, plasma, inferno, magma, jet
```

**Custom field range:**
```bash
simnibs_python regions_to_blender.py \
    --regions-dir output_regions_simple \
    --output-dir blender_meshes \
    --field-file your_field.nii.gz \
    --field-range 0.0 1.0  # Specify min and max values
```

**Store as scalars instead of colors:**
```bash
simnibs_python regions_to_blender.py \
    --regions-dir output_regions_simple \
    --output-dir blender_meshes \
    --field-file your_field.nii.gz \
    --scalars  # Store field values as vertex attributes
```

#### Without Field Data

If you don't have field data, you can still visualize the region geometry:

```bash
# Convert regions without field data (gray meshes)
simnibs_python regions_to_blender.py \
    --regions-dir output_regions_simple \
    --output-dir blender_meshes
```

### Output Structure

After running the complete workflow, you'll have:

```
project_directory/
├── output_regions_simple/           # Individual region mesh files
│   ├── lh.bankssts_region.msh
│   ├── lh.superiorfrontal_region.msh
│   ├── rh.bankssts_region.msh
│   └── ...
├── blender_meshes/                  # Individual PLY files for Blender
│   ├── lh.bankssts_region.ply
│   ├── lh.superiorfrontal_region.ply
│   ├── rh.bankssts_region.ply
│   ├── import_regions.py            # Auto-generated Blender script
│   └── ...
└── all_regions_combined.ply         # Single file with all regions
```

## Supported Atlases

### HCP_MMP1 (Human Connectome Project Multi-Modal Parcellation)
- **Regions**: ~180 cortical areas per hemisphere
- **Based on**: Multi-modal MRI features
- **Reference**: Glasser et al. (2016) Nature

### DKTatlas40 (Desikan-Killiany-Tourville)
- **Regions**: ~40 cortical areas per hemisphere
- **Based on**: Structural MRI
- **Reference**: Desikan et al. (2006) NeuroImage

### aparc.a2009s (Destrieux Atlas)
- **Regions**: ~75 cortical areas per hemisphere
- **Based on**: Sulco-gyral anatomy
- **Reference**: Destrieux et al. (2010) NeuroImage

## Output Files

### Generated Region Files
For each cortical region, the script generates:
- **`<region_name>_region.msh`**: Individual region mesh file (SimNIBS format)
- **`<region_name>_region.ply`**: Individual region PLY file (Blender format)
- Contains the original mesh with region-specific node fields and field visualization

### Summary Files
- **`cortical_regions_summary.txt`**: Text summary of all generated regions
- **`regions_summary.csv`**: CSV table with region statistics (optional)
- **`import_regions.py`**: Auto-generated Blender import script

### Blender-Ready Files
- **Individual PLY files**: Each region as a separate Blender-importable file
- **Combined PLY file**: All regions merged into a single visualization file
- **Vertex colors**: Field data (e.g., TI_max) mapped to vertex colors using configurable colormaps
- **Blender import script**: Automated batch import with proper naming and color settings

### Example Output Structure
```
project_directory/
├── region_meshes/                   # SimNIBS mesh files
│   ├── lh.area1_region.msh
│   ├── lh.area2_region.msh
│   ├── rh.area1_region.msh
│   ├── ...
│   └── cortical_regions_summary.txt
├── blender_meshes/                  # Blender-ready PLY files
│   ├── lh.area1_region.ply         # Individual regions with field colors
│   ├── lh.area2_region.ply
│   ├── rh.area1_region.ply
│   ├── ...
│   └── import_regions.py           # Batch import script
└── all_regions_combined.ply        # Single combined file
```

## Workflow Integration

### Typical SimNIBS Workflow

1. **Create head model** using SimNIBS:
   ```bash
   # Using SimNIBS GUI or command line
   headreco -i T1.nii.gz -o m2m_subject
   ```

2. **Run simulation** (TMS/tDCS):
   ```bash
   # Your simulation script here
   simnibs_python your_simulation.py
   ```

3. **Map to cortical surface**:
   ```bash
   msh2cortex -i simulation_result.msh -m m2m_subject -o subject_overlays
   ```

4. **Generate cortical regions**:
   ```bash
   simnibs_python simnibs_cortical_regions.py \
       -i subject_overlays/simulation_result_central.msh \
       -m m2m_subject \
       -o cortical_regions
   ```

5. **Convert to PLY for Blender visualization**:
   ```bash
   simnibs_python regions_to_blender.py \
       --regions-dir cortical_regions \
       --output-dir blender_visualization \
       --field-file subject_overlays/simulation_result_TI_max.nii.gz \
       --create-blender-script
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

### Essential Commands

**Generate cortical regions:**
```bash
simnibs_python simnibs_cortical_regions.py -i mesh.msh -m m2m_subject -a DKTatlas40 -o regions
```

**Convert to Blender (individual PLYs):**
```bash
simnibs_python regions_to_blender.py --regions-dir regions --output-dir ply_files --field-file field.nii.gz
```

**Convert single region:**
```bash
simnibs_python regions_to_blender.py --region regions/lh.area.msh --field-file field.nii.gz --output region.ply
```

### Script Files

- **`simnibs_cortical_regions.py`**: Main region extraction script
- **`simnibs_region_utils.py`**: Utility functions (batch processing, analysis)
- **`regions_to_blender.py`**: Convert regions to PLY format with field visualization
- **`mesh_to_ply_converter.py`**: General mesh-to-PLY conversion utilities

### Required with simnibs_python

⚠️ **Important**: Always use `simnibs_python` instead of regular `python` to ensure proper SimNIBS environment activation. 
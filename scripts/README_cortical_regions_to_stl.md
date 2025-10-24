# SimNIBS Cortical Mesh → STL (Regions + Whole GM)

This repository provides a single tool to export subject‑specific cortical regions and the whole gray‑matter (GM) surface from SimNIBS `.msh` files to binary STL format for Blender (or other 3D software). It uses the subject's atlas from the `m2m_*` directory to ensure region accuracy (default atlas: `DK40`).

## Overview

The single script `scripts/cortical_regions_to_stl.py`:
- **Exports individual cortical regions to binary STL** using the chosen atlas (default `DK40`)
- **Exports the whole GM surface to binary STL**
- **Supports cortical surface generation** from tetrahedral GM meshes using `msh2cortex`
- **Note**: STL format does not support vertex colors or field data - only geometry is exported

## Requirements

### Software Dependencies
- **SimNIBS 4.0+** (with Python API)
- **Python 3.7+**
- **NumPy**

### Input Requirements
- EITHER a cortical surface mesh (`.msh`) produced by `msh2cortex` OR a tetrahedral GM `.msh` (the tool can run `msh2cortex` for you)
- Subject's `m2m_*` directory (for the subject atlas)
- Supported atlases: `DK40` (default), `DKTatlas40`, `HCP_MMP1`, `aparc.a2009s`

## Usage

### Basic Usage

Export atlas‑accurate region STLs and the whole GM STL (default atlas: `DK40`):

```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out
```

### Command Line Options

```bash
simnibs_python scripts/cortical_regions_to_stl.py [OPTIONS]

Required (one of):
  --mesh           Cortical surface .msh (from msh2cortex)
  --gm-mesh        Tetrahedral GM .msh (the tool will run msh2cortex)
  --m2m            Subject m2m directory
  --output-dir     Output directory

Optional:
  --atlas          Atlas name (default: DK40)
  --surface        Surface when using --gm-mesh: central|pial|white (default: central)
  --msh2cortex     Path to msh2cortex executable (if not on PATH)
  --min-triangles  Minimum number of triangles required for a region (default: 10)
  --segmentation-method  Segmentation method: connected (recommended) or original (default: connected)
  --skip-regions   Do not export individual region STLs
  --skip-whole-gm  Do not export the whole GM STL
  --keep-meshes    Keep individual cortical region meshes as .msh files
  --debug          Enable debug logging for troubleshooting
```

### Examples

1) Regions + whole GM with default atlas (surface mesh input):
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out
```

2) Start from tetrahedral GM mesh (auto-runs msh2cortex):
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --gm-mesh m2m_subject/subject.msh \
  --surface central \
  --m2m m2m_subject \
  --output-dir out
```

3) Custom atlas and keep original meshes:
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --atlas HCP_MMP1 \
  --output-dir out \
  --keep-meshes
```

4) Export only whole GM (skip individual regions):
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --skip-regions
```

5) Filter out small fragmented regions:
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --min-triangles 50
```

6) Debug mode for troubleshooting:
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --debug
```

7) Use original segmentation method (if connected method fails):
```bash
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_overlays/subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --segmentation-method original
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

- **Region STLs**: `<region_name>_region.stl` for each atlas region
- **Whole GM STL**: `whole_gm.stl`
- **Optional MSH files**: `<region_name>_region.msh` and `whole_gm.msh` (if `--keep-meshes`)

## STL Format Notes

- **Binary STL format**: Compact, efficient for 3D software
- **No vertex colors**: STL format only supports geometry (vertices and faces)
- **Surface normals**: Automatically calculated for proper lighting in 3D software
- **Blender compatibility**: Direct import support in Blender and most 3D software

## Usage Tips

- **Use `--keep-meshes`** to preserve original `.msh` files alongside STL exports
- **Skip regions or whole GM** with `--skip-regions` or `--skip-whole-gm` for faster processing
- **Binary STL files** are much smaller than ASCII STL and load faster in 3D software
- **For field data visualization**, use `cortical_regions_to_ply.py` instead, which supports vertex colors
- **Filter fragmented regions** with `--min-triangles` to exclude small, disconnected mesh fragments
- **Quality validation** automatically removes degenerate triangles and validates mesh connectivity

## Segmentation Methods

### Connected Component Method (Default)

The **connected component method** is the recommended approach that solves fragmentation issues:

1. **Strict Triangle Inclusion**: Only includes triangles where ALL 3 vertices are in the region
2. **Connected Component Analysis**: Uses graph theory to find the largest connected component
3. **Clean Boundaries**: Eliminates scattered faces and ensures proper region boundaries
4. **Automatic Fallback**: Falls back to original method if connected analysis fails

### Original Method

The **original method** uses the same approach as the PLY script:

1. **Node-based Cropping**: Uses SimNIBS's built-in `crop_mesh(nodes=nodes_to_keep)`
2. **Element Filtering**: Fallback to triangle filtering with ≥2 vertices in region
3. **May Produce Fragmentation**: Can result in scattered faces across region boundaries

## Troubleshooting

### Fragmentation Issues

If you experience fragmented STL files with faces from all over the cortex:

1. **Use the connected method** (default) - this should solve most fragmentation issues
2. **Use `--debug`** to see detailed extraction information
3. **Check the whole GM export first** - if this is also fragmented, the issue is with the input mesh
4. **Use `--min-triangles`** to filter out regions with too few valid triangles
5. **Try `--segmentation-method original`** if the connected method fails

### Debugging Steps

1. **Test whole GM export first**:
   ```bash
   simnibs_python scripts/cortical_regions_to_stl.py \
     --mesh your_mesh.msh \
     --m2m your_m2m_dir \
     --output-dir out \
     --skip-regions \
     --debug
   ```

2. **Check mesh statistics** in the debug output - verify you have triangular elements

3. **Test with a single region** by using `--min-triangles 0` to see all regions

4. **Verify atlas alignment** - the atlas nodes should correspond to your mesh nodes

### Common Issues

- **Mixed element types**: Ensure your mesh contains triangular surface elements (elm_type == 2)
- **Atlas mismatch**: The atlas must be generated for the same subject as your mesh
- **Surface mesh required**: Use `msh2cortex` to generate proper cortical surface meshes from tetrahedral meshes

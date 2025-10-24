# SimNIBS Mesh Region Extractor

Tools for exporting SimNIBS mesh data to PLY and STL formats for visualization in Blender and other 3D software. Provides atlas-accurate cortical region extraction with preserved field data and consistent visualization.
<img width="1270" height="644" alt="gmsh-ss" src="https://github.com/user-attachments/assets/7ebf77d9-76ca-4ed9-af74-57b5ca2300e2" />
<img width="1050" height="575" alt="blender-nonrender" src="https://github.com/user-attachments/assets/1e387f22-51b7-49d9-9b57-8801c66e1c35" />
<img width="1920" height="1080" alt="blender-render" src="https://github.com/user-attachments/assets/407a9bea-5d1f-4b32-a06b-b412ddb2da53" />

## Scripts

### `scripts/cortical_regions_to_ply.py`
Export atlas-accurate cortical regions and whole gray matter surface from SimNIBS `.msh` files to PLY format with field data visualization.

**Features:**
- Individual cortical region PLY files with preserved field values
- Whole gray matter surface PLY
- Field data mapping to vertex colors or scalars
- Global color scaling for consistent visualization across regions
- Support for multiple atlases (DK40, HCP_MMP1, DKTatlas40, aparc.a2009s)

### `scripts/cortical_regions_to_stl.py`
Export atlas-accurate cortical regions from SimNIBS `.msh` files to binary STL format for Blender.

**Features:**
- Individual cortical region STL files (geometry only)
- Automatic surface mesh generation from tetrahedral meshes
- ROI extraction with preserved field values
- Support for multiple atlases

### `scripts/vector_ply.py`
Export TDCS simulation E-field vectors as arrow PLY files for visualization of CH1, CH2, TI/mTI patterns.

**Features:**
- CH1, CH2, TI/mTI vector visualization
- Optional SUM and TI_normal projections
- Configurable arrow styling and coloring
- Support for both TI and mTI modes

## Quick Start

```bash
# Export cortical regions to PLY (with field data)
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --field-file subject_TI_max.nii.gz

# Export cortical regions to STL (geometry only)
simnibs_python scripts/cortical_regions_to_stl.py \
  --mesh subject_central.msh \
  --m2m m2m_subject \
  --output-dir out \
  --atlas DK40

# Export TDCS vector fields
simnibs_python scripts/vector_ply.py \
  tdcs1.msh tdcs2.msh output/TI
```

## Output Structure

### PLY Script Output
```
{output_dir}/
└── cortical_plys/
    ├── regions/
    │   ├── region1_region.ply
    │   ├── region2_region.ply
    │   └── ...
    └── whole_gm.ply
```

### STL Script Output
```
{output_dir}/
└── cortical_stls/
    └── {atlas_type}/  (e.g., DK40/)
        ├── region1.stl
        ├── region2.stl
        └── ...
```

### Vector PLY Output
```
{output_prefix}_CH1.ply
{output_prefix}_CH2.ply
{output_prefix}_TI.ply (or mTI.ply)
{output_prefix}_SUM.ply (optional)
{output_prefix}_TI_normal.ply (optional)
```

## Requirements

- **SimNIBS 4.0+** with Python API
- **Python 3.7+**
- **Required packages**: numpy, nibabel
- **Optional packages**: trimesh, scipy, matplotlib (for advanced features)

## Supported Atlases

- **DK40** (Desikan-Killiany 40) - ~40 regions per hemisphere
- **DKTatlas40** (Desikan-Killiany-Tourville) - ~40 regions per hemisphere  
- **HCP_MMP1** (Human Connectome Project) - ~180 regions per hemisphere
- **aparc.a2009s** (Destrieux Atlas) - ~75 regions per hemisphere

## Key Features

- **Atlas-accurate extraction**: Uses subject-specific atlases from m2m directories
- **Field data preservation**: Maintains original field values in ROI regions
- **Consistent visualization**: Global color scaling ensures comparable visualization across regions
- **Flexible input**: Supports both surface meshes and tetrahedral meshes (with automatic surface generation)
- **Progress feedback**: Clean progress messages ("Starting...", "Converting...", "Finishing...")

See individual script READMEs for detailed usage, examples, and advanced options.

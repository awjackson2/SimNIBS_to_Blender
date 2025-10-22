# SimNIBS Mesh Region Extractor

Tools for exporting SimNIBS mesh data to PLY format for visualization in Blender and other 3D software.

## Scripts

### `scripts/cortical_regions_to_ply.py`
Export atlas-accurate cortical regions and whole gray matter surface from SimNIBS `.msh` files to PLY format.

### `scripts/vector_ply.py`
Export TDCS simulation E-field vectors as arrow PLY files for visualization of CH1, CH2, TI/mTI patterns.

## Quick Start

```bash
# Export cortical regions
simnibs_python scripts/cortical_regions_to_ply.py \
  --mesh subject_central.msh \
  --m2m m2m_subject \
  --output-dir out

# Export TDCS vector fields
simnibs_python scripts/vector_ply.py \
  tdcs1.msh tdcs2.msh output/TI
```

## Requirements

- SimNIBS 4.0+ with Python API
- Python packages: numpy, nibabel, trimesh, scipy
- matplotlib (optional, for colormaps)

See individual script READMEs for detailed usage and examples.

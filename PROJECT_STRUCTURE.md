# Project Structure

## Directory Organization

```
regions_with_field/
├── README.md                    # Complete documentation and usage guide
├── PROJECT_STRUCTURE.md         # This file - project organization guide
├── data/                        # Input data files
│   ├── input_meshes/            # Original SimNIBS mesh files (.msh)
│   ├── field_data/              # NIfTI field files (.nii.gz) 
│   └── m2m_directories/         # SimNIBS m2m subject directories
├── scripts/                     # All Python scripts
│   ├── simnibs_cortical_regions.py    # Main region extraction script
│   ├── simnibs_region_utils.py        # Utility functions
│   ├── regions_to_blender.py          # Convert regions to PLY
│   ├── mesh_to_ply_converter.py       # General mesh conversion
│   └── inspect_mesh.py                # Mesh analysis tool
├── output/                      # Generated output files
│   ├── region_meshes/           # Individual cortical region .msh files
│   └── blender_files/           # PLY files for Blender with import script
├── examples/                    # Example usage scripts and data
└── temp/                        # Temporary files and test outputs
```

## Usage from New Structure

### Generate Regions
```bash
simnibs_python scripts/simnibs_cortical_regions.py \
    -i data/input_meshes/your_mesh.msh \
    -m data/m2m_directories/m2m_subject \
    -o output/region_meshes
```

### Convert to Blender
```bash
simnibs_python scripts/regions_to_blender.py \
    --regions-dir output/region_meshes \
    --field-file data/field_data/your_field.nii.gz \
    --output-dir output/blender_files \
    --create-blender-script
```

## File Types

- **Input Data**: Original SimNIBS meshes, NIfTI fields, m2m directories
- **Scripts**: All Python tools for processing and conversion
- **Outputs**: Generated region meshes and Blender-ready PLY files
- **Temp**: Testing files, intermediate outputs, temporary data


#!/usr/bin/env python3
"""
Convert cortical region meshes with TI_max fields to PLY format for Blender.

This script is specifically designed to work with the output from simnibs_cortical_regions.py
and convert the regional meshes with their TI_max field data to PLY format for import into Blender.

Usage:
    python regions_to_blender.py --regions-dir output_regions_simple --output-dir blender_meshes
    python regions_to_blender.py --field-file 003_A_TI_TI_max.nii.gz --regions-dir output_regions_simple --output-dir blender_meshes
    python regions_to_blender.py --region lh.superiorfrontal_region.msh --field-file 003_A_TI_TI_max.nii.gz --output region.ply
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
import logging

try:
    import simnibs
    from simnibs import read_msh
    import nibabel as nib
except ImportError:
    print("Error: SimNIBS and/or nibabel not found. Please install SimNIBS and activate the environment.")
    sys.exit(1)

from mesh_to_ply_converter import (
    write_ply_with_colors, write_ply_with_scalars, 
    field_to_colormap, simple_colormap, convert_mesh_to_ply
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_field_from_nifti(mesh, nifti_path, field_name="TI_max"):
    """
    Apply field data from a NIfTI file to mesh nodes.
    
    Parameters:
    -----------
    mesh : simnibs.Msh
        SimNIBS mesh object
    nifti_path : str
        Path to NIfTI file containing field data
    field_name : str
        Name for the field
        
    Returns:
    --------
    simnibs.Msh
        Mesh with field data added
    """
    logger.info(f"Loading field data from {nifti_path}")
    
    try:
        # Load NIfTI file
        nii = nib.load(nifti_path)
        field_data = nii.get_fdata()
        affine = nii.affine
        
        logger.info(f"NIfTI shape: {field_data.shape}, affine: {affine.shape}")
        
        # Get mesh node coordinates
        node_coords = mesh.nodes.node_coord
        
        # Transform mesh coordinates to voxel indices
        # Convert from RAS (SimNIBS) to voxel coordinates
        ones = np.ones((len(node_coords), 1))
        coords_homog = np.hstack([node_coords, ones])
        
        # Apply inverse affine transformation
        inv_affine = np.linalg.inv(affine)
        voxel_coords = (inv_affine @ coords_homog.T).T[:, :3]
        
        # Round to nearest voxel indices
        voxel_indices = np.round(voxel_coords).astype(int)
        
        # Extract field values at mesh node locations
        field_values = np.zeros(len(node_coords))
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < field_data.shape[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < field_data.shape[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < field_data.shape[2])
        )
        
        logger.info(f"Found {np.sum(valid_mask)}/{len(node_coords)} nodes within field volume")
        
        # Sample field data at valid locations
        valid_indices = voxel_indices[valid_mask]
        field_values[valid_mask] = field_data[
            valid_indices[:, 0], 
            valid_indices[:, 1], 
            valid_indices[:, 2]
        ]
        
        # Add field to mesh
        from simnibs.mesh_tools.mesh_io import NodeData
        nodedata = NodeData(field_values, field_name)
        mesh.add_node_field(nodedata, field_name)
        
        logger.info(f"Added field '{field_name}' with range [{np.min(field_values):.6f}, {np.max(field_values):.6f}]")
        
        return mesh
        
    except Exception as e:
        logger.error(f"Failed to apply field from NIfTI: {e}")
        return mesh


def convert_region_with_field(region_mesh_path, field_file_path, output_path, 
                             field_name="TI_max", **kwargs):
    """
    Convert a single cortical region mesh with field data to PLY format.
    
    Parameters:
    -----------
    region_mesh_path : str
        Path to regional mesh file
    field_file_path : str
        Path to NIfTI field file
    output_path : str
        Output PLY file path
    field_name : str
        Name of the field
    **kwargs : dict
        Additional arguments for convert_mesh_to_ply
    """
    logger.info(f"Processing region: {region_mesh_path}")
    
    # Read the regional mesh
    try:
        mesh = read_msh(region_mesh_path)
        logger.info(f"Loaded regional mesh with {len(mesh.nodes.node_coord)} nodes")
    except Exception as e:
        logger.error(f"Failed to read mesh {region_mesh_path}: {e}")
        return False
    
    # Apply field data if provided
    if field_file_path and os.path.exists(field_file_path):
        mesh = apply_field_from_nifti(mesh, field_file_path, field_name)
    
    # Write the mesh to a temporary file and then convert it
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as tmp_file:
        temp_mesh_path = tmp_file.name
        mesh.write(temp_mesh_path)
    
    try:
        # Convert to PLY format
        # For regional meshes, we want all triangles (no tissue filtering)
        kwargs['tissue_types'] = None  # Include all triangles
        
        result = convert_mesh_to_ply(temp_mesh_path, output_path, field_name, **kwargs)
    finally:
        # Clean up temporary file
        os.unlink(temp_mesh_path)
    
    return result


def batch_convert_regions(regions_dir, output_dir, field_file_path=None, 
                         field_name="TI_max", pattern="*_region.msh", **kwargs):
    """
    Batch convert all cortical region meshes to PLY format.
    
    Parameters:
    -----------
    regions_dir : str
        Directory containing regional mesh files
    output_dir : str
        Output directory for PLY files
    field_file_path : str
        Path to NIfTI field file
    field_name : str
        Name of the field
    pattern : str
        Glob pattern for regional mesh files
    **kwargs : dict
        Additional arguments for conversion
    """
    regions_path = Path(regions_dir)
    output_path = Path(output_dir)
    
    if not regions_path.exists():
        logger.error(f"Regions directory does not exist: {regions_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all regional mesh files
    region_files = list(regions_path.glob(pattern))
    
    if not region_files:
        logger.warning(f"No regional mesh files found with pattern '{pattern}' in {regions_dir}")
        return
    
    logger.info(f"Found {len(region_files)} regional mesh files to convert")
    
    success_count = 0
    for region_file in region_files:
        output_file = output_path / (region_file.stem + ".ply")
        
        if convert_region_with_field(str(region_file), field_file_path, str(output_file), 
                                    field_name, **kwargs):
            success_count += 1
    
    logger.info(f"Successfully converted {success_count}/{len(region_files)} regional meshes")


def create_blender_import_script(output_dir, script_name="import_regions.py"):
    """
    Create a Blender Python script for importing all PLY files.
    """
    script_path = Path(output_dir) / script_name
    
    script_content = '''#!/usr/bin/env python3
"""
Blender script to import all cortical region PLY files.

Run this script in Blender to import all the converted cortical regions.
You can either:
1. Open this script in Blender's text editor and run it
2. Run from command line: blender --python import_regions.py
"""

import bpy
import bmesh
from pathlib import Path
import os

def import_region_ply(ply_path):
    """Import a single PLY file as a mesh object."""
    try:
        # Import PLY file
        bpy.ops.import_mesh.ply(filepath=str(ply_path))
        
        # Get the imported object (last selected)
        obj = bpy.context.selected_objects[-1]
        
        # Set object name based on filename
        obj.name = ply_path.stem
        
        # Enable vertex colors if available
        if obj.data.vertex_colors:
            # Switch to material preview or rendered view to see colors
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'MATERIAL_PREVIEW'
                            break
        
        print(f"Imported: {obj.name}")
        return obj
        
    except Exception as e:
        print(f"Failed to import {ply_path}: {e}")
        return None

def main():
    """Import all PLY files in the current directory."""
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Get directory of this script
    script_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
    
    # Find all PLY files
    ply_files = list(script_dir.glob("*.ply"))
    
    if not ply_files:
        print("No PLY files found in directory")
        return
    
    print(f"Found {len(ply_files)} PLY files to import")
    
    imported_count = 0
    for ply_file in ply_files:
        if import_region_ply(ply_file):
            imported_count += 1
    
    print(f"Successfully imported {imported_count}/{len(ply_files)} regions")
    
    # Fit view to show all objects
    bpy.ops.view3d.view_all()

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created Blender import script: {script_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert cortical region meshes to PLY format for Blender")
    
    parser.add_argument("--regions-dir", help="Directory containing regional mesh files")
    parser.add_argument("--output-dir", help="Output directory for PLY files")
    parser.add_argument("--region", help="Single regional mesh file to convert")
    parser.add_argument("--output", help="Output PLY file (for single region conversion)")
    parser.add_argument("--field-file", help="NIfTI file containing field data (e.g., TI_max)")
    parser.add_argument("--field", default="TI_max", help="Field name (default: TI_max)")
    parser.add_argument("--pattern", default="*_region.msh", help="Glob pattern for regional mesh files")
    parser.add_argument("--scalars", action="store_true", help="Store field as scalars instead of colors")
    parser.add_argument("--colormap", default="viridis", help="Colormap for field visualization")
    parser.add_argument("--field-range", nargs=2, type=float, metavar=("MIN", "MAX"), 
                       help="Field value range for color mapping")
    parser.add_argument("--create-blender-script", action="store_true", 
                       help="Create Blender Python import script")
    
    args = parser.parse_args()
    
    if not args.regions_dir and not args.region:
        parser.error("Must specify either --regions-dir for batch conversion or --region for single file")
    
    if args.region and not args.output:
        parser.error("Must specify --output when using --region")
    
    kwargs = {
        'field_name': args.field,
        'use_colors': not args.scalars,
        'colormap': args.colormap,
        'field_range': tuple(args.field_range) if args.field_range else None,
    }
    
    if args.region:
        # Single region conversion
        success = convert_region_with_field(args.region, args.field_file, args.output, **kwargs)
        if success:
            logger.info("Region conversion completed successfully")
        else:
            logger.error("Region conversion failed")
    else:
        # Batch conversion
        batch_convert_regions(args.regions_dir, args.output_dir, args.field_file, 
                             pattern=args.pattern, **kwargs)
        
        if args.create_blender_script:
            create_blender_import_script(args.output_dir)
    
    print("\nConversion complete! To import into Blender:")
    print("1. Open Blender")
    print("2. File -> Import -> Stanford PLY (.ply)")
    print("3. Select your PLY files")
    print("4. In Material Properties, set Viewport Display to Vertex Color to see field data")


if __name__ == "__main__":
    main() 
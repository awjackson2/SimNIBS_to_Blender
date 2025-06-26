#!/usr/bin/env python3
"""
Convert cortical regions to PLY with global colormap scaling.

This script ensures all regions use the same colormap range based on the 
global min/max values from the original field data, making colors comparable 
across all regions.

Usage:
    simnibs_python global_colormap_converter.py \
        --regions-dir output/region_meshes \
        --field-file data/field_data/003_A_TI_TI_max.nii.gz \
        --output-dir output/blender_files_global \
        --create-blender-script
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

# Import our existing conversion functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regions_to_blender import (
    apply_field_from_nifti, convert_region_with_field, 
    create_blender_import_script
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_global_field_range(field_file_path, mesh_file_path=None):
    """
    Calculate the global min/max values from the field data.
    
    Parameters:
    -----------
    field_file_path : str
        Path to NIfTI field file
    mesh_file_path : str, optional
        Path to original mesh file to get field range from mesh data
        
    Returns:
    --------
    tuple : (global_min, global_max)
    """
    logger.info(f"Calculating global field range from {field_file_path}")
    
    try:
        # Method 1: From NIfTI file directly
        nii = nib.load(field_file_path)
        field_data = nii.get_fdata()
        
        # Remove zeros and invalid values
        valid_data = field_data[field_data > 0]
        
        global_min = np.min(valid_data)
        global_max = np.max(valid_data)
        
        logger.info(f"Global field range from NIfTI: [{global_min:.6f}, {global_max:.6f}]")
        
        # Method 2: If we have a mesh file, check if it has field data too
        if mesh_file_path and os.path.exists(mesh_file_path):
            try:
                mesh = read_msh(mesh_file_path)
                if hasattr(mesh, 'nodedata') and len(mesh.nodedata) > 0:
                    for nodedata in mesh.nodedata:
                        if hasattr(nodedata, 'field_name'):
                            mesh_values = nodedata.value
                            mesh_min = np.min(mesh_values[mesh_values > 0])
                            mesh_max = np.max(mesh_values)
                            logger.info(f"Field range from mesh '{nodedata.field_name}': [{mesh_min:.6f}, {mesh_max:.6f}]")
                            
                            # Use mesh values if they seem more restrictive (more realistic for cortical data)
                            if mesh_min > global_min:
                                global_min = mesh_min
                            if mesh_max < global_max:
                                global_max = mesh_max
            except Exception as e:
                logger.warning(f"Could not read field data from mesh: {e}")
        
        return global_min, global_max
        
    except Exception as e:
        logger.error(f"Failed to calculate global field range: {e}")
        return None, None


def convert_regions_with_global_colormap(regions_dir, field_file_path, output_dir, 
                                       field_name="TI_max", mesh_file_path=None,
                                       colormap='viridis', **kwargs):
    """
    Convert all regions using a global colormap range.
    
    Parameters:
    -----------
    regions_dir : str
        Directory containing regional mesh files
    field_file_path : str
        Path to NIfTI field file
    output_dir : str
        Output directory for PLY files
    field_name : str
        Name of the field
    mesh_file_path : str, optional
        Path to original mesh file for global range calculation
    colormap : str
        Colormap name
    **kwargs : dict
        Additional arguments
    """
    # Calculate global field range
    global_min, global_max = calculate_global_field_range(field_file_path, mesh_file_path)
    
    if global_min is None or global_max is None:
        logger.error("Could not determine global field range")
        return False
    
    logger.info(f"Using global colormap range: [{global_min:.6f}, {global_max:.6f}]")
    
    # Create output directory
    regions_path = Path(regions_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all regional mesh files
    region_files = list(regions_path.glob("*_region.msh"))
    
    if not region_files:
        logger.warning(f"No regional mesh files found in {regions_dir}")
        return False
    
    logger.info(f"Converting {len(region_files)} regions with global colormap")
    
    # Set the global field range for all conversions
    kwargs['field_range'] = (global_min, global_max)
    kwargs['colormap'] = colormap
    
    success_count = 0
    for region_file in region_files:
        output_file = output_path / (region_file.stem + ".ply")
        
        logger.info(f"Converting {region_file.name}...")
        
        if convert_region_with_field(str(region_file), field_file_path, str(output_file), 
                                   field_name, **kwargs):
            success_count += 1
        else:
            logger.warning(f"Failed to convert {region_file.name}")
    
    logger.info(f"Successfully converted {success_count}/{len(region_files)} regions")
    
    # Create summary file with global range info
    summary_file = output_path / "global_colormap_info.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Global Colormap Conversion Summary\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Field file: {field_file_path}\n")
        f.write(f"Global range: [{global_min:.6f}, {global_max:.6f}]\n")
        f.write(f"Colormap: {colormap}\n")
        f.write(f"Regions converted: {success_count}/{len(region_files)}\n")
        f.write(f"Field name: {field_name}\n\n")
        f.write(f"All regions use the same color scaling for comparative visualization.\n")
        f.write(f"Blue represents values near {global_min:.6f}\n")
        f.write(f"Red represents values near {global_max:.6f}\n")
    
    logger.info(f"Created summary: {summary_file}")
    
    return success_count > 0


def analyze_field_distribution(field_file_path, output_dir):
    """
    Analyze and visualize the field value distribution.
    """
    try:
        nii = nib.load(field_file_path)
        field_data = nii.get_fdata()
        valid_data = field_data[field_data > 0]
        
        # Calculate statistics
        stats = {
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'percentile_5': np.percentile(valid_data, 5),
            'percentile_95': np.percentile(valid_data, 95),
            'total_voxels': len(valid_data)
        }
        
        # Write analysis file
        analysis_file = Path(output_dir) / "field_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write("Field Data Analysis\n")
            f.write("=" * 20 + "\n\n")
            for key, value in stats.items():
                if 'voxels' in key:
                    f.write(f"{key:>15}: {value:,}\n")
                else:
                    f.write(f"{key:>15}: {value:.6f}\n")
        
        logger.info(f"Field analysis saved to: {analysis_file}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to analyze field distribution: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert cortical regions with global colormap scaling")
    
    parser.add_argument("--regions-dir", required=True, help="Directory containing regional mesh files")
    parser.add_argument("--field-file", required=True, help="NIfTI field file for global range calculation")
    parser.add_argument("--output-dir", required=True, help="Output directory for PLY files")
    parser.add_argument("--mesh-file", help="Original mesh file (optional, for better range estimation)")
    parser.add_argument("--field", default="TI_max", help="Field name (default: TI_max)")
    parser.add_argument("--colormap", default="viridis", help="Colormap (default: viridis)")
    parser.add_argument("--analyze", action="store_true", help="Create field distribution analysis")
    parser.add_argument("--create-blender-script", action="store_true", help="Create Blender import script")
    parser.add_argument("--scalars", action="store_true", help="Store as scalars instead of colors")
    
    args = parser.parse_args()
    
    # Check required inputs
    if not os.path.exists(args.regions_dir):
        logger.error(f"Regions directory not found: {args.regions_dir}")
        return 1
    
    if not os.path.exists(args.field_file):
        logger.error(f"Field file not found: {args.field_file}")
        return 1
    
    # Analyze field distribution if requested
    if args.analyze:
        analyze_field_distribution(args.field_file, args.output_dir)
    
    # Convert regions with global colormap
    kwargs = {
        'use_colors': not args.scalars,
        'tissue_types': None  # Include all triangles for regional meshes
    }
    
    success = convert_regions_with_global_colormap(
        regions_dir=args.regions_dir,
        field_file_path=args.field_file,
        output_dir=args.output_dir,
        field_name=args.field,
        mesh_file_path=args.mesh_file,
        colormap=args.colormap,
        **kwargs
    )
    
    if not success:
        logger.error("Conversion failed")
        return 1
    
    # Create Blender import script
    if args.create_blender_script:
        create_blender_import_script(args.output_dir)
    
    print(f"\n{'='*60}")
    print("GLOBAL COLORMAP CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ All regions now use the same colormap scaling")
    print(f"✓ Colors are directly comparable across regions")
    print(f"✓ Output directory: {args.output_dir}")
    print(f"✓ Check 'global_colormap_info.txt' for range details")
    
    if args.create_blender_script:
        print(f"✓ Blender import script: {args.output_dir}/import_regions.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
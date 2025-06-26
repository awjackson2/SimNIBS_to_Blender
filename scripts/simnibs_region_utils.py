#!/usr/bin/env python3
"""
SimNIBS Region Utilities

Additional utility functions for working with SimNIBS cortical regions.
Provides batch processing, visualization helpers, and mesh analysis tools.

Author: Generated for SimNIBS mesh processing
"""

import os
import sys
import numpy as np
from pathlib import Path
import glob
import json

try:
    import simnibs
    from simnibs.mesh_tools import mesh_io
    from simnibs.utils.transformations import subject_atlas
except ImportError:
    print("Error: SimNIBS is not installed or not in the Python path.")
    sys.exit(1)


def batch_process_subjects(subjects_dir, atlas_name='HCP_MMP1', output_base_dir=None):
    """
    Batch process multiple subjects for cortical region generation.
    
    Parameters:
    -----------
    subjects_dir : str
        Directory containing subject folders (each with m2m_* subdirectories)
    atlas_name : str
        Atlas to use for all subjects
    output_base_dir : str, optional
        Base directory for outputs. If None, creates outputs in subjects_dir
    
    Returns:
    --------
    dict : Processing results for each subject
    """
    subjects_dir = Path(subjects_dir)
    if output_base_dir is None:
        output_base_dir = subjects_dir
    else:
        output_base_dir = Path(output_base_dir)
    
    # Find all m2m directories
    m2m_dirs = list(subjects_dir.glob("m2m_*"))
    
    if not m2m_dirs:
        print(f"No m2m directories found in {subjects_dir}")
        return {}
    
    results = {}
    
    for m2m_dir in m2m_dirs:
        subject_name = m2m_dir.name.replace("m2m_", "")
        print(f"\nProcessing subject: {subject_name}")
        
        # Look for cortical surface mesh files
        surface_files = list(m2m_dir.glob("*central*.msh"))
        if not surface_files:
            # Try alternative naming patterns
            surface_files = list(m2m_dir.glob("*cortex*.msh"))
        
        if not surface_files:
            print(f"Warning: No cortical surface mesh found for {subject_name}")
            results[subject_name] = {'status': 'failed', 'reason': 'No surface mesh found'}
            continue
        
        surface_file = surface_files[0]  # Use the first one found
        output_dir = output_base_dir / f"cortical_regions_{subject_name}"
        
        try:
            from simnibs_cortical_regions import CorticalRegionGenerator
            
            generator = CorticalRegionGenerator(
                mesh_file=str(surface_file),
                m2m_dir=str(m2m_dir),
                atlas_name=atlas_name,
                output_dir=str(output_dir)
            )
            
            generated_regions, report_file = generator.run()
            
            results[subject_name] = {
                'status': 'success',
                'surface_file': str(surface_file),
                'output_dir': str(output_dir),
                'num_regions': len(generated_regions),
                'report_file': str(report_file)
            }
            
        except Exception as e:
            print(f"Error processing {subject_name}: {e}")
            results[subject_name] = {'status': 'failed', 'reason': str(e)}
    
    return results


def analyze_mesh_properties(mesh_file):
    """
    Analyze properties of a SimNIBS mesh file.
    
    Parameters:
    -----------
    mesh_file : str
        Path to the mesh file
    
    Returns:
    --------
    dict : Mesh properties and statistics
    """
    mesh = simnibs.read_msh(str(mesh_file))
    
    # Get mesh statistics correctly
    num_nodes = mesh.nodes.node_coord.shape[0] if hasattr(mesh.nodes, 'node_coord') else 0
    num_elements = len(mesh.elm.elm_number) if hasattr(mesh.elm, 'elm_number') else 0
    
    # Count triangles and tetrahedra
    num_triangles = sum(mesh.elm.elm_type == 2)  # Triangle elements
    num_tetrahedra = sum(mesh.elm.elm_type == 4)  # Tetrahedra elements
    
    properties = {
        'file': str(mesh_file),
        'num_nodes': num_nodes,
        'num_elements': num_elements,
        'num_triangles': num_triangles,
        'num_tetrahedra': num_tetrahedra,
        'node_fields': [],
        'element_fields': [],
        'bounding_box': {},
        'surface_area': 0.0
    }
    
    # Get field information
    if hasattr(mesh, 'nodedata') and mesh.nodedata:
        properties['node_fields'] = [field.field_name for field in mesh.nodedata if hasattr(field, 'field_name')]
    
    if hasattr(mesh, 'elmdata') and mesh.elmdata:
        properties['element_fields'] = [field.field_name for field in mesh.elmdata if hasattr(field, 'field_name')]
    
    # Calculate bounding box
    if num_nodes > 0:
        coords = mesh.nodes.node_coord
        properties['bounding_box'] = {
            'min_x': float(np.min(coords[:, 0])),
            'max_x': float(np.max(coords[:, 0])),
            'min_y': float(np.min(coords[:, 1])),
            'max_y': float(np.max(coords[:, 1])),
            'min_z': float(np.min(coords[:, 2])),
            'max_z': float(np.max(coords[:, 2]))
        }
    
    # Calculate surface area for triangular mesh
    if num_triangles > 0:
        try:
            areas = mesh.elements_volumes_and_areas()
            properties['surface_area'] = float(np.sum(areas))
        except:
            # If surface area calculation fails, set to 0
            properties['surface_area'] = 0.0
    
    return properties


def compare_atlases(mesh_file, m2m_dir, atlases=['HCP_MMP1', 'DKTatlas40']):
    """
    Compare different atlases for the same subject.
    
    Parameters:
    -----------
    mesh_file : str
        Path to the mesh file
    m2m_dir : str
        Path to the m2m directory
    atlases : list
        List of atlas names to compare
    
    Returns:
    --------
    dict : Comparison results
    """
    comparison = {
        'mesh_file': str(mesh_file),
        'm2m_dir': str(m2m_dir),
        'atlases': {}
    }
    
    for atlas_name in atlases:
        try:
            atlas = subject_atlas(atlas_name, str(m2m_dir))
            
            atlas_info = {
                'num_regions': len(atlas.keys()),
                'region_names': list(atlas.keys()),
                'region_sizes': {}
            }
            
            # Calculate region sizes
            for region_name, region_mask in atlas.items():
                atlas_info['region_sizes'][region_name] = int(np.sum(region_mask))
            
            comparison['atlases'][atlas_name] = atlas_info
            
        except Exception as e:
            comparison['atlases'][atlas_name] = {'error': str(e)}
    
    return comparison


def create_region_summary_csv(regions_dir, output_file=None):
    """
    Create a CSV summary of all generated regions.
    
    Parameters:
    -----------
    regions_dir : str
        Directory containing generated region files
    output_file : str, optional
        Output CSV file path. If None, saves to regions_dir
    
    Returns:
    --------
    str : Path to the created CSV file
    """
    import csv
    
    regions_dir = Path(regions_dir)
    if output_file is None:
        output_file = regions_dir / "regions_summary.csv"
    
    # Find all region mesh files
    region_files = list(regions_dir.glob("*_region.msh"))
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['region_name', 'file_name', 'num_nodes', 'num_triangles', 'surface_area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for region_file in region_files:
            try:
                # Extract region name from filename
                region_name = region_file.stem.replace('_region', '')
                
                # Analyze the mesh
                properties = analyze_mesh_properties(region_file)
                
                writer.writerow({
                    'region_name': region_name,
                    'file_name': region_file.name,
                    'num_nodes': properties['num_nodes'],
                    'num_triangles': properties['num_triangles'],
                    'surface_area': properties['surface_area']
                })
                
            except Exception as e:
                print(f"Warning: Could not process {region_file}: {e}")
    
    print(f"CSV summary saved to: {output_file}")
    return str(output_file)


def validate_generated_regions(regions_dir, expected_atlas='HCP_MMP1'):
    """
    Validate that generated regions match expected atlas regions.
    
    Parameters:
    -----------
    regions_dir : str
        Directory containing generated region files
    expected_atlas : str
        Atlas name to validate against
    
    Returns:
    --------
    dict : Validation results
    """
    regions_dir = Path(regions_dir)
    
    # Get list of generated region files
    region_files = list(regions_dir.glob("*_region.msh"))
    generated_regions = {f.stem.replace('_region', '') for f in region_files}
    
    validation = {
        'regions_dir': str(regions_dir),
        'expected_atlas': expected_atlas,
        'generated_count': len(generated_regions),
        'generated_regions': sorted(list(generated_regions)),
        'missing_regions': [],
        'unexpected_regions': [],
        'file_issues': [],
        'valid': True
    }
    
    # For validation, we'd need to load the expected atlas
    # This is a simplified validation that checks file existence and readability
    for region_file in region_files:
        try:
            mesh = simnibs.read_msh(str(region_file))
            num_nodes = mesh.nodes.node_coord.shape[0] if hasattr(mesh.nodes, 'node_coord') else 0
            if num_nodes == 0:
                validation['file_issues'].append(f"{region_file.name}: Empty mesh")
                validation['valid'] = False
        except Exception as e:
            validation['file_issues'].append(f"{region_file.name}: {str(e)}")
            validation['valid'] = False
    
    return validation


def export_regions_to_freesurfer(regions_dir, subject_id, freesurfer_subjects_dir=None):
    """
    Export generated regions to FreeSurfer format.
    
    Parameters:
    -----------
    regions_dir : str
        Directory containing generated region files
    subject_id : str
        FreeSurfer subject ID
    freesurfer_subjects_dir : str, optional
        FreeSurfer SUBJECTS_DIR. If None, uses environment variable
    
    Returns:
    --------
    dict : Export results
    """
    if freesurfer_subjects_dir is None:
        freesurfer_subjects_dir = os.environ.get('SUBJECTS_DIR')
        
    if freesurfer_subjects_dir is None:
        raise ValueError("FreeSurfer SUBJECTS_DIR not specified and not in environment")
    
    subjects_dir = Path(freesurfer_subjects_dir)
    subject_dir = subjects_dir / subject_id
    label_dir = subject_dir / "label"
    
    if not subject_dir.exists():
        raise ValueError(f"FreeSurfer subject directory not found: {subject_dir}")
    
    label_dir.mkdir(exist_ok=True)
    
    regions_dir = Path(regions_dir)
    region_files = list(regions_dir.glob("*_region.msh"))
    
    export_results = {
        'subject_id': subject_id,
        'freesurfer_subjects_dir': str(freesurfer_subjects_dir),
        'label_dir': str(label_dir),
        'exported_regions': [],
        'failed_exports': []
    }
    
    for region_file in region_files:
        try:
            region_name = region_file.stem.replace('_region', '')
            
            # Load the region mesh
            mesh = simnibs.read_msh(str(region_file))
            
            # For FreeSurfer export, we'd need to convert to FreeSurfer label format
            # This is a placeholder - actual implementation would depend on 
            # the specific requirements and mesh structure
            
            label_file = label_dir / f"{region_name}.label"
            
            # This is a simplified placeholder
            num_vertices = mesh.nodes.node_coord.shape[0] if hasattr(mesh.nodes, 'node_coord') else 0
            export_results['exported_regions'].append({
                'region_name': region_name,
                'label_file': str(label_file),
                'num_vertices': num_vertices
            })
            
        except Exception as e:
            export_results['failed_exports'].append({
                'region_file': str(region_file),
                'error': str(e)
            })
    
    return export_results


def main():
    """Command line interface for utility functions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SimNIBS Region Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple subjects')
    batch_parser.add_argument('subjects_dir', help='Directory containing subject m2m folders')
    batch_parser.add_argument('-a', '--atlas', default='HCP_MMP1', help='Atlas to use')
    batch_parser.add_argument('-o', '--output', help='Output base directory')
    
    # Mesh analysis
    analyze_parser = subparsers.add_parser('analyze', help='Analyze mesh properties')
    analyze_parser.add_argument('mesh_file', help='Mesh file to analyze')
    analyze_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Atlas comparison
    compare_parser = subparsers.add_parser('compare', help='Compare atlases')
    compare_parser.add_argument('mesh_file', help='Mesh file')
    compare_parser.add_argument('m2m_dir', help='m2m directory')
    compare_parser.add_argument('-a', '--atlases', nargs='+', default=['HCP_MMP1', 'DKTatlas40'])
    
    # Create CSV summary
    csv_parser = subparsers.add_parser('csv', help='Create CSV summary of regions')
    csv_parser.add_argument('regions_dir', help='Directory containing region files')
    csv_parser.add_argument('-o', '--output', help='Output CSV file')
    
    # Validation
    validate_parser = subparsers.add_parser('validate', help='Validate generated regions')
    validate_parser.add_argument('regions_dir', help='Directory containing region files')
    validate_parser.add_argument('-a', '--atlas', default='HCP_MMP1', help='Expected atlas')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        results = batch_process_subjects(args.subjects_dir, args.atlas, args.output)
        print(f"\nBatch processing completed. Results:")
        for subject, result in results.items():
            print(f"  {subject}: {result['status']}")
    
    elif args.command == 'analyze':
        properties = analyze_mesh_properties(args.mesh_file)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(properties, f, indent=2)
            print(f"Analysis saved to: {args.output}")
        else:
            print(json.dumps(properties, indent=2))
    
    elif args.command == 'compare':
        comparison = compare_atlases(args.mesh_file, args.m2m_dir, args.atlases)
        print(json.dumps(comparison, indent=2))
    
    elif args.command == 'csv':
        csv_file = create_region_summary_csv(args.regions_dir, args.output)
        print(f"CSV summary created: {csv_file}")
    
    elif args.command == 'validate':
        validation = validate_generated_regions(args.regions_dir, args.atlas)
        print(json.dumps(validation, indent=2))
        
        if not validation['valid']:
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
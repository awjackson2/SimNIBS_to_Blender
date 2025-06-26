#!/usr/bin/env python3
"""
SimNIBS Cortical Region Generator

This script takes a SimNIBS mesh file (.msh) that has been produced by msh2cortex
and generates cortical regions for each cortex in a specified atlas.

Requirements:
    - simnibs
    - numpy
    - argparse
    - os

Usage:
    python simnibs_cortical_regions.py -i input.msh -m m2m_folder -a atlas_name -o output_dir
    
    # With global field scaling (NEW):
    python simnibs_cortical_regions.py -i input.msh -m m2m_folder -a atlas_name -o output_dir \
        --global-field data/field_data/TI_max.nii.gz

Author: Generated for SimNIBS mesh processing
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import simnibs
    from simnibs.mesh_tools import mesh_io
    from simnibs.utils.transformations import subject_atlas
    from simnibs import read_msh
    from simnibs.mesh_tools.mesh_io import NodeData
    import nibabel as nib
except ImportError:
    print("Error: SimNIBS is not installed or not in the Python path.")
    print("Please install SimNIBS from: https://simnibs.github.io/simnibs/")
    sys.exit(1)

class CorticalRegionGenerator:
    """Class to handle generation of cortical regions from SimNIBS mesh files."""
    
    def __init__(self, mesh_file, m2m_dir, atlas_name='HCP_MMP1', output_dir=None, global_field_file=None):
        """
        Initialize the cortical region generator.

        Parameters:
        -----------
        mesh_file : str
            Path to the SimNIBS mesh file (.msh) produced by msh2cortex
        m2m_dir : str  
            Path to the m2m directory for the subject
        atlas_name : str, default='HCP_MMP1'
            Name of the atlas to use. Common options:
            - 'HCP_MMP1' (Human Connectome Project Multi-Modal Parcellation)
            - 'DKTatlas40' (Desikan-Killiany-Tourville atlas)
            - 'DK40' (Desikan-Killiany 40 regions)
            - 'aparc.a2009s' (Destrieux atlas)
        output_dir : str, optional
            Directory to save output files. If None, uses current directory
        global_field_file : str, optional
            Path to NIfTI file containing field data for global scaling (NEW FEATURE)
        """
        self.mesh_file = Path(mesh_file)
        self.m2m_dir = Path(m2m_dir)
        self.atlas_name = atlas_name
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.global_field_file = Path(global_field_file) if global_field_file else None
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize variables
        self.mesh = None
        self.atlas = None
        self.region_names = None
        
        # Global field data (optional feature)
        self.global_field_data = None
        self.global_field_range = None
        self.global_field_affine = None
        
    def _validate_inputs(self):
        """Validate input files and directories."""
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")
        
        if not self.m2m_dir.exists():
            raise FileNotFoundError(f"m2m directory not found: {self.m2m_dir}")
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")
        
    def load_mesh(self):
        """Load the SimNIBS mesh file."""
        print(f"Loading mesh file: {self.mesh_file}")
        try:
            self.mesh = simnibs.read_msh(str(self.mesh_file))
            
            # Get mesh statistics correctly
            num_nodes = self.mesh.nodes.node_coord.shape[0] if hasattr(self.mesh.nodes, 'node_coord') else 0
            num_elements = len(self.mesh.elm.elm_number) if hasattr(self.mesh.elm, 'elm_number') else 0
            
            print(f"Successfully loaded mesh with {num_nodes} nodes and {num_elements} elements")
            
            # Check if this is a cortical surface mesh (should have triangular elements)
            # Triangle elements have type 2 in SimNIBS
            triangles = self.mesh.elm.elm_type == 2
            num_triangles = sum(triangles)
            
            if num_triangles == 0:
                raise ValueError("The mesh file does not contain cortical surface triangles. "
                               "Make sure the file was produced by msh2cortex.")
            else:
                print(f"Found {num_triangles} triangular elements")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh file: {e}")
    
    def load_global_field_data(self):
        """Load and analyze global field data for consistent scaling."""
        if not self.global_field_file or not self.global_field_file.exists():
            print("No global field file provided. Using local field scaling.")
            return
            
        print(f"Loading global field data from {self.global_field_file}")
        try:
            # Load NIfTI file
            nii = nib.load(str(self.global_field_file))
            self.global_field_data = nii.get_fdata()
            self.global_field_affine = nii.affine
            
            # Calculate global range (excluding zeros for better scaling)
            nonzero_data = self.global_field_data[self.global_field_data > 0]
            if len(nonzero_data) > 0:
                self.global_field_range = (np.min(nonzero_data), np.max(nonzero_data))
            else:
                self.global_field_range = (np.min(self.global_field_data), np.max(self.global_field_data))
                
            print(f"Global field range: [{self.global_field_range[0]:.6f}, {self.global_field_range[1]:.6f}]")
            print(f"Field data shape: {self.global_field_data.shape}")
            
        except Exception as e:
            print(f"Error loading field data: {e}")
            self.global_field_data = None
            self.global_field_range = None

    def apply_global_field_to_nodes(self, mesh, field_name="TI_max"):
        """
        Apply global field data to mesh nodes (optional feature).
        
        Parameters:
        -----------
        mesh : simnibs.Msh
            Mesh object to add field data to
        field_name : str
            Name for the field data
            
        Returns:
        --------
        simnibs.Msh
            Mesh with field data added
        """
        if self.global_field_data is None:
            return mesh
            
        try:
            # Get mesh node coordinates
            node_coords = mesh.nodes.node_coord
            
            # Transform mesh coordinates to voxel indices
            ones = np.ones((len(node_coords), 1))
            coords_homog = np.hstack([node_coords, ones])
            
            # Apply inverse affine transformation
            inv_affine = np.linalg.inv(self.global_field_affine)
            voxel_coords = (inv_affine @ coords_homog.T).T[:, :3]
            
            # Round to nearest voxel indices
            voxel_indices = np.round(voxel_coords).astype(int)
            
            # Extract field values at mesh node locations
            field_values = np.zeros(len(node_coords))
            valid_mask = (
                (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.global_field_data.shape[0]) &
                (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.global_field_data.shape[1]) &
                (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.global_field_data.shape[2])
            )
            
            # Sample field data at valid locations
            valid_indices = voxel_indices[valid_mask]
            field_values[valid_mask] = self.global_field_data[
                valid_indices[:, 0], 
                valid_indices[:, 1], 
                valid_indices[:, 2]
            ]
            
            # Normalize field values using global range for consistent visualization
            normalized_field_values = np.zeros_like(field_values)
            if self.global_field_range[1] > self.global_field_range[0]:
                # Normalize to [0,1] based on global range
                normalized_field_values = (field_values - self.global_field_range[0]) / (self.global_field_range[1] - self.global_field_range[0])
                # Clamp to [0,1] to handle any outliers
                normalized_field_values = np.clip(normalized_field_values, 0.0, 1.0)
            
            # Add normalized field to mesh
            nodedata = NodeData(normalized_field_values, field_name)
            mesh.add_node_field(nodedata, field_name)
            
            # Also store the raw field values for reference
            raw_nodedata = NodeData(field_values, f"{field_name}_raw")
            mesh.add_node_field(raw_nodedata, f"{field_name}_raw")
            
            # Report the ranges
            region_range = (np.min(field_values[valid_mask]), np.max(field_values[valid_mask]))
            normalized_range = (np.min(normalized_field_values[valid_mask]), np.max(normalized_field_values[valid_mask]))
            print(f"  Applied global field '{field_name}' to {np.sum(valid_mask)}/{len(node_coords)} nodes")
            print(f"  Raw regional range: [{region_range[0]:.6f}, {region_range[1]:.6f}]")
            print(f"  Normalized range: [{normalized_range[0]:.6f}, {normalized_range[1]:.6f}]")
            print(f"  Global range: [{self.global_field_range[0]:.6f}, {self.global_field_range[1]:.6f}]")
            
            return mesh
            
        except Exception as e:
            print(f"Warning: Failed to apply global field data: {e}")
            return mesh

    def load_atlas(self):
        """Load the specified atlas for the subject."""
        print(f"Loading atlas: {self.atlas_name}")
        try:
            # Load atlas using SimNIBS subject_atlas function
            self.atlas = subject_atlas(self.atlas_name, str(self.m2m_dir))
            
            # Get region names
            self.region_names = list(self.atlas.keys())
            print(f"Successfully loaded atlas with {len(self.region_names)} regions")
            
            # Display some region names for verification
            print("Sample regions:")
            for i, name in enumerate(self.region_names[:10]):
                print(f"  {name}")
            if len(self.region_names) > 10:
                print(f"  ... and {len(self.region_names) - 10} more regions")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load atlas: {e}")
    
    def generate_regions(self):
        """Generate individual cortical region meshes."""
        print("\nGenerating cortical regions...")
        
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        if self.atlas is None:
            raise ValueError("Atlas not loaded. Call load_atlas() first.")
        
        generated_regions = []
        
        for region_name in self.region_names:
            try:
                # Get nodes belonging to this region
                region_mask = self.atlas[region_name]
                
                if not np.any(region_mask):
                    print(f"Warning: No vertices found for region '{region_name}', skipping...")
                    continue
                    
                # Get the node indices for this region
                region_node_indices = np.where(region_mask)[0]
                
                if len(region_node_indices) == 0:
                    print(f"Warning: No vertices found for region '{region_name}', skipping...")
                    continue
                
                # Find triangular elements that have at least one vertex in the region
                triangular_elements = self.mesh.elm.elm_type == 2
                triangle_indices = np.where(triangular_elements)[0]
                
                # Get the node connectivity for triangular elements
                triangle_nodes = self.mesh.elm.node_number_list[triangular_elements] - 1  # Convert to 0-based indexing
                
                # Find triangles where at least 2 out of 3 vertices belong to the region
                # This is less restrictive and will capture more triangles at region boundaries
                triangles_in_region = []
                region_node_set = set(region_node_indices)
                
                for i, triangle in enumerate(triangle_nodes):
                    # Check how many vertices of the triangle are in the region
                    vertices_in_region = sum(1 for node_idx in triangle if node_idx in region_node_set)
                    # Include triangle if at least 2 out of 3 vertices are in the region
                    if vertices_in_region >= 2:
                        triangles_in_region.append(triangle_indices[i])
                
                if len(triangles_in_region) == 0:
                    print(f"Warning: No triangular elements found entirely within region '{region_name}', skipping...")
                    continue
                
                # Use SimNIBS's crop_mesh with node selection
                try:
                    # Create a node selection based on the region
                    nodes_to_keep = np.where(region_mask)[0]
                    
                    # Use SimNIBS crop functionality - try with nodes parameter
                    region_mesh = self.mesh.crop_mesh(nodes=nodes_to_keep)
                    
                except Exception as e:
                    try:
                        # Alternative: use element-based cropping
                        elements_to_keep = np.array(triangles_in_region)
                        region_mesh = self.mesh.crop_mesh(elements=elements_to_keep)
                    except Exception as e2:
                        print(f"Warning: Could not crop mesh for region '{region_name}': {e}, {e2}")
                        # Fallback: just save the original mesh with the region field
                        region_mesh = self.mesh
                
                # Apply global field data to this region (NEW FEATURE)
                if self.global_field_data is not None:
                    region_mesh = self.apply_global_field_to_nodes(region_mesh, "TI_max")
                
                # Create output filename
                output_file = self.output_dir / f"{region_name}_region.msh"
                
                # Save the cropped region mesh
                region_mesh.write(str(output_file))
                
                generated_regions.append({
                    'name': region_name,
                    'file': output_file,
                    'num_nodes': len(region_mesh.nodes.node_coord),
                    'num_vertices': np.sum(region_mask),
                    'num_triangles': len(triangles_in_region)
                })
                
                print(f"Generated region '{region_name}': {len(triangles_in_region)} triangles, {len(region_mesh.nodes.node_coord)} nodes -> {output_file}")
                
            except Exception as e:
                print(f"Warning: Failed to generate region '{region_name}': {e}")
                continue
        
        return generated_regions

    def generate_summary_report(self, generated_regions):
        """Generate a summary report of all generated regions."""
        report_file = self.output_dir / "cortical_regions_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("SimNIBS Cortical Regions Generation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input mesh file: {self.mesh_file}\n")
            f.write(f"m2m directory: {self.m2m_dir}\n")
            f.write(f"Atlas used: {self.atlas_name}\n")
            f.write(f"Global field file: {self.global_field_file}\n")
            if self.global_field_range:
                f.write(f"Global field range: [{self.global_field_range[0]:.6f}, {self.global_field_range[1]:.6f}]\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Total regions generated: {len(generated_regions)}\n\n")
            
            f.write("Generated Regions:\n")
            f.write("-" * 30 + "\n")
            
            for region in generated_regions:
                f.write(f"Region: {region['name']}\n")
                f.write(f"  File: {region['file'].name}\n")
                f.write(f"  Original vertices in region: {region['num_vertices']}\n")
                f.write(f"  Cropped mesh nodes: {region['num_nodes']}\n")
                f.write(f"  Triangular elements: {region.get('num_triangles', 'N/A')}\n\n")
        
        print(f"\nSummary report saved to: {report_file}")
        return report_file
    
    def run(self):
        """Run the complete cortical region generation process."""
        print("Starting SimNIBS Cortical Region Generation")
        print("=" * 50)
        
        try:
            # Load mesh and atlas
            self.load_mesh()
            
            # Load global field data if provided (NEW FEATURE)
            if self.global_field_file:
                self.load_global_field_data()
            
            self.load_atlas()
            
            # Generate regions
            generated_regions = self.generate_regions()
            
            # Generate summary report
            report_file = self.generate_summary_report(generated_regions)
            
            # Save global field range information if available
            if self.global_field_range:
                global_info_file = self.output_dir / "global_field_info.txt"
                with open(global_info_file, 'w') as f:
                    f.write("Global Field Information\n")
                    f.write("=" * 30 + "\n")
                    f.write(f"Field file: {self.global_field_file}\n")
                    f.write(f"Global min: {self.global_field_range[0]:.6f}\n")
                    f.write(f"Global max: {self.global_field_range[1]:.6f}\n")
                    f.write(f"Field data shape: {self.global_field_data.shape}\n")
                    f.write("\nNote: Field values in .msh files are normalized to [0,1] based on this global range.\n")
                    f.write("Raw field values are stored as '{field_name}_raw' in each mesh.\n")
                print(f"Global field info saved to: {global_info_file}")
            
            print(f"\nProcess completed successfully!")
            print(f"Generated {len(generated_regions)} cortical regions")
            if self.global_field_range:
                print(f"Applied global field scaling: [{self.global_field_range[0]:.6f}, {self.global_field_range[1]:.6f}]")
                print("Field values are normalized to [0,1] in .msh files for consistent visualization")
            print(f"Output files saved to: {self.output_dir}")
            
            return generated_regions, report_file
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise

def main():
    """Main function to handle command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate cortical regions from SimNIBS mesh file using specified atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with HCP_MMP1 atlas
    python simnibs_cortical_regions.py -i subject_central.msh -m m2m_subject -o output_regions
    
    # Use different atlas
    python simnibs_cortical_regions.py -i subject_central.msh -m m2m_subject -a DK40
    
    # With global field scaling (NEW FEATURE)
    python simnibs_cortical_regions.py -i subject_central.msh -m m2m_subject -a DK40 \
        --global-field data/field_data/TI_max.nii.gz
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input SimNIBS mesh file (.msh) produced by msh2cortex'
    )
    
    parser.add_argument(
        '-m', '--m2m',
        required=True,
        help='Path to the m2m directory for the subject'
    )
    
    parser.add_argument(
        '-a', '--atlas',
        default='HCP_MMP1',
        help='Atlas name to use (default: HCP_MMP1). Options: HCP_MMP1, DK40, DKTatlas40, aparc.a2009s'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for generated regions (default: current directory)'
    )
    
    parser.add_argument(
        '--global-field',
        help='Path to NIfTI field file for global scaling (NEW FEATURE)'
    )
    
    parser.add_argument(
        '--list-atlases',
        action='store_true',
        help='List available atlases and exit'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.list_atlases:
        print("Available atlases:")
        print("- HCP_MMP1: Human Connectome Project Multi-Modal Parcellation")
        print("- DK40: Desikan-Killiany 40 regions")
        print("- DKTatlas40: Desikan-Killiany-Tourville atlas (40 regions)")
        print("- aparc.a2009s: Destrieux atlas")
        return
    
    try:
        # Create the generator
        generator = CorticalRegionGenerator(
            mesh_file=args.input,
            m2m_dir=args.m2m,
            atlas_name=args.atlas,
            output_dir=args.output,
            global_field_file=args.global_field
        )
        
        # Run the generation process
        generated_regions, report_file = generator.run()
        
        print(f"\n✓ Successfully generated {len(generated_regions)} cortical regions")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
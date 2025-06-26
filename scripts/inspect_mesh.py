#!/usr/bin/env python3
"""
Mesh Inspector

Simple script to inspect the structure of a SimNIBS mesh object
to understand the correct attributes and methods.
"""

import sys
try:
    import simnibs
except ImportError:
    print("SimNIBS not found. Please ensure it's installed and in your Python path.")
    sys.exit(1)

def inspect_mesh(mesh_file):
    """Inspect a mesh file and print its structure."""
    print(f"Loading mesh: {mesh_file}")
    
    try:
        mesh = simnibs.read_msh(mesh_file)
        print(f"✓ Successfully loaded mesh")
        
        print("\n=== MESH OBJECT STRUCTURE ===")
        print(f"Type: {type(mesh)}")
        print(f"Available attributes:")
        
        # List all attributes
        attrs = [attr for attr in dir(mesh) if not attr.startswith('_')]
        for attr in sorted(attrs):
            try:
                value = getattr(mesh, attr)
                if callable(value):
                    print(f"  {attr}() - method")
                else:
                    print(f"  {attr} - {type(value)}")
            except:
                print(f"  {attr} - (unable to access)")
        
        print("\n=== DETAILED INSPECTION ===")
        
        # Check nodes
        if hasattr(mesh, 'nodes'):
            print(f"✓ nodes attribute found: {type(mesh.nodes)}")
            if hasattr(mesh.nodes, 'node_coord'):
                print(f"  - node_coord shape: {mesh.nodes.node_coord.shape}")
            else:
                print("  - Available node attributes:")
                node_attrs = [attr for attr in dir(mesh.nodes) if not attr.startswith('_')]
                for attr in node_attrs[:10]:  # First 10 only
                    print(f"    {attr}")
        
        # Check elements - try different ways
        element_attrs = ['elements', 'elm', 'triangles', 'elmdata']
        for attr in element_attrs:
            if hasattr(mesh, attr):
                value = getattr(mesh, attr)
                print(f"✓ {attr} attribute found: {type(value)}")
                
                if hasattr(value, '__len__'):
                    try:
                        print(f"  - length: {len(value)}")
                    except:
                        print(f"  - cannot get length")
                
                # List sub-attributes
                if hasattr(value, '__dict__') or hasattr(value, '__dir__'):
                    sub_attrs = [a for a in dir(value) if not a.startswith('_')][:10]
                    print(f"  - sub-attributes: {sub_attrs}")
        
        # Check for data fields
        data_attrs = ['nodedata', 'elmdata', 'field']
        for attr in data_attrs:
            if hasattr(mesh, attr):
                value = getattr(mesh, attr)
                print(f"✓ {attr} attribute found: {type(value)}")
                if hasattr(value, '__len__'):
                    try:
                        print(f"  - length: {len(value)}")
                    except:
                        pass
        
    except Exception as e:
        print(f"✗ Error loading mesh: {e}")
        return None
    
    return mesh

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_mesh.py <mesh_file>")
        print("Example: python inspect_mesh.py 003_A_TI_central.msh")
        sys.exit(1)
    
    mesh_file = sys.argv[1]
    mesh = inspect_mesh(mesh_file) 
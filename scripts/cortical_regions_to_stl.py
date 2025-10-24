#!/usr/bin/env python3
"""
Single-tool converter: generate atlas-accurate cortical region STLs and a whole-GM STL.

Features:
- Loads subject cortical surface mesh (.msh from msh2cortex) and subject atlas (default DKTatlas40)
- Exports individual region meshes as binary STL
- Exports the full GM surface as a single STL
- Optional: keep individual region meshes and whole GM mesh as .msh files
- Optional: generate cortical surface from tetrahedral GM mesh using msh2cortex

Examples:
    simnibs_python cortical_regions_to_stl.py \
        --mesh subject_central.msh \
        --m2m m2m_subject \
        --output-dir out \
        --atlas DKTatlas40

    simnibs_python cortical_regions_to_stl.py \
        --gm-mesh m2m_subject/subject.msh \
        --m2m m2m_subject \
        --output-dir out \
        --surface central

    simnibs_python cortical_regions_to_stl.py \
        --mesh subject_central.msh \
        --m2m m2m_subject \
        --output-dir out \
        --keep-meshes
"""

import argparse
import os
import sys
import logging
import struct
from pathlib import Path
import numpy as np
import tempfile
import subprocess
import shutil
import platform

try:
    import simnibs
    from simnibs import read_msh
    from simnibs.utils.transformations import subject_atlas
except ImportError:
    print("Error: SimNIBS not found. Please install SimNIBS and activate the environment.")
    sys.exit(1)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError:
    print("Error: scipy not found. Please install scipy in the SimNIBS environment.")
    sys.exit(1)


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# STL Writer
# ──────────────────────────────────────────────────────────────────────────────

def write_binary_stl(vertices, faces, output_path):
    """Write binary STL file from vertices and faces."""
    n_faces = len(faces)
    
    with open(output_path, 'wb') as f:
        # Write 80-byte header
        header = f"Generated from SimNIBS mesh".encode('ascii')
        header = header.ljust(80, b'\x00')
        f.write(header)
        
        # Write number of triangles (4 bytes, little-endian)
        f.write(struct.pack('<I', n_faces))
        
        # Write each triangle
        for face in faces:
            # Get triangle vertices
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Calculate normal vector
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_length = np.linalg.norm(normal)
            
            if normal_length > 1e-12:
                normal = normal / normal_length
            else:
                normal = np.array([0.0, 0.0, 1.0])  # Default normal
            
            # Write normal (3 floats, little-endian)
            f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))
            
            # Write vertices (9 floats, little-endian)
            f.write(struct.pack('<fff', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<fff', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<fff', v2[0], v2[1], v2[2]))
            
            # Write attribute byte count (2 bytes, little-endian)
            f.write(struct.pack('<H', 0))


# ──────────────────────────────────────────────────────────────────────────────
# Mesh/Atlas Utilities
# ──────────────────────────────────────────────────────────────────────────────

def validate_mesh_quality(mesh, min_triangles=10):
    """Validate that a mesh has sufficient quality for STL export."""
    if mesh is None:
        return False
    
    triangles = np.sum(mesh.elm.elm_type == 2)
    if triangles < min_triangles:
        return False
        
    # Check for degenerate triangles (very small area)
    try:
        vertices, faces = mesh_vertices_faces(mesh)
        if vertices is None or faces is None:
            return False
            
        # Calculate triangle areas
        areas = []
        for face in faces:
            v0, v1, v2 = vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        
        areas = np.array(areas)
        valid_triangles = np.sum(areas > 1e-10)  # Remove degenerate triangles
        
        return valid_triangles >= min_triangles
    except Exception:
        return False


def extract_region_mesh_connected(mesh, region_mask, min_triangles=10):
    """Extract mesh region using connected component analysis for clean segmentation."""
    try:
        # Get triangular elements only
        triangular_elements = mesh.elm.elm_type == 2
        if not np.any(triangular_elements):
            logger.warning("No triangular elements found in mesh")
            return None
            
        triangle_indices = np.where(triangular_elements)[0]
        triangle_nodes = mesh.elm.node_number_list[triangular_elements] - 1
        region_node_set = set(np.where(region_mask)[0])
        
        # Step 1: Find triangles where ALL vertices are in the region (strict inclusion)
        complete_triangles = []
        for i, tri in enumerate(triangle_nodes):
            if all(v in region_node_set for v in tri[:3]):
                complete_triangles.append(triangle_indices[i])
        
        if len(complete_triangles) == 0:
            logger.warning("No complete triangles found in region")
            return None
        
        logger.debug(f"Found {len(complete_triangles)} complete triangles in region")
        
        # Step 2: Build adjacency graph for connected component analysis
        
        # Create node-to-triangle mapping
        node_to_triangles = {}
        for i, tri_idx in enumerate(complete_triangles):
            tri_nodes = triangle_nodes[triangle_indices == tri_idx][0]
            for node in tri_nodes:
                if node not in node_to_triangles:
                    node_to_triangles[node] = []
                node_to_triangles[node].append(i)
        
        # Build adjacency matrix for triangles
        n_triangles = len(complete_triangles)
        adjacency = np.zeros((n_triangles, n_triangles), dtype=bool)
        
        for node, tri_list in node_to_triangles.items():
            # Connect all triangles that share this node
            for i in range(len(tri_list)):
                for j in range(i + 1, len(tri_list)):
                    tri_i, tri_j = tri_list[i], tri_list[j]
                    adjacency[tri_i, tri_j] = True
                    adjacency[tri_j, tri_i] = True
        
        # Step 3: Find largest connected component
        n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
        
        if n_components == 0:
            logger.warning("No connected components found")
            return None
        
        # Find the largest component
        component_sizes = np.bincount(labels)
        largest_component = np.argmax(component_sizes)
        largest_size = component_sizes[largest_component]
        
        logger.debug(f"Found {n_components} connected components, largest has {largest_size} triangles")
        
        if largest_size < min_triangles:
            logger.warning(f"Largest component too small: {largest_size} < {min_triangles}")
            return None
        
        # Step 4: Extract the largest connected component
        component_triangles = []
        for i, label in enumerate(labels):
            if label == largest_component:
                component_triangles.append(complete_triangles[i])
        
        # Step 5: Create the region mesh
        elements_to_keep = np.array(component_triangles)
        region_mesh = mesh.crop_mesh(elements=elements_to_keep)
        
        if region_mesh is not None and len(region_mesh.nodes.node_coord) > 0:
            final_triangles = np.sum(region_mesh.elm.elm_type == 2)
            logger.debug(f"Extracted connected region with {final_triangles} triangles")
            return region_mesh
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to extract connected region mesh: {e}")
        return None


def extract_region_mesh(mesh, region_mask, min_triangles=10, method="connected"):
    """Extract mesh region using specified method."""
    if method == "connected":
        # Try the new connected component method first
        region_mesh = extract_region_mesh_connected(mesh, region_mask, min_triangles)
        if region_mesh is not None:
            return region_mesh
        logger.debug("Connected component method failed, falling back to original")
    
    # Use original method
    try:
        # Primary method: use SimNIBS crop_mesh with nodes
        nodes_to_keep = np.where(region_mask)[0]
        if nodes_to_keep.size == 0:
            return None
        
        logger.debug(f"Attempting node-based cropping with {len(nodes_to_keep)} nodes")
        region_mesh = mesh.crop_mesh(nodes=nodes_to_keep)
        
        # Validate the result
        if region_mesh is not None and len(region_mesh.nodes.node_coord) > 0:
            triangles = np.sum(region_mesh.elm.elm_type == 2)
            if triangles >= min_triangles:
                logger.debug(f"Node-based cropping successful: {triangles} triangles")
                return region_mesh
        
    except Exception as e:
        logger.debug(f"Node-based cropping failed: {e}")
    
    # Final fallback: element-based filtering
    try:
        logger.debug("Falling back to element-based filtering")
        triangular_elements = mesh.elm.elm_type == 2
        if not np.any(triangular_elements):
            return None
            
        triangle_indices = np.where(triangular_elements)[0]
        triangle_nodes = mesh.elm.node_number_list[triangular_elements] - 1
        region_node_set = set(np.where(region_mask)[0].tolist())
        
        triangles_in_region = []
        for i, tri in enumerate(triangle_nodes):
            if sum((v in region_node_set) for v in tri[:3]) >= 2:
                triangles_in_region.append(triangle_indices[i])
        
        if len(triangles_in_region) == 0:
            return None
            
        elements_to_keep = np.array(triangles_in_region)
        region_mesh = mesh.crop_mesh(elements=elements_to_keep)
        
        if region_mesh is not None and len(region_mesh.nodes.node_coord) > 0:
            triangles = np.sum(region_mesh.elm.elm_type == 2)
            if triangles >= min_triangles:
                logger.debug(f"Element-based filtering successful: {triangles} triangles")
                return region_mesh
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to extract region mesh: {e}")
        return None


def mesh_vertices_faces(mesh):
    """Extract vertices and faces from mesh (geometry only)."""
    triangles = mesh.elm[mesh.elm.elm_type == 2]
    if len(triangles) == 0:
        return None, None
    if hasattr(triangles, 'node_number_list'):
        triangle_nodes = triangles.node_number_list[:, :3] - 1
    else:
        triangle_nodes = triangles[:, :3] - 1
    unique_nodes = np.unique(triangle_nodes.flatten())
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
    vertices = mesh.nodes.node_coord[unique_nodes]
    faces = np.array([[node_map[idx] for idx in tri] for tri in triangle_nodes], dtype=np.int32)
    return vertices, faces


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def export_mesh_to_stl(mesh, stl_path):
    """Export mesh to STL format."""
    vertices, faces = mesh_vertices_faces(mesh)
    if vertices is None or faces is None:
        logger.error("Mesh has no triangular elements to export")
        return False
    
    write_binary_stl(vertices, faces, stl_path)
    return True


def run_conversion(mesh_path, m2m_dir, output_dir, atlas_name,
                   export_regions, export_whole_gm, keep_meshes, min_triangles, segmentation_method):
    """Main conversion workflow."""
    mesh = read_msh(mesh_path)
    logger.info(f"Loaded mesh: {mesh_path}")
    
    # Log mesh statistics
    total_nodes = len(mesh.nodes.node_coord)
    total_triangles = np.sum(mesh.elm.elm_type == 2)
    total_elements = len(mesh.elm.elm_type)
    logger.info(f"Mesh statistics: {total_nodes} nodes, {total_triangles} triangles, {total_elements} total elements")
    
    atlas = subject_atlas(atlas_name, str(m2m_dir))
    logger.info(f"Loaded atlas '{atlas_name}' with {len(atlas.keys())} regions")
    
    # Log atlas statistics
    for region_name, region_mask in list(atlas.items())[:5]:  # Show first 5 regions
        region_nodes = np.sum(region_mask)
        logger.debug(f"Region '{region_name}': {region_nodes} nodes ({region_nodes/total_nodes*100:.1f}%)")

    regions_out_dir = Path(output_dir) / "regions"
    regions_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create meshes directory if keeping meshes
    if keep_meshes:
        meshes_out_dir = Path(output_dir) / "meshes"
        meshes_out_dir.mkdir(parents=True, exist_ok=True)

    if export_regions:
        success_count = 0
        mesh_success_count = 0
        for region_name, region_mask in atlas.items():
            # Log region statistics
            region_nodes = np.sum(region_mask)
            logger.info(f"Processing region '{region_name}' with {region_nodes} nodes")
            
            region_mesh = extract_region_mesh(mesh, region_mask, min_triangles, segmentation_method)
            if region_mesh is None:
                logger.warning(f"Skipping region with no mesh: {region_name}")
                continue
            
            # Log extracted mesh statistics
            extracted_nodes = len(region_mesh.nodes.node_coord)
            extracted_triangles = np.sum(region_mesh.elm.elm_type == 2)
            logger.info(f"Extracted {extracted_nodes} nodes and {extracted_triangles} triangles for {region_name}")
            
            # Export STL
            stl_path = regions_out_dir / f"{region_name}_region.stl"
            if export_mesh_to_stl(region_mesh, str(stl_path)):
                success_count += 1
                logger.info(f"Successfully exported STL: {stl_path}")
            else:
                logger.warning(f"Failed to export STL for region: {region_name}")
            
            # Export MSH if requested
            if keep_meshes:
                msh_path = meshes_out_dir / f"{region_name}_region.msh"
                try:
                    region_mesh.write(str(msh_path))
                    mesh_success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save mesh for region {region_name}: {e}")
        
        logger.info(f"Converted {success_count}/{len(atlas.keys())} regions to STL")
        if keep_meshes:
            logger.info(f"Saved {mesh_success_count}/{len(atlas.keys())} region meshes as MSH")

    if export_whole_gm:
        whole_stl = Path(output_dir) / "whole_gm.stl"
        logger.info("Exporting whole GM surface to STL...")
        if export_mesh_to_stl(mesh, str(whole_stl)):
            logger.info(f"Successfully wrote whole GM STL: {whole_stl}")
        else:
            logger.error("Failed to export whole GM STL")
    
    # Export whole GM mesh if requested
    if keep_meshes and export_whole_gm:
        whole_msh = Path(output_dir) / "whole_gm.msh"
        try:
            mesh.write(str(whole_msh))
            logger.info(f"Wrote whole GM MSH: {whole_msh}")
        except Exception as e:
            logger.warning(f"Failed to save whole GM mesh: {e}")


def _resolve_msh2cortex(explicit_path: str | None) -> str | None:
    """Resolve msh2cortex executable path."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
    # Try PATH
    exe_name = "msh2cortex.exe" if platform.system().lower().startswith("win") else "msh2cortex"
    found = shutil.which("msh2cortex") or shutil.which(exe_name)
    if found:
        return found
    # Try locating near simnibs installation
    try:
        simnibs_root = Path(simnibs.__file__).resolve().parents[1]
        candidates = [
            simnibs_root / "bin" / exe_name,
            simnibs_root / "bin" / "msh2cortex",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Fallback: recursive search (limited depth)
        for sub in ("bin", "."):
            for p in (simnibs_root / sub).rglob("msh2cortex*"):
                if p.is_file():
                    return str(p)
    except Exception:
        pass
    return None


def generate_cortical_surface_from_tetra(gm_mesh_path, m2m_dir, surface="central", msh2cortex_path: str | None = None):
    """Generate cortical surface mesh from tetrahedral GM mesh using msh2cortex."""
    logger.info(f"Running msh2cortex on tetrahedral GM mesh: {gm_mesh_path}")
    exe = _resolve_msh2cortex(msh2cortex_path)
    if not exe:
        logger.error("msh2cortex command not found. Provide --msh2cortex or ensure it is on PATH in the SimNIBS environment.")
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        cmd = [
            exe,
            "-i", gm_mesh_path,
            "-m", str(m2m_dir),
            "-o", str(out_dir)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"msh2cortex failed: {e.stderr.decode(errors='ignore')}")
            return None

        # Try to find the requested surface mesh
        candidates = list(out_dir.glob(f"*_{surface}.msh"))
        if not candidates:
            # Fallbacks: try central if requested not found, then any *_central.msh present
            if surface != "central":
                candidates = list(out_dir.glob("*_central.msh"))
        if not candidates:
            # Last resort: any .msh produced
            candidates = list(out_dir.glob("*.msh"))
        if not candidates:
            logger.error("msh2cortex did not produce a cortical surface .msh")
            return None

        # Move/copy the selected mesh to a stable temp file we control
        selected = candidates[0]
        tmp_copy = Path(tempfile.mkstemp(suffix=f"_{surface}.msh")[1])
        tmp_copy.write_bytes(selected.read_bytes())
        logger.info(f"Using cortical surface mesh: {tmp_copy}")
        return str(tmp_copy)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export cortical regions and whole GM surface to STL")
    parser.add_argument("--mesh", help="Input cortical surface mesh (.msh) from msh2cortex")
    parser.add_argument("--gm-mesh", help="Input tetrahedral GM .msh (volumetric); will run msh2cortex")
    parser.add_argument("--m2m", required=True, help="Subject m2m directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for STL files")
    parser.add_argument("--atlas", default="DK40", help="Atlas name (default: DK40)")
    parser.add_argument("--surface", default="central", choices=["central", "pial", "white"], help="Cortical surface to extract when using --gm-mesh (default: central)")
    parser.add_argument("--msh2cortex", help="Path to msh2cortex executable (optional override)")
    parser.add_argument("--min-triangles", type=int, default=10, help="Minimum number of triangles required for a region (default: 10)")
    parser.add_argument("--segmentation-method", choices=["connected", "original"], default="connected", help="Segmentation method: connected (recommended) or original (default: connected)")
    parser.add_argument("--skip-regions", action="store_true", help="Do not export individual region STLs")
    parser.add_argument("--skip-whole-gm", action="store_true", help="Do not export the whole GM STL")
    parser.add_argument("--keep-meshes", action="store_true", help="Keep individual cortical region meshes as .msh files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    mesh_path = args.mesh
    m2m_dir = args.m2m
    output_dir = args.output_dir
    atlas_name = args.atlas
    surface = args.surface
    export_regions = not args.skip_regions
    export_whole_gm = not args.skip_whole_gm
    keep_meshes = args.keep_meshes

    if not mesh_path and not args.gm_mesh:
        logger.error("Must provide either --mesh (cortical surface) or --gm-mesh (tetrahedral GM)")
        return 1
    if args.gm_mesh:
        if not os.path.exists(args.gm_mesh):
            logger.error(f"GM mesh file not found: {args.gm_mesh}")
            return 1
        generated_surface = generate_cortical_surface_from_tetra(args.gm_mesh, m2m_dir, surface, args.msh2cortex)
        if not generated_surface:
            return 1
        mesh_path = generated_surface
    if not os.path.exists(mesh_path):
        logger.error(f"Mesh file not found: {mesh_path}")
        return 1
    if not os.path.isdir(m2m_dir):
        logger.error(f"m2m directory not found: {m2m_dir}")
        return 1
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_conversion(mesh_path, m2m_dir, output_dir, atlas_name,
                   export_regions, export_whole_gm, keep_meshes, args.min_triangles, args.segmentation_method)
    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Single-tool converter: generate atlas-accurate cortical region PLYs and a whole-GM PLY.

Features:
- Loads subject cortical surface mesh (.msh from msh2cortex) and subject atlas (default DKTatlas40)
- Exports individual region meshes as PLY
- Exports the full GM surface as a single PLY
- Optional: keep individual region meshes and whole GM mesh as .msh files
- Optional: sample a NIfTI field onto mesh nodes; colorize via colormap or store scalars
- Optional: global colormap normalization from NIfTI min/max
- Optional: generate a Blender import script

Usage examples:
    simnibs_python cortical_regions_to_ply.py \
        --mesh subject_central.msh \
        --m2m m2m_subject \
        --output-dir out \
        --atlas DKTatlas40 \
        --field-file subject_TI_max.nii.gz \
        --create-blender-script

    simnibs_python cortical_regions_to_ply.py \
        --mesh subject_central.msh \
        --m2m m2m_subject \
        --output-dir out \
        --global-from-nifti subject_TI_max.nii.gz

    simnibs_python cortical_regions_to_ply.py \
        --mesh subject_central.msh \
        --m2m m2m_subject \
        --output-dir out \
        --keep-meshes
"""

import argparse
import os
import sys
import logging
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
    from simnibs.mesh_tools.mesh_io import NodeData
except ImportError:
    print("Error: SimNIBS not found. Please install SimNIBS and activate the environment.")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel not found. Please install nibabel in the SimNIBS environment.")
    sys.exit(1)


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------- PLY Writers and Colormaps ----------

def write_ply_with_colors(vertices, faces, colors, output_path, field_name="TI_max"):
    n_vertices = len(vertices)
    n_faces = len(faces)
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Generated from SimNIBS mesh with {field_name} field\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n_vertices):
            x, y, z = vertices[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_ply_with_scalars(vertices, faces, scalars, output_path, field_name="TI_max"):
    n_vertices = len(vertices)
    n_faces = len(faces)
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Generated from SimNIBS mesh with {field_name} field\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"property float {field_name}\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n_vertices):
            x, y, z = vertices[i]
            s = scalars[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {s:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def simple_colormap(field_values, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(field_values)
    if vmax is None:
        vmax = np.nanmax(field_values)
    if vmax == vmin:
        colors = np.zeros((len(field_values), 3), dtype=np.uint8)
        colors[:, 2] = 255
        return colors
    normalized = (field_values - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    colors = np.zeros((len(field_values), 3), dtype=np.uint8)
    colors[:, 0] = (normalized * 255).astype(np.uint8)
    colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)
    return colors


def field_to_colormap(field_values, colormap='viridis', vmin=None, vmax=None):
    try:
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("Matplotlib not available, using simple blue-red colormap")
        return simple_colormap(field_values, vmin, vmax)
    if vmin is None:
        vmin = np.nanmin(field_values)
    if vmax is None:
        vmax = np.nanmax(field_values)
    if vmax == vmin:
        normalized = np.zeros_like(field_values)
    else:
        normalized = (field_values - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
    cmap = cm.get_cmap(colormap)
    colors_rgba = cmap(normalized)
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    return colors_rgb


# ---------- Field Utilities ----------

def calculate_global_field_range(field_file_path, mesh_file_path=None):
    try:
        nii = nib.load(field_file_path)
        field_data = nii.get_fdata()
        valid_data = field_data[field_data > 0]
        global_min = float(np.min(valid_data)) if valid_data.size else float(np.min(field_data))
        global_max = float(np.max(valid_data)) if valid_data.size else float(np.max(field_data))
        if mesh_file_path and os.path.exists(mesh_file_path):
            try:
                mesh = read_msh(mesh_file_path)
                if hasattr(mesh, 'nodedata') and len(mesh.nodedata) > 0:
                    for nodedata in mesh.nodedata:
                        if hasattr(nodedata, 'field_name'):
                            mesh_values = nodedata.value
                            mesh_pos = mesh_values[mesh_values > 0]
                            if mesh_pos.size:
                                mesh_min = float(np.min(mesh_pos))
                            else:
                                mesh_min = float(np.min(mesh_values))
                            mesh_max = float(np.max(mesh_values))
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


def apply_field_from_nifti(mesh, nifti_path, field_name="TI_max"):
    try:
        nii = nib.load(nifti_path)
        field_data = nii.get_fdata()
        affine = nii.affine
        node_coords = mesh.nodes.node_coord
        ones = np.ones((len(node_coords), 1))
        coords_homog = np.hstack([node_coords, ones])
        inv_affine = np.linalg.inv(affine)
        voxel_coords = (inv_affine @ coords_homog.T).T[:, :3]
        voxel_indices = np.round(voxel_coords).astype(int)
        field_values = np.zeros(len(node_coords))
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < field_data.shape[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < field_data.shape[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < field_data.shape[2])
        )
        valid_indices = voxel_indices[valid_mask]
        field_values[valid_mask] = field_data[
            valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        ]
        nodedata = NodeData(field_values, field_name)
        mesh.add_node_field(nodedata, field_name)
        return mesh
    except Exception as e:
        logger.error(f"Failed to apply field from NIfTI: {e}")
        return mesh


# ---------- Mesh/Atlas Utilities ----------

def extract_region_mesh(mesh, region_mask):
    try:
        nodes_to_keep = np.where(region_mask)[0]
        if nodes_to_keep.size == 0:
            return None
        region_mesh = mesh.crop_mesh(nodes=nodes_to_keep)
        return region_mesh
    except Exception:
        try:
            triangular_elements = mesh.elm.elm_type == 2
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
            return region_mesh
        except Exception:
            return None


def mesh_vertices_faces_and_field(mesh, field_name="TI_max"):
    triangles = mesh.elm[mesh.elm.elm_type == 2]
    if len(triangles) == 0:
        return None, None, None
    if hasattr(triangles, 'node_number_list'):
        triangle_nodes = triangles.node_number_list[:, :3] - 1
    else:
        triangle_nodes = triangles[:, :3] - 1
    unique_nodes = np.unique(triangle_nodes.flatten())
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
    vertices = mesh.nodes.node_coord[unique_nodes]
    faces = np.array([[node_map[idx] for idx in tri] for tri in triangle_nodes], dtype=np.int32)
    field_data = None
    if hasattr(mesh, 'nodedata') and len(mesh.nodedata) > 0:
        field_idx = None
        for i, nd in enumerate(mesh.nodedata):
            if hasattr(nd, 'field_name') and nd.field_name == field_name:
                field_idx = i
                break
        if field_idx is not None:
            field_full = mesh.nodedata[field_idx].value
            field_data = field_full[unique_nodes]
    return vertices, faces, field_data




# ---------- Main Orchestration ----------

def export_mesh_to_ply(mesh, ply_path, field_name, use_colors, colormap, field_range):
    vertices, faces, field_data = mesh_vertices_faces_and_field(mesh, field_name)
    if vertices is None or faces is None:
        logger.error("Mesh has no triangular elements to export")
        return False
    if field_data is None:
        if use_colors:
            colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
            write_ply_with_colors(vertices, faces, colors, ply_path, field_name)
        else:
            scalars = np.zeros(len(vertices))
            write_ply_with_scalars(vertices, faces, scalars, ply_path, field_name)
        return True
    vmin, vmax = field_range if field_range else (None, None)
    if use_colors:
        colors = field_to_colormap(field_data, colormap, vmin, vmax)
        write_ply_with_colors(vertices, faces, colors, ply_path, field_name)
    else:
        write_ply_with_scalars(vertices, faces, field_data, ply_path, field_name)
    return True


def run_conversion(mesh_path, m2m_dir, output_dir, atlas_name, field_file, field_name,
                   use_colors, colormap, field_range, global_from_nifti,
                   export_regions, export_whole_gm, keep_meshes):
    mesh = read_msh(mesh_path)
    logger.info(f"Loaded mesh: {mesh_path}")
    atlas = subject_atlas(atlas_name, str(m2m_dir))
    logger.info(f"Loaded atlas '{atlas_name}' with {len(atlas.keys())} regions")

    regions_out_dir = Path(output_dir) / "regions"
    regions_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create meshes directory if keeping meshes
    if keep_meshes:
        meshes_out_dir = Path(output_dir) / "meshes"
        meshes_out_dir.mkdir(parents=True, exist_ok=True)

    if field_file:
        logger.info(f"Sampling field from NIfTI for whole mesh and regions: {field_file}")
        mesh = apply_field_from_nifti(mesh, field_file, field_name)

    effective_field_range = None
    if field_range is not None:
        effective_field_range = tuple(field_range)
    elif global_from_nifti:
        vmin, vmax = calculate_global_field_range(global_from_nifti, mesh_path)
        if vmin is None or vmax is None:
            logger.error("Failed to compute global field range from NIfTI")
        else:
            effective_field_range = (vmin, vmax)
            logger.info(f"Using global colormap range: [{vmin:.6f}, {vmax:.6f}]")

    if export_regions:
        success_count = 0
        mesh_success_count = 0
        for region_name, region_mask in atlas.items():
            region_mesh = extract_region_mesh(mesh, region_mask)
            if region_mesh is None:
                logger.warning(f"Skipping region with no mesh: {region_name}")
                continue
            if field_file and (field_name not in [nd.field_name for nd in getattr(region_mesh, 'nodedata', []) if hasattr(nd, 'field_name')]):
                region_mesh = apply_field_from_nifti(region_mesh, field_file, field_name)
            
            # Export PLY
            ply_path = regions_out_dir / f"{region_name}_region.ply"
            if export_mesh_to_ply(region_mesh, str(ply_path), field_name, use_colors, colormap, effective_field_range):
                success_count += 1
            
            # Export MSH if requested
            if keep_meshes:
                msh_path = meshes_out_dir / f"{region_name}_region.msh"
                try:
                    region_mesh.write(str(msh_path))
                    mesh_success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save mesh for region {region_name}: {e}")
        
        logger.info(f"Converted {success_count}/{len(atlas.keys())} regions to PLY")
        if keep_meshes:
            logger.info(f"Saved {mesh_success_count}/{len(atlas.keys())} region meshes as MSH")

    if export_whole_gm:
        whole_ply = Path(output_dir) / "whole_gm.ply"
        export_mesh_to_ply(mesh, str(whole_ply), field_name, use_colors, colormap, effective_field_range)
        logger.info(f"Wrote whole GM PLY: {whole_ply}")
    
    # Export whole GM mesh if requested
    if keep_meshes and export_whole_gm:
        whole_msh = Path(output_dir) / "whole_gm.msh"
        try:
            mesh.write(str(whole_msh))
            logger.info(f"Wrote whole GM MSH: {whole_msh}")
        except Exception as e:
            logger.warning(f"Failed to save whole GM mesh: {e}")


def _resolve_msh2cortex(explicit_path: str | None) -> str | None:
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
    parser = argparse.ArgumentParser(description="Export cortical regions and whole GM surface to PLY")
    parser.add_argument("--mesh", help="Input cortical surface mesh (.msh) from msh2cortex")
    parser.add_argument("--gm-mesh", help="Input tetrahedral GM .msh (volumetric); will run msh2cortex")
    parser.add_argument("--m2m", required=True, help="Subject m2m directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for PLY files")
    parser.add_argument("--atlas", default="DK40", help="Atlas name (default: DK40)")
    parser.add_argument("--surface", default="central", choices=["central", "pial", "white"], help="Cortical surface to extract when using --gm-mesh (default: central)")
    parser.add_argument("--msh2cortex", help="Path to msh2cortex executable (optional override)")
    parser.add_argument("--field-file", help="NIfTI file to sample field values onto nodes")
    parser.add_argument("--field", default="TI_max", help="Field name to use/store (default: TI_max)")
    parser.add_argument("--scalars", action="store_true", help="Store field as scalars instead of colors")
    parser.add_argument("--colormap", default="viridis", help="Colormap for colors mode")
    parser.add_argument("--field-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Explicit field range for mapping")
    parser.add_argument("--global-from-nifti", help="Use global min/max from this NIfTI for color mapping")
    parser.add_argument("--skip-regions", action="store_true", help="Do not export individual region PLYs")
    parser.add_argument("--skip-whole-gm", action="store_true", help="Do not export the whole GM PLY")
    parser.add_argument("--keep-meshes", action="store_true", help="Keep individual cortical region meshes as .msh files")
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_path = args.mesh
    m2m_dir = args.m2m
    output_dir = args.output_dir
    atlas_name = args.atlas
    surface = args.surface
    field_file = args.field_file
    field_name = args.field
    use_colors = not args.scalars
    colormap = args.colormap
    field_range = tuple(args.field_range) if args.field_range else None
    global_from_nifti = args.global_from_nifti
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

    run_conversion(mesh_path, m2m_dir, output_dir, atlas_name, field_file, field_name,
                   use_colors, colormap, field_range, global_from_nifti,
                   export_regions, export_whole_gm, keep_meshes)
    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())



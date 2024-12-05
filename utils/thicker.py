import trimesh

# Load your original mesh
path = "data/mesh/3742_nocutedge.obj"
import open3d as o3d
import numpy as np

# Load your original mesh
mesh = o3d.io.read_triangle_mesh(path)
# Compute vertex normals if not already computed
mesh.compute_vertex_normals()

# Specify the thickness you want to add (positive for outward, negative for inward)
thickness = 0.05  # Adjust this value as needed

# Create a copy of the original mesh
# thicker_mesh = mesh.clone()

# Offset each vertex along its normal
vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)
offset_vertices = vertices + thickness * normals
mesh.vertices = o3d.utility.Vector3dVector(offset_vertices)

# Calculate face normals
o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)

# Convert face normals to vertex normals
mesh.compute_vertex_normals()

# [Open3D WARNING] Write OBJ can not include triangle normals.
mesh.triangle_normals = o3d.utility.Vector3dVector([])

# Visualize the result
# o3d.visualization.draw_geometries([thicker_mesh])
o3d.io.write_triangle_mesh('thicker.obj', mesh)
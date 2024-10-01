"""
    This file creates the mean, max, and min mesh from a ED, ES, and ED+ES dataset that can be used as a template mesh for further processing
"""
import open3d as o3d
import numpy as np
import os

#
# I/O
#
# random case just to get connectivity info which is the same across the dataset
# template_mesh_path = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/meshes_jorge_ED_ply/case_1_nstemi_ED.ply"
# template_mesh_path = "/home/vaen/Desktop/mesh_diffusion/data/german_cohort/NSTEMI-0001-ED.ply"
template_mesh_path = "/home/vaen/Desktop/mesh_diffusion/data/meshes_german_cohort_preprocessed/case_1_nstemi_ED.ply"

# mean_mesh_path = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/mean_mesh.ply"
# min_mesh_path = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/min_mesh.ply"
# max_mesh_path = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/max_mesh.ply"
mean_mesh_path = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/mean_mesh.ply"
min_mesh_path = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/min_mesh.ply"
max_mesh_path = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/max_mesh.ply"

# mean_mesh_path_ED = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/mean_mesh_ED.ply"
# min_mesh_path_ED = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/min_mesh_ED.ply"
# max_mesh_path_ED = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/max_mesh_ED.ply"
mean_mesh_path_ED = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/mean_mesh_ED.ply"
min_mesh_path_ED = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/min_mesh_ED.ply"
max_mesh_path_ED = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/max_mesh_ED.ply"

# mean_mesh_path_ES = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/mean_mesh_ES.ply"
# min_mesh_path_ES = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/min_mesh_ES.ply"
# max_mesh_path_ES = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/max_mesh_ES.ply"
mean_mesh_path_ES = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/mean_mesh_ES.ply"
min_mesh_path_ES = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/min_mesh_ES.ply"
max_mesh_path_ES = "/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/max_mesh_ES.ply"

# in_dir_ED = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/meshes_jorge_ED_ply"
# in_dir_ES = "/home/mars007/data/meshes_jorge_original_format/mesh_unet_data/meshes_jorge_ES_ply"
# in_dir_ED = "/home/vaen/Desktop/mesh_diffusion/data/german_cohort_ED"
# in_dir_ES = "/home/vaen/Desktop/mesh_diffusion/data/german_cohort_ES"
in_dir_ED = "/home/vaen/Desktop/mesh_diffusion/data/gh_preprocessed_ED"
in_dir_ES = "/home/vaen/Desktop/mesh_diffusion/data/gh_preprocessed_ES"


#
# Read meshes and get vertex coordinate values
#
mesh_vertices_list, mesh_vertices_list_ED, mesh_vertices_list_ES = [], [], []

for filename in os.listdir(in_dir_ED):
    heart_mesh = o3d.io.read_triangle_mesh(os.path.join(in_dir_ED, filename))
    heart_mesh_vertices = np.asarray(heart_mesh.vertices)
    mesh_vertices_list.append(heart_mesh_vertices)
    mesh_vertices_list_ED.append(heart_mesh_vertices)

for filename in os.listdir(in_dir_ES):
    heart_mesh = o3d.io.read_triangle_mesh(os.path.join(in_dir_ES, filename))
    heart_mesh_vertices = np.asarray(heart_mesh.vertices)
    mesh_vertices_list.append(heart_mesh_vertices)
    mesh_vertices_list_ES.append(heart_mesh_vertices)


#
# Calcualte mean, max, and min vertex coordinate values
#

# ED+ES data
mesh_vertices_array = np.asarray(mesh_vertices_list)
mean_mesh_vertices = np.mean(mesh_vertices_array, axis=0)
min_mesh_vertices = np.min(mesh_vertices_array, axis=0)
max_mesh_vertices = np.max(mesh_vertices_array, axis=0)

# ED data
mesh_vertices_array_ED = np.asarray(mesh_vertices_list_ED)
mean_mesh_vertices_ED = np.mean(mesh_vertices_array_ED, axis=0)
min_mesh_vertices_ED = np.min(mesh_vertices_array_ED, axis=0)
max_mesh_vertices_ED = np.max(mesh_vertices_array_ED, axis=0)

# ES data
mesh_vertices_array_ES = np.asarray(mesh_vertices_list_ES)
mean_mesh_vertices_ES = np.mean(mesh_vertices_array_ES, axis=0)
min_mesh_vertices_ES = np.min(mesh_vertices_array_ES, axis=0)
max_mesh_vertices_ES = np.max(mesh_vertices_array_ES, axis=0)



#
# Read mesh of random dummy case to get connectivity info, update its vertex coordinates, and save it
#

# ED+ES data
mean_mesh = o3d.io.read_triangle_mesh(template_mesh_path)
mean_mesh.vertices = o3d.utility.Vector3dVector(mean_mesh_vertices)
o3d.io.write_triangle_mesh(mean_mesh_path, mean_mesh)

min_mesh = o3d.io.read_triangle_mesh(template_mesh_path)
min_mesh.vertices = o3d.utility.Vector3dVector(min_mesh_vertices)
o3d.io.write_triangle_mesh(min_mesh_path, min_mesh)

max_mesh = o3d.io.read_triangle_mesh(template_mesh_path)
max_mesh.vertices = o3d.utility.Vector3dVector(max_mesh_vertices)
o3d.io.write_triangle_mesh(max_mesh_path, max_mesh)


# ED data
mean_mesh_ED = o3d.io.read_triangle_mesh(template_mesh_path)
mean_mesh_ED.vertices = o3d.utility.Vector3dVector(mean_mesh_vertices_ED)
o3d.io.write_triangle_mesh(mean_mesh_path_ED, mean_mesh_ED)

min_mesh_ED = o3d.io.read_triangle_mesh(template_mesh_path)
min_mesh_ED.vertices = o3d.utility.Vector3dVector(min_mesh_vertices_ED)
o3d.io.write_triangle_mesh(min_mesh_path_ED, min_mesh_ED)

max_mesh_ED = o3d.io.read_triangle_mesh(template_mesh_path)
max_mesh_ED.vertices = o3d.utility.Vector3dVector(max_mesh_vertices_ED)
o3d.io.write_triangle_mesh(max_mesh_path_ED, max_mesh_ED)


# ES data
mean_mesh_ES = o3d.io.read_triangle_mesh(template_mesh_path)
mean_mesh_ES.vertices = o3d.utility.Vector3dVector(mean_mesh_vertices_ES)
o3d.io.write_triangle_mesh(mean_mesh_path_ES, mean_mesh_ES)

min_mesh_ES = o3d.io.read_triangle_mesh(template_mesh_path)
min_mesh_ES.vertices = o3d.utility.Vector3dVector(min_mesh_vertices_ES)
o3d.io.write_triangle_mesh(min_mesh_path_ES, min_mesh_ES)

max_mesh_ES = o3d.io.read_triangle_mesh(template_mesh_path)
max_mesh_ES.vertices = o3d.utility.Vector3dVector(max_mesh_vertices_ES)
o3d.io.write_triangle_mesh(max_mesh_path_ES, max_mesh_ES)


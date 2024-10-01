import numpy as np
import open3d as o3d
import os

def average_meshes(mesh_dir, output_file):
    mesh_files = os.listdir(mesh_dir)
    vertices_all_meshes = []

    for mesh_file in mesh_files:
        mesh = o3d.io.read_triangle_mesh(mesh_dir + mesh_file)
        vertices = np.asarray(mesh.vertices)
        vertices_all_meshes.append(vertices)

    vertices_all_meshes = np.array(vertices_all_meshes)
    print("Shape of vertices_all_meshes: ", vertices_all_meshes.shape)

    # get mean of the mesh vertices
    mean_vertices = np.mean(vertices_all_meshes, axis=0)
    print("Shape of mean_vertices: ", mean_vertices.shape)

    # load a randomly chosen mesh to get the mesh structure
    random_mesh_file = "/home/vaen/Desktop/mesh_diffusion/data/gt_lv_endo_ED_mesh_0_0.ply"
    mean_mesh = o3d.io.read_triangle_mesh(random_mesh_file)
    mean_mesh.vertices = o3d.utility.Vector3dVector(mean_vertices)

    # visualise mean_mesh
    print("Visualising mean_mesh")
    o3d.visualization.draw_geometries([mean_mesh])
    o3d.io.write_triangle_mesh(output_file, mean_mesh)


def convert_ply_to_obj(input_file, output_file):
    mesh = o3d.io.read_triangle_mesh(input_file)

    o3d.io.write_triangle_mesh(output_file,
                               mesh,
                               write_triangle_uvs=True)

    # reread_mesh = o3d.io.read_triangle_mesh(output_file)
    # o3d.visualization.draw_geometries([reread_mesh])


if __name__ == '__main__':
    # average_meshes(mesh_dir="/home/vaen/Desktop/mesh_diffusion/data/marcel_synthetic_data/",
    #                output_file="/home/vaen/Desktop/mesh_diffusion/data/test_mean_mesh.ply")

    # convert_ply_to_obj(input_file="/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/mean_mesh.ply",
    #                     output_file="/home/vaen/Desktop/mesh_diffusion/data/no_disjoint_template/mean_mesh.obj")

    # convert_ply_to_obj(input_file="/home/vaen/Desktop/mesh_diffusion/data/not_used_marcel_synthetic_data/gt_lv_endo_ED_mesh_58_0.ply",
    #                 output_file="/home/vaen/Desktop/mesh_diffusion/data/not_used_marcel_synthetic_data/gt_lv_endo_ED_mesh_58_0.obj")

    convert_ply_to_obj(input_file="/home/vaen/Desktop/mesh_diffusion/data/gc_preprocessed_template/mean_mesh_ES.ply",
                    output_file="/home/vaen/Desktop/mesh_diffusion/data/gc_preprocessed_template/mean_mesh_ES.obj")
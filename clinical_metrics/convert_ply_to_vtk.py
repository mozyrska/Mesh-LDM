import subprocess
import meshio
import os

# Uncomment for ES
# ply_dir = "path/to/the/project/MeshLDM_publication/data/decoded/ES"
# Uncomment for ED
ply_dir = "path/to/the/project/data/decoded/ED"
vtk_dir = "./all_samples/vtk"

if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

for case in range(0, 1000):
    ply_file = ply_dir + "/case_" + str(case) + ".ply"
    vtk_file = vtk_dir + "/case_" + str(case) + ".vtk"
    mesh = meshio.read(ply_file)
    meshio.write(vtk_file, mesh, binary=False)
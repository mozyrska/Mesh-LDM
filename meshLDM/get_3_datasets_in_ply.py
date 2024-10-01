import argparse
import os
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import time
from psbody.mesh import Mesh
import mesh_operations
from config_parser import read_config_encode_decode
from data import MeshDataset
from model import MeshVAE
from transform import Normalize

from diffusion_utils import scipy_to_torch_sparse

def main(args):
    # Read config file
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    config = read_config_encode_decode(args.conf, "encode")

    # Read template mesh
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    workers_thread = config['workers_thread']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    # Calculate transforms for mesh sampling
    #

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    #
    # Load dataset
    #

    print('Loading Dataset')
    data_dir = config['data_dir']

    normalize_transform = Normalize()

    dataset = MeshDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_val = MeshDataset(data_dir, dtype='val', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_test = MeshDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers_thread)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    # UNCOMMENT FOR TEST DATASET
    dataset_test_std_numpy = dataset_test.std.numpy()
    dataset_test_mean_numpy = dataset_test.mean.numpy()

    # params to set
    gt_mesh_filepath = "/home/vaen/Desktop/mesh_diffusion/data/gc_preprocessed_ED/test/case_"

    for i, data in enumerate(test_loader):
        expected_out = (data.y.numpy()) * dataset_test_std_numpy + dataset_test_mean_numpy
        gt_mesh = o3d.io.read_triangle_mesh(template_file_path)
        gt_vertices = np.asarray(expected_out)
        gt_mesh.vertices = o3d.utility.Vector3dVector(gt_vertices)
        o3d.io.write_triangle_mesh(gt_mesh_filepath + str(i) + ".ply", gt_mesh)


    # UNCOMMENT FOR VALIDATION DATASET
    # dataset_val_std_numpy = dataset_val.std.numpy()
    # dataset_val_mean_numpy = dataset_val.mean.numpy()

    # # params to set
    # gt_mesh_filepath = "/home/vaen/Desktop/mesh_diffusion/data/gc_preprocessed_ES/val/case_"

    # for i, data in enumerate(val_loader):
    #     expected_out = (data.y.numpy()) * dataset_val_std_numpy + dataset_val_mean_numpy
    #     gt_mesh = o3d.io.read_triangle_mesh(template_file_path)
    #     gt_vertices = np.asarray(expected_out)
    #     gt_mesh.vertices = o3d.utility.Vector3dVector(gt_vertices)
    #     o3d.io.write_triangle_mesh(gt_mesh_filepath + str(i) + ".ply", gt_mesh)

    # z_data_train, y_data_train = encode(mesh_vae, train_loader, device)
    # encoded_output_dir = config['encoded_output_dir']
    # with open(encoded_output_dir + '/train/z_data.npy', 'wb') as f:
    #     np.save(f, z_data_train)
    # with open(encoded_output_dir + '/train/y_data.npy', 'wb') as f:
    #     np.save(f, y_data_train)


    # z_data_val, y_data_val = encode(mesh_vae, val_loader, device)
    # with open(encoded_output_dir + '/val/z_data.npy', 'wb') as f:
    #     np.save(f, z_data_val)
    # with open(encoded_output_dir + '/val/y_data.npy', 'wb') as f:
    #     np.save(f, y_data_val)

    # z_data_test, y_data_test = encode(mesh_vae, test_loader, device)
    # with open(encoded_output_dir + '/test/z_data.npy', 'wb') as f:
    #     np.save(f, z_data_test)
    # with open(encoded_output_dir + '/test/y_data.npy', 'wb') as f:
    #     np.save(f, y_data_test)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced',
                        help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced',
                        help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir',
                        help='path where the downloaded data is stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'encode_decode.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
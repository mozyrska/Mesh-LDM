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


def main(args, internal_args):
    # Read config file
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    config = read_config_encode_decode(args.conf, "decode")

    # Read template mesh
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

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

    #
    # Load model
    #

    print('Loading model')
    start_epoch = 1
    mesh_vae = MeshVAE(dataset, config, D_t, U_t, A_t, num_nodes)

    # Load model checkpoint if set
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        mesh_vae.load_state_dict(checkpoint['state_dict'])

    mesh_vae.to(device)
    mesh_vae.eval()

    dataset_std_numpy = dataset.std.numpy()
    dataset_mean_numpy = dataset.mean.numpy()
    dataset_std_device = dataset.std.to(device)
    dataset_mean_device = dataset.mean.to(device)

    with open(config['denoised_dir'], 'rb') as f:
        z_prim = torch.tensor(np.load(f))

    z_prim = z_prim.to(device)

    for i in range(z_prim.shape[0]):

        with torch.no_grad():
            # print("z_prim[i].shape: ", z_prim[i].shape)
            sample = (z_prim[i])[None, :]
            out = mesh_vae.get_mesh(sample)

        # print("Saving output")
        # Detach and denormalize data
        save_out = out.detach().cpu().numpy()

        save_out = save_out*dataset_std_numpy+dataset_mean_numpy
        pred = (out.detach())*dataset_std_device+dataset_mean_device

        pred_mesh_filepath = os.path.join(config['decoded_output_dir'], "case_" + str(i) + "_sample.ply")

        pred_mesh = o3d.io.read_triangle_mesh(template_file_path)
        pred_vertices = np.asarray(save_out)
        pred_mesh.vertices = o3d.utility.Vector3dVector(pred_vertices)
        o3d.io.write_triangle_mesh(pred_mesh_filepath, pred_mesh)

    return


if __name__ == '__main__':
    internal_args = dict()
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

    main(args, internal_args)
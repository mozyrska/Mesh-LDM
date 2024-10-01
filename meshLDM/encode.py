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


def encode(mesh_vae, loader, device):
    z_data = []
    y_data = []

    # Switch to eval mode
    mesh_vae.eval()

    for i, data in enumerate(loader):
        # Move to device
        data = data.to(device)
        with torch.no_grad():
            out = mesh_vae.get_z(data)

        # Detach and denormalize data
        save_out = out.detach().cpu().numpy()
        save_y = data.y.detach().cpu().numpy()

        z_data.append(save_out)
        y_data.append(save_y)

    return z_data, y_data


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

    #
    # Load model
    #

    # Load model
    print('Loading model')
    start_epoch = 1
    mesh_vae = MeshVAE(dataset, config, D_t, U_t, A_t, num_nodes)
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        mesh_vae.load_state_dict(checkpoint['state_dict'])

    mesh_vae.to(device)

    print('Encoding meshes into the latent space')
    z_data_train, y_data_train = encode(mesh_vae, train_loader, device)

    print('Saving encoded data')
    encoded_output_dir = config['encoded_output_dir']
    with open(encoded_output_dir + '/train/z_data.npy', 'wb') as f:
        np.save(f, z_data_train)
    with open(encoded_output_dir + '/train/y_data.npy', 'wb') as f:
        np.save(f, y_data_train)

    z_data_val, y_data_val = encode(mesh_vae, val_loader, device)
    with open(encoded_output_dir + '/val/z_data.npy', 'wb') as f:
        np.save(f, z_data_val)
    with open(encoded_output_dir + '/val/y_data.npy', 'wb') as f:
        np.save(f, y_data_val)

    z_data_test, y_data_test = encode(mesh_vae, test_loader, device)
    with open(encoded_output_dir + '/test/z_data.npy', 'wb') as f:
        np.save(f, z_data_test)
    with open(encoded_output_dir + '/test/y_data.npy', 'wb') as f:
        np.save(f, y_data_test)

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
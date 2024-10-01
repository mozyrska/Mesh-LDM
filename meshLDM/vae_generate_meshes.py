import argparse
import os
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh
import mesh_operations
from data import MeshDataset
from model import MeshVAE
from transform import Normalize

from config_parser import read_config_encode_decode


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config_encode_decode(args.conf, "decode")

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset = MeshDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)

    print('Loading model')
    mesh_vae = MeshVAE(dataset, config, D_t, U_t, A_t, num_nodes)

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        mesh_vae.load_state_dict(checkpoint['state_dict'])
    mesh_vae.to(device)

    generate_meshes(mesh_vae, dataset, device)
    return


def generate_meshes(mesh_vae, dataset, device):
    print("Generate meshes with pretrained model")
    mesh_vae.eval()

    num_gen_meshes = 1000
    for i in range(num_gen_meshes):
        with torch.no_grad():

            print("Create latent space vector")
            sample_mode = "standard_gaussian"
            print("Sample mode: {}".format(sample_mode))
            latent_dim = 16

            if sample_mode == "train_data_metrics_ES":
                means = np.array([-0.0494,  1.5750,  2.2315,  1.8140,  1.8572, -0.0514,  1.7915,  1.8904, 1.9286, -0.0092,  1.6116,  1.9383,  2.1109,  1.9209,  1.9336,  2.0033])
                means = torch.from_numpy(means)

                stds = np.array([1.0329, 2.0033, 2.9584, 3.1641, 2.2091, 0.9602, 2.8612, 2.6477, 2.9608, 0.9766, 2.1016, 2.5876, 2.2940, 2.5560, 2.0430, 2.4994])

                stds = torch.from_numpy(stds)
                stds = torch.exp(0.5*stds)

            elif sample_mode == "train_data_metrics_ED":
                means = np.array([-0.0086,  2.5938,  0.0278,  3.4601,  3.2333,  3.3586,  0.0415,  0.0328, 2.4860, -0.0790, -0.0303,  0.0653,  3.4665, 3.2370, 0.0225, -0.0159])
                means = torch.from_numpy(means)

                stds = np.array([0.9634, 2.6815, 1.0325, 3.3553, 3.2094, 3.3775, 1.0034, 0.9920, 2.7689, 1.0292, 1.0005, 1.0040, 3.3220, 3.5437, 0.9682, 1.0255])

                stds = torch.from_numpy(stds)
                stds = torch.exp(0.5*stds)

            elif sample_mode == "original_mesh_vae_train_data_metrics":
                means = np.array([0.0000, 1.9822, 2.6226,
                                  2.1887, 2.6137, 2.9143,
                                  2.4811, 2.2930, 0.0000,
                                  2.4803, 2.3890, 0.0000,
                                  1.9581, 2.1651, 0.0000,
                                  0.0000])
                means = torch.from_numpy(means)
                stds = np.array([0.0000, 2.2555, 2.3448, 2.5397, 2.5697, 3.1372, 2.3823, 2.3025, 0.0000, 3.3148, 2.8080, 0.0000, 2.9010, 2.1681, 0.0000, 0.0000])
                stds = torch.from_numpy(stds)
                stds = torch.exp(0.5*stds)

            elif sample_mode == "standard_gaussian":
                means = torch.zeros(latent_dim)
                stds = torch.ones(latent_dim)


            latent_z_vec = torch.normal(means,stds)

            print(latent_z_vec)
            print(latent_z_vec.shape)
            latent_z_vec = latent_z_vec[None, ...]
            print(latent_z_vec.shape)
            print(latent_z_vec.dtype)
            latent_z_vec = latent_z_vec.to(torch.float32)
            print(latent_z_vec.dtype)

            print("Get decoder output")
            out = mesh_vae.decoder(latent_z_vec)
            print(out.shape)
            print("Saving generated output")
            save_out = out.detach().cpu().numpy()

            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            save_out = save_out[0,...]
            print(save_out.shape)
            print(type(save_out))

            template_mesh_filepath = "/home/vaen/Desktop/MeshLDM_publication/data/gc_preprocessed_template/mean_mesh_ES.ply"
            pred_mesh_filepath = "/home/vaen/Desktop/mesh_diffusion/data/vae/ES/case_" + str(i) + ".ply"


            pred_mesh = o3d.io.read_triangle_mesh(template_mesh_filepath)
            pred_vertices = np.asarray(save_out)
            pred_mesh.vertices = o3d.utility.Vector3dVector(pred_vertices)
            o3d.io.write_triangle_mesh(pred_mesh_filepath, pred_mesh)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='sliced', help='split can be sliced, expression or identity ')
    parser.add_argument('-st', '--split_term', default='sliced', help='split term can be sliced, expression name '
                                                               'or identity name')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)

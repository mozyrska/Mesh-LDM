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
from config_parser import read_config
from data import MeshDataset
from model import MeshVAE
from transform import Normalize

#
# Define helper functions
#

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = mesh_vae.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def main(args, internal_args):

    #
    # I/O
    #

    # Read config file
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    config = read_config(args.conf)

    # Read template mesh
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    # Set checkpoint dir
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Set output dirs
    visualize = config['visualize']
    # output_dir = config['visual_output_dir']
    # if visualize is True and not output_dir:
        # print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
    #     output_dir = checkpoint_dir
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Get other config settings
    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    # Move to device
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
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset = MeshDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)

    dataset_test = MeshDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    #
    # Load model
    #

    # Load model
    print('Loading model')
    start_epoch = 1
    mesh_vae = MeshVAE(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(mesh_vae.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(mesh_vae.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    # Load model checkpoint if set
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        mesh_vae.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    mesh_vae.to(device)

    #
    # Eval pre-trained model if set
    #

    if eval_flag:
        dataset = MeshDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
        dataset_test = MeshDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers_thread)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)
        val_loss_train_dataset = evaluate(mesh_vae, train_loader, dataset, device, internal_args, config, visualize)

        val_loss = evaluate(mesh_vae, test_loader, dataset_test, device, internal_args, config, visualize)

        print('\nTraining loss', val_loss_train_dataset)
        print('Validation loss', val_loss)
        return


    #
    # Start network training
    #

    # Set training schedule for loss weighting parameter beta
    beta_values = [0.0001, 0.00025, 0.0005, 0.00075, 0.001]
    beta_epoch_changes = [50, 100, 150, 200]
    beta = beta_values[0]

    # Iterate over epochs
    best_val_loss = float('inf')
    val_loss_history = []
    for epoch in range(start_epoch, total_epochs + 1):

        # Adjust beta based on current epoch
        if epoch == beta_epoch_changes[0]:
            beta = beta_values[1]
        elif epoch == beta_epoch_changes[1]:
            beta = beta_values[2]
        elif epoch == beta_epoch_changes[2]:
            beta = beta_values[3]
        elif epoch == beta_epoch_changes[3]:
            beta = beta_values[4]

        # Train network
        epoch_start = time.time()
        print("Training for epoch ", epoch)
        train_loss_total, train_loss, train_loss_l1, train_loss_kl = train(mesh_vae, train_loader, len(dataset), optimizer, device, beta)
        print('epoch ', epoch,' Train loss ', train_loss, ' beta ', beta)
        print(' Train L1 loss ', train_loss_l1)
        print(' Train KL loss ', train_loss_kl)
        epoch_end = time.time()
        print('Epoch execution time: {}'.format(epoch_end-epoch_start))


        # Eval current model checkpoint
        if epoch % 25 == 0:

            val_loss = evaluate(mesh_vae, test_loader, dataset_test, device, internal_args, config, visualize=visualize)
            print('epoch ', epoch,' Train loss ', train_loss, ' Val loss ', val_loss)
            save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss
            val_loss_history.append(val_loss)
            val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()



def train(mesh_vae, train_loader, len_dataset, optimizer, device, beta):

    # Switch to train mode
    mesh_vae.train()

    # Iterate through train data
    total_loss = 0
    for data in train_loader:

        # Move to device
        data = data.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Pass through network
        reconstruction, mu, log_var, z = mesh_vae(data)

        # Calculate loss
        loss_l1 = F.mse_loss(reconstruction, data.y)
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)
        loss_kl = torch.mean(loss_kl)
        loss = loss_l1 + beta*loss_kl
        total_loss += loss.item()

        # Backprop
        loss.backward()
        optimizer.step()

    return total_loss / len_dataset, loss, loss_l1, loss_kl



def evaluate(mesh_vae, test_loader, dataset, device, internal_args, config, visualize=False):

    # Switch to eval mode
    mesh_vae.eval()

    # Iterate through eval data
    total_loss = 0

    dataset_std_numpy = dataset.std.numpy()
    dataset_mean_numpy = dataset.mean.numpy()
    dataset_std_device = dataset.std.to(device)
    dataset_mean_device = dataset.mean.to(device)

    for i, data in enumerate(test_loader):

        # Move to device
        data = data.to(device)
        with torch.no_grad():
            out, _, _, _ = mesh_vae(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()


        if visualize:

            print("Saving output")

            # Detach and denormalize data
            save_out = out.detach().cpu().numpy()

            save_out = save_out*dataset_std_numpy+dataset_mean_numpy
            expected_out = (data.y.detach().cpu().numpy())*dataset_std_numpy+dataset_mean_numpy
            pred = (out.detach())*dataset_std_device+dataset_mean_device
            gt = (data.y.detach())*dataset_std_device+dataset_mean_device

            # Calculate eval loss
            eval_l1_loss = F.l1_loss(pred, gt)
            print("Current eval L1 loss: {}".format(eval_l1_loss))

            # Set output filepath

            template_mesh_filepath = config['template_fname']

            pred_mesh_filepath = os.path.join(out_basedir, "case_" + str(i) + "_pred.ply")
            gt_mesh_filepath = os.path.join(out_basedir, "case_" + str(i) + "_gt.ply")

            # Save predicted mesh
            pred_mesh = o3d.io.read_triangle_mesh(template_mesh_filepath)
            pred_vertices = np.asarray(save_out)
            pred_mesh.vertices = o3d.utility.Vector3dVector(pred_vertices)
            o3d.io.write_triangle_mesh(pred_mesh_filepath, pred_mesh)

            # Save ground truth mesh
            gt_mesh = o3d.io.read_triangle_mesh(template_mesh_filepath)
            gt_vertices = np.asarray(expected_out)
            gt_mesh.vertices = o3d.utility.Vector3dVector(gt_vertices)
            o3d.io.write_triangle_mesh(gt_mesh_filepath, gt_mesh)


    return total_loss/len(dataset)



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
    parser.add_argument('-cp', '--checkpoint_dir',
                        help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args, internal_args)

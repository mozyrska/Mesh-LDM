# ✨Cardiac anatomy generation with Latent Diffusion Models✨

## Requirements
Following the requirements provided by Beetz et al. for Mesh VAE: https://github.com/marcel-beetz/cardiac-mesh-vae

1. Install the mesh processing libraries described in the following Github repository: https://github.com/MPI-IS/mesh
2. Create a new virtual environment and install required packages:
* virtualenv mesh_vae_venv
* source mesh_vae_venv/bin/activate
* pip install -r mesh_vae_requirements.txt


## Mesh LDM
You can train the model on end-diastolic (ED) data or end-systolic (ES) data.

1. Adjust the following parameters in main.cfg depending on the data (ED or ES) you want to use:
    1. data_dir - where dataset is located
    2. checkpoint_dir - where checkpoints will be saved
    3. template_fname - a blueprint mesh used for processing meshes (it is already provided and it was created with the following code: data_processing/template_mesh.py)
    4. checkpoint_file - pretrained model


2. Training Mesh VAE (the autoencoder part of the Mesh LDM):
    1. In main.cfg change parameter: eval=False
    2. python meshLDM/main.py --conf main.cfg


3. Evaluate Mesh VAE (the autoencoder part of the Mesh LDM):
    1. In main.cfg change parameter: eval=True
    2. python meshLDM/main.py --conf main.cfg


4. Encode meshes into latent space using the encoder from VAE:
    1. Adjust paths in encode_decode.cfg:
        1. data_dir - path where dataset is located
        2. checkpoint_dir - path where checkpoints will be saved
        3. template_fname - a blueprint mesh used for processing meshes (it is already provided and it was created with the following code: data_processing/template_mesh.py)
        4. mesh_to_fill_path - a random mesh used for getting the mesh structure (already provided)
        5. checkpoint_file - pretrained model
        6. encoded_output_dir - where the encoded data will be saved
        7. denoised_dir - where denoised data will be read from in the decoding phase
        8. decoded_output_dir - where the decoded data (3D meshes) will be saved

    2. python meshLDM/encode.py --conf encode_decode.cfg


5. Train a denoising model:
    Code: diffusion.ipynb. It was run in a code editor using a separate virtual environment (Why? Some versions of libraries used for Mesh VAE aren't compatible with the diffusers library)

    1. Virtual environment for diffusion.ipynb:
        * virtualenv diffusion_venv
        * source diffusion_venv/bin/activate
        * pip install -r diffusion_requirements.txt
        * deactivate
        * Change kernel to diffusion_venv for diffusion.ipynb (In VSCode the option is located in the upper right corner.)

    2. Adjust parameters in diffusion.ipynb:
        1. cardiac_phase - "ED" for end-distolic, "ES" for end-systolic
        2. encoded_output_dir - same path as in encode_decode.cfg
        3. denoised_output_dir - same path as in encode_decode.cfg

    3. Run all cells


6. Decode data from the latent space into mesh shapes using a decoder from Mesh VAE:
    1. source mesh_vae_venv/bin/activate
    2. python meshLDM/decode.py --conf encode_decode.cfg


7. The generated are located in the "decoded_output_dir" folder (e.g. data/decoded/)


## Calculate clinical metrics - LV mass and volume
Code in folder: clinical_metrics

1. Convert .ply -> .vtk -> .vtk polydata:
    1. Adjust parameter in convert_ply_to_vtk.py:
        * ply_dir - path to the decoded meshes
    2. python convert_ply_to_vtk.py
    3. matlab  -nosplash -nodesktop -r "run('convert_unstructured_to_polydata.m');"


2. Calculate left ventricule (LV) volume & mass:
    1. Adjust parameter in calculate_volume_mass.m
        * MeshSampleFile - choose reference mesh depending on ED or ES data
    2. matlab -nosplash -nodesktop -r "run('calculate_volume_mass.m');"
    3. 

## (Optional) Get latent space distribution
After VAE training and data encoding:
* Get mean and std of the latent space distribution: meshLDM/mean_std_training_data.py

## Acknowledgements
Parts of this code are based on software from other repositories. Please see the [Acknowledgements] (Acknowledgements.txt) file for more details.

## License
[MIT](LICENSE.txt)

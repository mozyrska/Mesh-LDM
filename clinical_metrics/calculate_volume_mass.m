% (c) Jorge Corral Acero
clc; clear all;
addpath('./src');

mesh_folder = './all_samples/vtk_polydata';
mesh_folder_ref = './references_meshes';

% Uncomment to use the ED mesh as reference
MeshSampleFile = sprintf( '%s/polydata_mean_mesh_ED.vtk', mesh_folder_ref);
% Uncomment to use the ES mesh as reference
% MeshSampleFile = sprintf( '%s/polydata_mean_mesh_ES.vtk', mesh_folder_ref);

[CleanMesh, CleanNodeIdx] = MeshTidy( read_VTK(MeshSampleFile) , 0);

vol_gen_vae = [ ];
mass_gen_vae = [ ];

for pat = 0:999

gen_vae.mesh = struct();
file_vae = sprintf( '%s/case_%d.vtk', mesh_folder, pat);
field = sprintf('M%d', pat);
gen_vae.mesh.(field) = read_VTK(file_vae);

%%%%%%%%%%%% VOLUME CALCULATION
% ENDO and EPI IDENTIFICATION
% Since the meshes have corresponding points and exactly the same
% triangulation thanks to the cleaning step, I just need to locate the epi
% and the endo nodes once.
[Etri, Bxyz, idx_extra_point] = reference_endo_mesh( CleanMesh );
opt.epi = 1; [epiEtri, epiBxyz, epi_idx_extra_point] = reference_endo_mesh( CleanMesh, opt );

% If this is not the case in your meshes, you would need to reidentify the
% endo and epi nodes for each of the patients.

% VOLUME CALCULATION
PatientID = sprintf('M%d', pat);

% Volumes endo
gen_vae.endoESV = mesh_volume_from_reference_endo( gen_vae.mesh.(PatientID), Etri, Bxyz, idx_extra_point )./1000; %mL
vol_gen_vae = [vol_gen_vae gen_vae.endoESV]

% Volumes epi
gen_vae.epiESV = mesh_volume_from_reference_endo( gen_vae.mesh.(PatientID), epiEtri, epiBxyz, idx_extra_point )./1000; %mL

% Myocardial mass
gen_vae.myoMass = (gen_vae.epiESV - gen_vae.endoESV) * 1.05;
mass_gen_vae = [mass_gen_vae gen_vae.myoMass]

end

printf("VAE vol:")
mean(vol_gen_vae)
std(vol_gen_vae)

printf("VAE mass:")
mean(mass_gen_vae)
std(mass_gen_vae)

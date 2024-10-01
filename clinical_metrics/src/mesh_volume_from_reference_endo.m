function V = mesh_volume_from_reference_endo( M, Etri, Bxyz, idx_extra_point );

%{
%%%%%%%%%%%%%%%%%%% MESH VOLUME CALCULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GET THE ENDO for each mesh and calcula the volume:
DESCRIPTION
Calculate the volume of a mesh given the nodes and triangulization of the
endo (previously calculated). If endo is not givem , it will be calculated.

INPUT:
(Compulsory)
M --> Mesh to calc volume

(Optional) -> If not given, "reference_endo_mesh" will be calles
idx_extra_point --> Index of extrapoint (large number not ot overwride data)
Etri --> triangulization of the endo
Bxyz --> id of the points in the boundary
idx_extra_point --> Index of extrapoint to close the mesh

OUTPUT:
V --> Volume of the mesh

Etri --> triangulization of the endo (endo + boundaries to the extra point to close the mesh)
Bxyz --> id of the points in the boundary
idx_extra_point --> Index of extrapoint to close the mesh

%}


%% DEFAULT Values
if nargin<4;  [Etri, Bxyz, idx_extra_point] = reference_endo_mesh( M ); end;


%% FUNCTION
% Initial assignement
ENDO = M;

% Close the mesh based on the mean of the points in the boundary
ENDO.xyz(idx_extra_point,:) = mean( ENDO.xyz( Bxyz ,:) ,1);

% Add new triangularization (the endo one)
ENDO.tri = Etri;

% Calculate volume
V = meshVolume( MeshFixCellOrientation( MeshTidy( ENDO ,0) ) );

end
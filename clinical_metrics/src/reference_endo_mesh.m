function [Etri, Bxyz, idx_extra_point] = reference_endo_mesh( RM0, opt );

%{
%%%%%%%%%%%%%%%%%%% MESH VOLUME CALCULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFINE REFERENCE ENDO mesh
DESCRIPTION
As there is point correspondance, we can identify the endo in one
reference mesh and this will apply to the rest of them

INPUT:
(Compulsory)
RM0 --> Reference Mesh

(Optional)
idx_extra_point --> Index of extrapoint (large number not ot overwride data)


OUTPUT:
Output the final triangulization of the endo (endo + boundaries to the
extra point to close the mesh), the id of the points in the boundary to
later on calculate the middle point for each case and the idx of this
extra point

Etri --> triangulization of the endo (endo + boundaries to the extra point to close the mesh)
Bxyz --> id of the points in the boundary
idx_extra_point --> Index of extrapoint to clos the mesh

%}


%% DEFAULT Values
% high number not to overwrite any existing point
if nargin<2 ;  opt = struct();  end;

    % Options
    if ~isfield(opt,'epi');                 opt.epi                 = 0;            end; 
    if ~isfield(opt,'idx_extra_point');     opt.idx_extra_point     = 50000;        end; 


%% FUNCTION
%  Add point IDs to track originals
RM0 = MeshGenerateIDs( RM0 , 'xyz' );

% Split mesh into ENDO, EPI and MYO by identifiying regions whose change in
% angle is > 40
RM = MeshSplit( RM0 , -40 );
RM = meshSeparate( RM );

% Get the endo (or epi) and the corresponding IDs with respect to the original Ref Mesh (RM0)
E = RM{1}; % For endo
if opt.epi, E = RM{2}; end % for epi
Etri = E.xyzID( E.tri );

% Get the boundaries (points and triangles) and clean them:
B = MeshTidy(  MeshBoundary( E ) ,0);
Btri = B.xyzID( B.tri );
Bxyz = unique( Btri(:,1:2) );

% Calculate the point in the middle of the boundary (idx_extra_point) and 
% connect all the points in the boundary to that node to close the mesh
c = mean( RM0.xyz( Bxyz ,:) ,1);
idx_extra_point = opt.idx_extra_point;
RM0.xyz( idx_extra_point ,:) = c;
Btri(:,3) = idx_extra_point;

% Output the final triangulization of the endo (endo + boundaries to the
% extra point to close the mesh), the id of the points in the boundary to
% later on calculate the middle point foe each case and the idx of this
% extra point
Etri = [ Etri ; Btri ];

end
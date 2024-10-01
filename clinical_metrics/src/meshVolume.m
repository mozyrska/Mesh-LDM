function [v] = meshVolume( M , varargin )
% 
% M.xyz= randn(1000,3);
% M.xyz = bsxfun(@rdivide,M.xyz,sqrt(sum(M.xyz.^2,2)));
% M.tri = convhulln( M.xyz );
% MeshVolume( M ) / (4/3*pi)
% 

  M = Mesh( M ,0);

  N = cross( ( M.xyz(M.tri(:,2),:) - M.xyz(M.tri(:,1),:) ) , ( M.xyz(M.tri(:,3),:) - M.xyz(M.tri(:,2),:) ) , 2 );
  A = sqrt( sum( N.^2 , 2 ) );
  N = bsxfun( @rdivide , N , A );
  A = A/2;

  v =  sum(  ( M.xyz( M.tri(:,1),: ) + M.xyz( M.tri(:,2),: ) + M.xyz( M.tri(:,3),: ) ) .* N , 2 );
  v = 2*sum( v .* A )/( 3 * 6 );

end

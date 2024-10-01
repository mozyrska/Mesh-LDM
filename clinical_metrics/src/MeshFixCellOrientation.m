function [M,w] = MeshFixCellOrientation( M , f )


  if nargin < 2, f = []; end


  M0 = M;
  M = Mesh(M,0);
  M.triID = (1:size(M.tri,1)).';
  w = NaN( size( M.tri ,1) ,1);
  try,    M = meshSeparate( M );
  catch,  M = {M};
  end

  for m = 1:numel(M)
    w( M{m}.triID ) = fixTri( M{m} , find( ismember( M{m}.triID , f ) ) );
  end

  if any( isnan(w) ), error('there are triangles that were not fixed!'); end

  w = ~~w;

  M = M0;
  M.tri( w ,[2,3]) = M.tri( w ,[3,2]);

end


function W = fixTri( M , f )

  if isempty( f )
    CH = convhulln( M.xyz );
    CH = CH(:,[1,3,2]);
    
    [~,a,b] = intersect( sort(CH,2) , sort(M.tri,2) ,'rows' );
    [~,pa] = sort( CH(a,:) , 2);
    [~,pb] = sort( M.tri(b,:) , 2);

    f = [ -b( parity(pb) ~= parity(pa) ) ;...
           b( parity(pb) == parity(pa) ) ];

  end
  if isempty( f )
    error('not implemented yet.');
  end

  nF  = size( M.tri ,1);
  W = NaN( nF ,1);
  

  fp =  f( f > 0 );
  W( fp ) = 0;

  
  fn = -f( f < 0 );
  W( fn ) = 1;
  M.tri( fn ,[2,3]) = M.tri( fn ,[3,2]);

  E = [ M.tri(:,[1 2]) ; M.tri(:,[2 3]) ; M.tri(:,[3 1]) ];
  E = E(:,1) + 1i * E(:,2);
  
  IDS = repmat( (1:nF).' ,3,1);

  f = ismember( IDS , [ fp ; fn ] );
  EF = E(f); E(f) = []; IDS(f) = [];
  while ~isempty( E ) && ~isempty( EF )
    EF0 = EF; EF = [];
    
    f = IDS( ismember( E , imag(EF0) + 1i*real(EF0) ) );
    if any(f)
      f = ismember( IDS , f ) & isnan( W( IDS ) );
      W( IDS(f) ) = 0;
      EF = [ EF ; E(f) ]; E(f) = []; IDS(f) = [];
    end
    
    f = IDS( ismember( E , EF0 ) );
    if any(f)
      f = ismember( IDS , f ) & isnan( W( IDS ) );
      W( IDS(f) ) = 1;
      EF = [ EF ; imag( E(f)) + 1i*real( E(f)) ]; E(f) = []; IDS(f) = [];
    end
  end
  
end

function p = parity(P)

  n = size( P , 2);
  F = sort(P,2);
  F = bsxfun( @eq , F , 1:n );
  if ~all( F(:) )
    error('rows of P should be a permutation.');
  end

  p = false( size(P,1) ,1);
  switch n
    case 1

    case 2
      p( P(:,1) > P(:,2) ) = true;
      
    case 3
      p( ismember( P , [1,3,2;2,1,3;3,2,1] , 'rows' ) ) = true;

    case 4
      p( ismember( P , [1,2,4,3;1,3,2,4;1,4,3,2;2,1,3,4;2,3,4,1;2,4,1,3;3,1,4,2;3,2,1,4;3,4,2,1;4,1,2,3;4,2,3,1;4,3,1,2] , 'rows' ) ) = true;

    otherwise
%       p = 0;
%       for i = 1:(size(P,2) - 1)
%         p = p + sum( bsxfun( @lt , P(:,i) , P(:, (i+1):end ) ),2);
%       end
%       p = ~~mod(p,2);
      M = speye( n , n );
      for r = 1:size(P,1)
        p(r) = det( M( P(r,:) ,:) ) < 0;
      end
  end
end

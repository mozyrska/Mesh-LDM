function eE = subsasgn(eE,s,in)
% 
% 

  ntypes= numel(s);
  if ntypes > 2, error('Invalid Access (at 1).'); end
  
  switch s(1).type
    case '.'
      switch s(1).subs
        case {'callback_fcn'}
          eE.callback_fcn = in;
          setappdata(eE.panel,'eEntry', eE );

          
        case {'continuous' }
          drawnow;
          js = handle2javaobject( eE.slider );
          if ~iscell( js )
            js = {js};
          end
          for i=1:numel(js)
            try
              jsi = handle( js{i} , 'callbackproperties');
              if in
                jsi.AdjustmentValueChangedCallback = get(eE.slider,'Callback');
                set(eE.slider,'Callback','');
                set( eE.slider ,'UserData', jsi );
              else
                jsi.AdjustmentValueChangedCallback = '';
              end
            end
          end
        case {'value','v','Value','V'}
          value =in;
          value= min( [ value , eE.range(2) ] );
          value= max( [ value , eE.range(1) ] );
    
          try
            set( eE.slider ,'Value' , value ); 
            set( eE.edit   ,'String', eE.slider2edit_fcn(value) );
          end
          if ~isempty( eE.callback_fcn )
            eE.callback_fcn( eE.return_fcn( value ));
          end
        case {'v_no_callback'}
          value =in;
          value= min( [ value , eE.range(2) ] );
          value= max( [ value , eE.range(1) ] );
    
          try
            set( eE.slider ,'Value' , value ); 
            set( eE.edit   ,'String', eE.slider2edit_fcn(value) );
          end
        case {'range_no_callback'}
          eE.range(:) = in(:);
          if eE.range(2) > eE.range(1)
            set( eE.slider , 'Min', eE.range(1) , 'Max', eE.range(2) , 'Enable' , 'on' );
          else
            set( eE.slider , 'Enable' , 'off' )
          end
        case {'range'}
          eE.range(:) = in(:);
          if eE.range(2) > eE.range(1)
            set( eE.slider , 'Min', eE.range(1) , 'Max', eE.range(2) , 'Enable' , 'on' );
          else
            set( eE.slider , 'Enable' , 'off' )
          end
          updateeEntry(eE);
        case 'step'
          if eE.range(2) > eE.range(1)
            set( eE.slider , 'SliderStep', [1 10]*in/( eE.range(2) - eE.range(1) ) , 'Enable' , 'on' );
          else
            set( eE.slider , 'Enable' , 'off' )
          end
          
      end
      
    otherwise
      error('Invalid Access.');
  end
end

function JS = handle2javaobject( hg , salvar )

  if nargin < 2, salvar = 0; end

  hFig = ancestor( hg , 'figure' );
  
  jFigPanel = get( get( handle(hFig) , 'JavaFrame' ) ,'FigurePanelContainer' );
  jRootPane = jFigPanel.getComponent(0).getRootPane.getTopLevelAncestor.getComponent(0);
  
  REC2NUM = @(r) [ r.getX , r.getY , r.getWidth , r.getHeight ];
  
  JS = {};
  queue = { jRootPane , [0 0 0 0] };
  while ~isempty( queue )
    obj = queue{end,1};
    JS(end+1,:) = { obj , REC2NUM( obj.getBounds ) + queue{end,2} };
    queue(end,:) = [];
    
    try
      for c = 1:obj.getComponentCount
        child = obj.getComponent(c-1);
        if ~isempty( regexp( class(child) , '.*Menu.*', 'ONCE' ) ) , continue; end
        if isa(child,'com.mathworks.mwswing.desk.DTToolBarContainer') , continue; end
        queue(end+1,:) = { child , [ JS{end,2}(1:2) 0 0 ] };
      end
    end
  end

  switch get(hg,'Type')
    case 'uicontrol'
      switch get(hg,'Style')
        case 'slider',     JS = JS( cellfun(@(j) isa(j,'com.mathworks.hg.peer.SliderPeer$MLScrollBar'  ), JS(:,1) ) ,:);
        case 'pushbutton', JS = JS( cellfun(@(j) isa(j,'com.mathworks.hg.peer.PushButtonPeer$1'        ), JS(:,1) ) ,:);
        case 'edit',       JS = JS( cellfun(@(j) isa(j,'com.mathworks.hg.peer.EditTextPeer$hgTextField'), JS(:,1) ) ,:);
        case 'text',       JS = JS( cellfun(@(j) isa(j,'com.mathworks.hg.peer.LabelPeer$1'             ), JS(:,1) ) ,:);
      end
  end
  
  POS        = cell2mat( JS(:,2) );
  POS(:, 1 ) = POS(:, 1 ) - jRootPane.getX;
  POS(:, 2 ) = POS(:, 2 ) - jRootPane.getY;
  POS(:, 2 ) = jRootPane.getHeight - POS(:,2) - POS(:,4);

  p = getposition( hg , 'pixels' );

  diffpos = abs( bsxfun( @minus , POS , p ) );

  JS = JS( all( diffpos <= 2 , 2 ) , 1 );
  
  if numel( JS ) == 1
    try,
      JS = handle( JS{1} , 'CallbackProperties' );
    catch
      JS = JS{1};
    end
  else
    
  end
  
  if ~isempty( JS ) && salvar
    setappdata( hg , 'JavaObject', JS );
  end
  
end
function p= getposition( h , varargin )

  [varargin,units]= parseargs(varargin,'Pixels','pix','pixel' ,'$FORCE$',{'pixels'    ,0     } );
  [varargin,units]= parseargs(varargin,'Normalized','nor'     ,'$FORCE$',{'normalized',units } );
  
  [varargin,local]= parseargs(varargin,'local');
  
  if units
    for i=1:numel(h)
      oldU{i} = get(h(i),'Units');
      set(h(i),'Units',units);
    end
  end
  
  p= get( h ,'Position' );
  
  if units
    for i=1:numel(h)
      set(h(i),'Units',oldU{i});
    end
  end

  if ~local
    parent= get(h,'Parent');
    if ~strcmp( get(parent,'Type') , 'figure' )
      p(1:2) = p(1:2) + getposition( parent , 1:2 ,'pixels' );
    end
  end

  if numel(varargin)
    p = p( varargin{:} );
  end
  
end

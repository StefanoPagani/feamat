function [h]= plot_solution_on_fespace(fespace,vec,varargin)
% Plot finite element function as a surface (note: only value at vertices 
% of triangles are used for the visualization)
%
% input=
%           fespace: finite element space
%           vec: vector of degrees of freedom
%           (optional)
%           'contour' or 'contourf' to change visualization mode
%

n_vertices = size(fespace.mesh.vertices,1);
n1 = size(fespace.mesh.X,1);
n2 = size(fespace.mesh.X,2);

if (nargin == 2)
    [h] = surf(fespace.mesh.X,fespace.mesh.Y,reshape(vec(1:n_vertices),n1,n2),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
else
    if (strcmp(varargin{1},'contourf'))
        [h] = contourf(fespace.mesh.X,fespace.mesh.Y,reshape(vec(1:n_vertices),n1,n2),varargin{2});
    elseif (strcmp(varargin{1},'contour'))
        [c,h] = contour(fespace.mesh.X,fespace.mesh.Y,reshape(vec(1:n_vertices),n1,n2),varargin{2},'ShowText','on');
        clabel(c,h,'labelspacing', 1000,'Fontsize',12);
    else
        error('Visualization option is not supported');
    end
end


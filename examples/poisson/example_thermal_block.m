% example with a parametrized thermal block (sub-block conductivity
% parametrized \mu = [0.01,1]^9) 
% Problems specs in Manzoni, Pagani, Lassila "Accurate Solution of Bayesian
% Inverse Uncertainty Quantification Problems Combining Reduced Basis
% Methods and Reduction Error Models", SIAM JUQ, 2016
% Read More: https://epubs.siam.org/doi/10.1137/140995817"

%   Author: Stefano Pagani <stefano.pagani at polimi.it>

clear all
clc

% geometry  definition
bottom_left_corner_x = 0;
bottom_left_corner_y = 0;

L = 1.5;
H = 1.5;

% number of elements
n_elements_x = 30;
n_elements_y = 30;

mesh = create_mesh(bottom_left_corner_x, ...
                   bottom_left_corner_y, ...
                   L,H,n_elements_x,n_elements_y);

% boundary conditions                
bc_flags = [0 0 1 0];
dirichlet_functions = @(x) [0;0;0;0];
neumann_functions = @(x) [1;0;0;0];

% finite elements space definition
fespace = create_fespace(mesh,'P2',bc_flags);

% forcing term
f = @(x) 0*x(1,:);

% thermal block parameters
%param = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1 ];
param = 0.01 + 0.99*rand(9,1);

% discontinuos conductivity definition
mu = @(x) param(1)*(x(1,:)<0.5).*(x(2,:)<0.5) ...
   + param(2)*(x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=0.0).*(x(2,:)<0.5) ...
   + param(3)*(x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=0.0).*(x(2,:)<0.5) ...
   + param(4)*(x(1,:)>=0.0).*(x(1,:)<0.5).*(x(2,:)>=0.5).*(x(2,:)<1.0) ...
   + param(5)*(x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=0.5).*(x(2,:)<1.0) ...
   + param(6)*(x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=0.5).*(x(2,:)<1.0) ...
   + param(7)*(x(1,:)>=0.0).*(x(1,:)<0.5).*(x(2,:)>=1.0).*(x(2,:)<=1.5) ...
   + param(8)*(x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=1.0).*(x(2,:)<=1.5) ...
   + param(9)*(x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=1.0).*(x(2,:)<=1.5) ;

% Assembler
[A,b,uL,iN] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);


% Affine decomposition
mu = @(x) (x(1,:)<0.5).*(x(2,:)<0.5) ;
[A_in{1},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=0.0).*(x(2,:)<0.5);
[A_in{2},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=0.0).*(x(2,:)<0.5);
[A_in{3},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=0.0).*(x(1,:)<0.5).*(x(2,:)>=0.5).*(x(2,:)<1.0);
[A_in{4},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=0.5).*(x(2,:)<1.0) ;
[A_in{5},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=0.5).*(x(2,:)<1.0) ;
[A_in{6},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=0.0).*(x(1,:)<0.5).*(x(2,:)>=1.0).*(x(2,:)<=1.5);
[A_in{7},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=0.5).*(x(1,:)<1.0).*(x(2,:)>=1.0).*(x(2,:)<=1.5);
[A_in{8},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);

mu = @(x) (x(1,:)>=1.0).*(x(1,:)<=1.5).*(x(2,:)>=1.0).*(x(2,:)<=1.5);
[A_in{9},~,~,~] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);


% Solver
sol = uL; % lifting

A = param(1)*A_in{1};
for i=2:length(param)
    A = A + param(i)*A_in{i};
end

sol(iN)  = A\b;  % solution on the internal nodes


% plot of the solution
plot_fe_function(sol,fespace)
%axis equal
%export_vtk_scalar(sol,fespace,'example_thermal_block.vtk');
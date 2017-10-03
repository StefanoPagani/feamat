%% solving the poisson problem on two subdomains with FEM
%  The final solution is glued by imposing the continuity

clear all
clc

fun = @(x) sin(x(1))*cos(10*x(2))*exp(x(1))+log(x(2));
mu = pi;

% L2 error of full solution
disp('Computing exact solution');

mesh = create_mesh(0,0,1,1,100,100);

fespace = create_fespace(mesh,'P2',[1 1 1 1]);
[A,b] = assembler_poisson(fespace,fun,@(x) mu,@(x) [0;0;0;0],@(x) [0;0;0;0]);

totalsol = A\b;
%%

% create mesh on left subdomain
xp1 = 0;
yp1 = 0;
L1 = 0.5;
H1 = 1;

n1x = 30;
n2x = 30;
n1y = 30;
n2y = 30;

mesh1 = create_mesh(xp1,yp1,L1,H1,n1x,n1y);

% create mesh on right subdomain
xp2 = 0.5;
yp2 = 0;
L2 = 0.5;
H2 = 1;

mesh2 = create_mesh(xp2,yp2,L2,H2,n2x,n2y);

% draw global mesh
meshes = {};
meshes{end+1} = mesh1;
meshes{end+1} = mesh2;
%draw_multimesh(meshes);

% solve problems on subdomains
mu = 1;

fespace1 = create_fespace(mesh1,'P2',[1 0 1 1]);
fespace2 = create_fespace(mesh2,'P2',[1 1 1 0]);

[A1,b1] = assembler_poisson(fespace1,fun,@(x) mu,@(x) [0;0;0;0],@(x) [0;0;0;0]);
[A2,b2] = assembler_poisson(fespace2,fun,@(x) mu,@(x) [0;0;0;0],@(x) [0;0;0;0]);

n1 = size(A1,1);
n2 = size(A2,1);

indices1 = 1:n1;
indices2 = n1+1:n1+n2;

V1 = [];
V2 = [];

nmodes = 4;

for i = 1:nmodes
    disp(['Solving with mode with frequency omega * ',num2str(i)]);
    v1 = zeros(n1,1);
    v2 = zeros(n2,1);
    
    v1 = apply_neumann_bc(fespace1,v1,@(x) [0;sin(x(2)*pi*i);0;0]);
    v2 = apply_neumann_bc(fespace2,v2,@(x) [0;0;0;sin(x(2)*pi*i)]);
    v11 = apply_neumann_bc(fespace1,v1,@(x) [0;cos(x(2)*pi*i);0;0]);
    v22 = apply_neumann_bc(fespace2,v2,@(x) [0;0;0;cos(x(2)*pi*i)]);
    
    V1 = [V1 v1 v11];
    V2 = [V2 v2 v22];
    
    mat = [A1 zeros(n1,n2) -V1; zeros(n2,n1) A2 V2; -V1' V2' zeros(i*2,i*2)];
    
    mat(indices1,:) = apply_dirichlet_bc_matrix(mat(indices1,:),fespace1,1);
    mat(indices2,n1+1:end) = apply_dirichlet_bc_matrix(mat(indices2,n1+1:end),fespace2,1);
    
    rhs = [b1;b2;zeros(2*i,1)];
    
    sol = mat\rhs;
    
    sol1 = sol(indices1);
    sol2 = sol(indices2);
    
    plot_solution_on_fespace(fespace1,sol1)
    hold on

    plot_solution_on_fespace(fespace2,sol2)
    
    hold off
    
    interpsol = interp_2solutions_on_fespace(fespace1,sol1,fespace2,sol2,fespace);
    err = compute_error(fespace,abs(totalsol-interpsol),@(x)0,@(x)[0;0],'L2');
    disp(['Total L2 error = ', num2str(err)]);

    pause()
end




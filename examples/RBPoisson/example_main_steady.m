%  example of RB approximation of the thermal block problem

%  Author: Stefano Pagani <stefano.pagani at polimi.it>


%% step 1 : construct geometry, fe-space and arrays

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
[~,b,uL,iN] = assembler_poisson_lifting(fespace,f,mu,dirichlet_functions,neumann_functions);


% affine decomposition
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


%% step 2 : offline phase
% we construct the transformation matrix V and project the FE arrays

N_train = 100;
N_param = 9;

% training sample
rng('default')
param_train = 0.01 + 0.99*lhsdesign(N_train,N_param);

% matrix of the snapshots
S_u = [];

for i_s = 1:N_train
    
    [solFOM] = TBpoisson(param_train(i_s,:),A_in,b,uL,iN);
    
    S_u = [ S_u , solFOM(iN) ]; 
    
end

meanv = mean(S_u,2);

S_u_nomean = S_u  - meanv;


[V, ~, sigma] = POD(S_u_nomean,1e-2);
semilogy(sigma)

% projection
for i=1:length(A_in)
    
    A_in_ROM{i} = V'*A_in{i}*V;
    
    A_in_mean{i} = V'*A_in{i}*meanv;
    
end

b_ROM = V'*b;

% number of basis functions
n = size(V,2);


%% step 3 : online phase 
% we test the ROM over a test sample

% test sample
N_test = 50;
param_test = 0.01 + 0.99*lhsdesign(N_test,N_param);

for i_s = 1:N_test
    
    [solROM] = TBpoisson(param_test(i_s,:),A_in_ROM,b_ROM,zeros(n,1),[1:n],A_in_mean);
    
    [solFOM] = TBpoisson(param_test(i_s,:),A_in,b,uL,iN);
    
    err_u(i_s) = norm( solFOM(iN) - ( meanv + V*solROM )   )./norm( solFOM(iN)  );
    
    %full ROM solution
    uROM = uL;
    uROM(iN) =  V*solROM ;
    
end

mean(err_u)





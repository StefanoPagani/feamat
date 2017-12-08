% common variables
tol = 1e-2;

%% Test 1: verify order convergence against exact solution P1

solex = @(x) sin(pi*x(1)).*sin(pi*x(2));
gradex = @(x) [cos(pi*x(1)).*sin(pi*x(2));cos(pi*x(2)).*sin(pi*x(1))]*pi;

f = @(x) 2*pi^2*sin(pi*x(1,:)).*sin(pi*x(2,:));
mu = @(x) 1;
dirichlet_functions = @(x) [0;0;0;0];
neumann_functions = @(x) [-pi*sin(pi*x(1)).*cos(pi*x(2));
    pi*cos(pi*x(1)).*sin(pi*x(2));
    pi*sin(pi*x(1))*cos(pi*x(2));
    -pi*cos(pi*x(1)).*sin(pi*x(2))];

h = [];
l2errs = [];
h1errs = [];
for i = 4:5

    n1 = 5*2^(i-1);
    n2 = n1;

    h = [h;1/n1];

    mesh = create_mesh(0,0,1,1,n1,n2);
    fespace = create_fespace(mesh,'P1',[1 0 0 0]);

    [A,b] = assembler_poisson(fespace,f,mu,dirichlet_functions,neumann_functions);
    sol = A\b;

    l2error = compute_L2_error(fespace,sol,solex);
    h1error = compute_H1_error(fespace,sol,solex,gradex);

    l2errs = [l2errs;l2error];
    h1errs = [h1errs;h1error];
end

c_orderl2 = log(l2errs(2:end)./l2errs(1:end-1))./log(h(2:end)./h(1:end-1));
c_orderh1 = log(h1errs(2:end)./h1errs(1:end-1))./log(h(2:end)./h(1:end-1));

assert(abs(c_orderl2-2) < tol);
assert(abs(c_orderh1-1) < tol);

%% Test 1: verify order convergence against exact solution P2

solex = @(x) sin(pi*x(1)).*sin(pi*x(2));
gradex = @(x) [cos(pi*x(1)).*sin(pi*x(2));cos(pi*x(2)).*sin(pi*x(1))]*pi;

f = @(x) 2*pi^2*sin(pi*x(1,:)).*sin(pi*x(2,:));
mu = @(x) 1;
dirichlet_functions = @(x) [0;0;0;0];
neumann_functions = @(x) [-pi*sin(pi*x(1)).*cos(pi*x(2));
    pi*cos(pi*x(1)).*sin(pi*x(2));
    pi*sin(pi*x(1))*cos(pi*x(2));
    -pi*cos(pi*x(1)).*sin(pi*x(2))];

h = [];
l2errs = [];
h1errs = [];
for i = 4:5

    n1 = 5*2^(i-1);
    n2 = n1;

    h = [h;1/n1];

    mesh = create_mesh(0,0,1,1,n1,n2);
    fespace = create_fespace(mesh,'P2',[1 0 0 0]);

    [A,b] = assembler_poisson(fespace,f,mu,dirichlet_functions,neumann_functions);
    sol = A\b;

    l2error = compute_L2_error(fespace,sol,solex);
    h1error = compute_H1_error(fespace,sol,solex,gradex);

    l2errs = [l2errs;l2error];
    h1errs = [h1errs;h1error];
end

c_orderl2 = log(l2errs(2:end)./l2errs(1:end-1))./log(h(2:end)./h(1:end-1));
c_orderh1 = log(h1errs(2:end)./h1errs(1:end-1))./log(h(2:end)./h(1:end-1));

assert(abs(c_orderl2-3) < tol);
assert(abs(c_orderh1-2) < tol);

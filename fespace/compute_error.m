function error = compute_error(fespace,values,fexact,gradexact,type)

connectivity = fespace.connectivity;
vertices = fespace.mesh.vertices;

n_elements = size(connectivity,1);

[gp,weights,n_gauss] = gauss_points(2);

nlocalfunctions = fespace.n_functions_per_element;


if (type == 'L2')
    disp('Computing L2 error');
    tic 
    
    error = 0;
    for i = 1:n_elements
        indices = connectivity(i,1:3);
        x1 = vertices(indices(1),1:2)';
        x2 = vertices(indices(2),1:2)';
        x3 = vertices(indices(3),1:2)';

        mattransf = [x2-x1 x3-x1];

        % transformation from parametric to physical
        transf = @(x) mattransf*x + x1;       
        dettransf = abs(det(mattransf));
    
        for j = 1:n_gauss
            functions = fespace.functions(gp(:,j));
            value_in_gp = 0;
            for k = 1:nlocalfunctions
                value_in_gp = value_in_gp + values(indices(k))*functions(k);
            end
            error = error + dettransf*(value_in_gp-fexact(transf(gp(:,j))))^2*weights(j)/2;
        end
    end  
    error = sqrt(error);
    elapsed = toc;
    disp(['Elapsed time = ', num2str(elapsed),' s']);
    disp('------------------------------');
elseif (type == 'H1')
  disp('Computing H1 error');
    tic 
    
    error = 0;
    for i = 1:n_elements
        indices = connectivity(i,1:3);
        x1 = vertices(indices(1),1:2)';
        x2 = vertices(indices(2),1:2)';
        x3 = vertices(indices(3),1:2)';

        mattransf = [x2-x1 x3-x1];
        invmat = inv(mattransf);


        % transformation from parametric to physical
        transf = @(x) mattransf*x + x1;       
        dettransf = abs(det(mattransf));
    
        for j = 1:n_gauss
            functions = fespace.functions(gp(:,j));
            grads =  invmat'*fespace.grads(gp(:,j));
            value_in_gp_f = 0;
            value_in_gp_grad = [0;0];
            for k = 1:nlocalfunctions
                value_in_gp_f = value_in_gp_f + values(indices(k))*functions(k);
                value_in_gp_grad = value_in_gp_grad + values(indices(k))*grads(:,k);
            end
            gradex = gradexact(transf(gp(:,j)));
            error = error + dettransf*((value_in_gp_f-fexact(transf(gp(:,j))))^2 + (value_in_gp_grad(1)-gradex(1))^2 + (value_in_gp_grad(2)-gradex(2))^2)*weights(j)/2;
        end
    end  
    error = sqrt(error);
    elapsed = toc;
    disp(['Elapsed time = ', num2str(elapsed),' s']);
    disp('------------------------------');
else
    error([type,' error is not implemented!']);
end


end
